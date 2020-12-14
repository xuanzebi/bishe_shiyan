import os
import sys
import argparse
import time
from collections import defaultdict
import random
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import codecs
import json
from sklearn.model_selection import StratifiedKFold, KFold

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    CamembertConfig,
    CamembertForTokenClassification,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForTokenClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizer,
    XLMRobertaConfig,
    XLMRobertaForTokenClassification,
    XLMRobertaTokenizer,
    get_linear_schedule_with_warmup,
)

package_dir = "/opt/hyp/NER/Cysecurity_pretrain/fine-tune"
sys.path.insert(0, package_dir)

import warnings
warnings.filterwarnings("ignore")

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from util import get_logger, compute_spans_bio, compute_spans_bieos, compute_instance_f1,compute_f1
from utils_ner import convert_examples_to_features, read_examples_from_file, get_labels

from dice_loss import SelfAdjDiceLoss
from distill_loss import kd_mse_loss,kd_ce_loss
from distill_util import str2bool,save_config,seed_everything,get_cyber_data,pregress,covert_input_rnnids_bertids,get_bert_word2id
from distill_model import Bilstm,get_teacher_model

def evaluate_entity(y_true, y_pred, tag):
    if tag == 'BIO':
        gold_sentences = [compute_spans_bio(i) for i in y_true]
        pred_sentences = [compute_spans_bio(i) for i in y_pred]
    elif tag == 'BIEOS':
        gold_sentences = [compute_spans_bieos(i) for i in y_true]
        pred_sentences = [compute_spans_bieos(i) for i in y_pred]
    metric = compute_f1(gold_sentences, pred_sentences)
    return metric

def evaluate(data, model, label_map, tag, args, train_logger, device, mode,epoch,raw_data):
    print("Evaluating on {} set...".format(mode))
    test_iterator = tqdm(data, desc="dev_test_interation")
    preds = None
    out_label_ids = None
    test_loss = 0.
    nb_eval_steps = 0

    for step, test_batch in enumerate(test_iterator):
        # print(len(test_batch))
        model.eval()
        _test_batch = tuple(t.to(device) for t in test_batch)
        with torch.no_grad():
            _batch = tuple(t.to(device) for t in test_batch)
            input_ids, input_mask, label_ids = _batch
            loss, logits = model(input_ids, input_mask, labels=label_ids)
        nb_eval_steps += 1
        if args.use_dataParallel:
            loss = torch.sum(loss)  # if DataParallel model.module
        test_loss += loss.item()

        if args.use_crf == False:
            logits = torch.argmax(F.log_softmax(logits, dim=-1), dim=-1)

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = label_ids.cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, label_ids.cpu().numpy(), axis=0)

        test_iterator.set_postfix(test_loss=loss.item())

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    raw_data_len = [len(line.split()) for line,la in raw_data]
    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if j < raw_data_len[i]:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    metric = evaluate_entity(out_label_list, preds_list, tag)
    metric['test_loss'] = test_loss / nb_eval_steps
    metric['epoch'] = epoch
    if mode == 'test':
        return metric, preds_list
    else:
        return metric

def train(teacher_model: list,student_model,train_dataloader, dev_dataloader, test_dataloader,args, device, tb_writer, label_map, tag, train_logger,
    dev_raw_data,test_raw_data):
    """
        teacher_model: 教师模型，放入列表，可以多个教师模型进行蒸馏 
        student_model: 蒸馏到学生的模型
    """

    train_loss_step = {}
    train_loss_epoch = {}
    dev_loss_epoch = {}

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(student_model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(student_model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=1e-8)

    if args.use_scheduler:
        decay_rate = 0.05
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 / (1 + decay_rate * epoch))
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

    dev_result,test_result = [],[]
    bestscore, best_epoch = -1, 0
    test_bestscore,test_best_epoch = -1,0

    tr_loss, logging_loss = 0.0, 0.0
    lr = defaultdict(list)
    global_step = 0
    tq = tqdm(range(args.num_train_epochs), desc="Epoch")

    for epoch in tq:
        avg_loss = 0.
        epoch_start_time = time.time()
        student_model.train()
        student_model.zero_grad()

        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):
            student_model.zero_grad()
            _batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, label_ids = _batch 
            loss, s_logits = student_model(input_ids, input_mask, label_ids)

            # 蒸馏
            bert_inputids,bert_maskids,bert_segmentids,bert_label_ids = covert_input_rnnids_bertids(input_ids,args.idx2word,args.bert_word2id)
            bert_inputids = torch.from_numpy(bert_inputids).to(device)
            bert_maskids = torch.from_numpy(bert_maskids).to(device)
            bert_segmentids = torch.from_numpy(bert_segmentids).to(device)
            bert_label_ids = torch.from_numpy(bert_label_ids).to(device)

            if len(teacher_model) > 0 :
                t_logits = torch.zeros_like(s_logits)
                for t_model in teacher_model:
                    t_logit = t_model(bert_inputids,bert_maskids,bert_segmentids)[0]
                    t_logits += t_logit
                t_logits = t_logits / len(teacher_model)
                if args.loss_func == 'mse':
                    loss_t_s = kd_mse_loss(s_logits,t_logits)
                elif args.loss_func == 'ce':
                    loss_t_s = kd_ce_loss(s_logits,t_logits)
                loss += loss_t_s * args.loss_func_alpha

            loss.backward()
            optimizer.step()
    
            if args.use_dataParallel:
                loss = torch.sum(loss)  # if DataParallel
            
            tr_loss += loss.item()
            avg_loss += loss.item() / len(train_dataloader)
            global_step += 1

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                if args.use_scheduler:
                    print('当前epoch {}, step{} 的学习率为{}'.format(epoch, step, scheduler.get_lr()[0]))
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    lr[epoch].append(scheduler.get_lr()[0])
                else:
                    for param_group in optimizer.param_groups:
                        lr[epoch].append(param_group['lr'])
                tb_writer.add_scalar('train_loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                train_loss_step[global_step] = (tr_loss - logging_loss) / args.logging_steps
                logging_loss = tr_loss

            epoch_iterator.set_postfix(train_loss=loss.item())
        
        if args.use_scheduler:
            scheduler.step()

        tq.set_postfix(avg_loss=avg_loss)
        train_loss_epoch[epoch] = avg_loss

        metric = evaluate(dev_dataloader, student_model, label_map, tag, args, train_logger, device, 'dev',epoch,dev_raw_data)
        test_metric = evaluate(test_dataloader, student_model, label_map, tag, args, train_logger, device,
                                    'dev',epoch,test_raw_data)

        metric['epoch'] = epoch
        dev_loss_epoch[epoch] = metric['test_loss']
        tb_writer.add_scalar('test_loss', metric['test_loss'], epoch)

        if metric['micro-f1'] > bestscore:
            bestscore = metric['micro-f1']
            best_epoch = epoch
            print('实体级别的F1的best model epoch is: %d' % epoch)
            train_logger.info('实体级别的F1的best model epoch is: %d' % epoch)
        
        if test_metric['micro-f1'] > test_bestscore:
            test_bestscore = test_metric['micro-f1']
            test_best_epoch = epoch
            model_name = args.model_save_dir + "/entity_best.pt"
            model_to_save = (student_model.module if hasattr(student_model, "module") else student_model)
            torch.save(model_to_save.state_dict(), model_name)

        print('epoch:{} , global_step:{}, train_loss:{}, train_avg_loss:{}, dev_avg_loss:{},epoch耗时:{}!'.format(epoch, global_step, 
                                                    tr_loss / global_step,avg_loss, metric['test_loss'],time.time()-epoch_start_time))
        train_logger.info('epoch:{}, global_step:{}, train_loss:{},train_avg_loss:{},dev_avg_loss:{},epoch耗时:{}!'.format(epoch, global_step, 
                                                     tr_loss / global_step,avg_loss,metric['test_loss'],time.time()-epoch_start_time))
        print('epoch:{} P:{}, R:{}, F1:{} ,best F1:{}!"\n"'.format(epoch, metric['precision-overall'],metric['recall-overall'],
                                                                         metric['f1-measure-overall'], bestscore))
        train_logger.info('epoch:{} P:{}, R:{}, F1:{},best F1:{}! "\n"'.format(epoch, metric['precision-overall'], metric['recall-overall'],
                                                                     metric['f1-measure-overall'], bestscore))

        print('Test: epoch:{} P:{}, R:{}, F1:{},best F1:{}!!"\n"'.format(epoch, test_metric['precision-overall'],test_metric['recall-overall'],
                                                                         test_metric['f1-measure-overall'],test_bestscore))
        train_logger.info('Test: epoch:{} P:{}, R:{}, F1:{},best F1:{}!! "\n"'.format(epoch, test_metric['precision-overall'], test_metric['recall-overall'],
                                                                     test_metric['f1-measure-overall'],test_bestscore))                                                            

        dev_result.append(metric)
        test_result.append(test_metric)

    dev_result.append({'best_dev_f1': bestscore,
                        'best_dev_epoch': best_epoch})
    test_result.append({'dev_best_epoch': best_epoch,
                        'best_test_f1': test_bestscore,
                        'best_test_epoch': test_best_epoch})

    tb_writer.close()
    return test_result, dev_result, lr, train_loss_step, train_loss_epoch,dev_loss_epoch

if __name__ == "__main__":
    print(os.getcwd())
    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", default=True, type=str2bool, help="Whether to run training.")
    parser.add_argument("--do_test", default=True, type=str2bool, help="Whether to run test on the test set.")
    parser.add_argument('--model_save_dir', type=str, default='/opt/hyp/NER/Cysecurity_pretrain/fine-tune/distill_save_models/test/',
                        help='Root dir for saving models.')
    parser.add_argument('--data_path', default='/opt/hyp/NER/NER-model/data/Cybersecurity/json_data', type=str,help='数据路径')
    parser.add_argument('--pred_embed_path', default='/opt/hyp/NER/embedding/Tencent_AILab_ChineseEmbedding.txt', type=str,
                        help="预训练词向量路径,'cc.zh.300.vec','sgns.baidubaike.bigram-char','Tencent_AILab_ChineseEmbedding.txt'")
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'], type=str)
    parser.add_argument('--rnn_type', default='LSTM', type=str, help='LSTM/GRU')
    parser.add_argument('--save_embed_path',default='/opt/hyp/NER/NER-model/data/embedding/Tencent_AILab_ChineseEmbedding_cyber.p', type=str,
                        help='词向量存储路径,如果load为True，则加载该词向量')

    # parser.add_argument('--data_type', default='conll', help='数据类型 -conll - cyber')
    parser.add_argument("--use_bieos", default=True, type=str2bool, help="True:BIEOS False:BIO")
    parser.add_argument('--token_level_f1', default=False, type=str2bool, help='Sequence max_length.')
    parser.add_argument('--do_lower_case', default=False, type=str2bool, help='False 不计算token-level f1，true 计算')
    parser.add_argument('--freeze', default=False, type=str2bool, help='是否冻结词向量')
    parser.add_argument('--use_crf', default=False, type=str2bool, help='是否使用crf')
    parser.add_argument('--gpu', default=torch.cuda.is_available(), type=str2bool)
    parser.add_argument('--use_number_norm', default=False, type=str2bool)
    parser.add_argument('--use_pre', default=True, type=str2bool, help='是否使用预训练的词向量')
    parser.add_argument('--use_dataParallel', default=False, type=str2bool, help='是否使用dataParallel并行训练')
    parser.add_argument('--use_char', default=False, type=str2bool, help='是否使用char向量')
    parser.add_argument('--use_scheduler', default=True, type=str2bool, help='学习率是否下降')
    parser.add_argument('--load', default=True, type=str2bool, help='是否加载事先保存好的词向量')
    parser.add_argument('--use_highway', default=False, type=str2bool)
    parser.add_argument('--dump_embedding', default=True, type=str2bool, help='是否保存词向量')
    parser.add_argument('--use_packpad', default=False, type=str2bool, help='是否使用packed_pad')
    parser.add_argument('--loss_func', default='mse', type=str, help='mse/ce')
    parser.add_argument('--loss_func_alpha', default=1, type=float, help='蒸馏loss的权重')
    parser.add_argument('--use_distill', default=True, type=str2bool, help='是否使用蒸馏')

    parser.add_argument("--learning_rate", default=0.015, type=float, help="The initial learning rate for Adam.")
    parser.add_argument('--char_emb_dim', default=30, type=int)
    parser.add_argument("--num_train_epochs", default=5, type=int, help="Total number of training epochs to perform.")
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size.')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_seq_length', default=200, type=int, help='Sequence max_length.')
    parser.add_argument('--logging_steps', default=50, type=int)
    parser.add_argument('--word_emb_dim', default=200, type=int, help='预训练词向量的维度')
    parser.add_argument('--rnn_hidden_dim', default=128, type=int, help='rnn的隐状态的大小')
    parser.add_argument('--num_layers', default=1, type=int, help='rnn中的层数')
    parser.add_argument('--lr_decay', default=0.05, type=float)
    parser.add_argument('--momentum', default=0.9, type=float, help="0 or 0.9")
    parser.add_argument('--min_count', default=1, type=int)
    parser.add_argument('--dropout', default=0.2, type=float, help='词向量后的dropout')
    parser.add_argument('--dropoutlstm', default=0.2, type=float, help='lstm后的dropout')

    args = parser.parse_args()
    args.tensorboard_dir = args.model_save_dir + '/runs/'

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    if args.do_train:
        if os.path.exists(args.model_save_dir):
            for root, dirs, files in os.walk(args.model_save_dir):
                for sub_dir in dirs:
                    for sub_root, sub_di, sub_files in os.walk(os.path.join(root,sub_dir)):
                        for sub_file in sub_files:
                            os.remove(os.path.join(sub_root,sub_file))
                for envent_file in files:
                    os.remove(os.path.join(root,envent_file))

    result_dir = args.model_save_dir + '/result'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    if not os.path.exists(args.tensorboard_dir):
        os.makedirs(args.tensorboard_dir)

    print(args)
    train_logger = get_logger(args.model_save_dir + '/train_log.log')
    train_logger.info('各参数数值为{}'.format(args))
    start_time = time.time()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_bieos == True:
        tag = 'BIEOS'
    else:
        tag = 'BIO'
    
    train_data_raw = json.load(open(args.data_path + '/train_data.json', encoding='utf-8'))
    test_data_raw = json.load(open(args.data_path + '/test_data.json', encoding='utf-8'))
    dev_data_raw = json.load(open(args.data_path + '/dev_data.json', encoding='utf-8'))
    train_logger.info('训练集大小为{}，验证集大小为{}，测试集大小为{}'.format(len(train_data_raw), len(dev_data_raw), len(test_data_raw)))

    new_data = []
    new_data.extend(train_data_raw)
    new_data.extend(test_data_raw)
    new_data.extend(dev_data_raw)
    pretrain_word_embedding, vocab, word2idx, idx2word, label2index, index2label = get_cyber_data(new_data, args)

    args.label = label2index
    args.idx2word = idx2word
    args.vocab_size = len(vocab)
    args.bert_word2id = get_bert_word2id('/opt/hyp/NER/Cysecurity_pretrain/mlm_dapt/save_model/final_mlm_data_epoch_10_LR5e-5_BATCH_SIZE8_GAS8/vocab.txt')

    print("Let's use", torch.cuda.device_count(), "GPUs!")
    train_logger.info("Let's use{}GPUS".format(torch.cuda.device_count()))

    tb_writer = SummaryWriter(args.tensorboard_dir)

    if args.do_train:
        train_data_id, train_mask_id, train_label_id = pregress(train_data_raw, word2idx, label2index,max_seq_lenth=args.max_seq_length)
        train_data = torch.tensor([f for f in train_data_id], dtype=torch.long)
        train_mask = torch.tensor([f for f in train_mask_id], dtype=torch.long)
        train_label = torch.tensor([f for f in train_label_id], dtype=torch.long)
        train_dataset = TensorDataset(train_data, train_mask, train_label)

        dev_data, dev_mask, dev_label = pregress(dev_data_raw, word2idx, label2index, max_seq_lenth=args.max_seq_length)
        dev_data = torch.tensor([f for f in dev_data], dtype=torch.long)
        dev_mask = torch.tensor([f for f in dev_mask], dtype=torch.long)
        dev_label = torch.tensor([f for f in dev_label], dtype=torch.long)
        dev_dataset = TensorDataset(dev_data, dev_mask, dev_label)

        test_data, test_mask, test_label = pregress(test_data_raw, word2idx, label2index, max_seq_lenth=args.max_seq_length)
        test_data = torch.tensor([f for f in test_data], dtype=torch.long)
        test_mask = torch.tensor([f for f in test_mask], dtype=torch.long)
        test_label = torch.tensor([f for f in test_label], dtype=torch.long)
        test_dataset = TensorDataset(test_data, test_mask, test_label)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size)

        student_model = Bilstm(args, pretrain_word_embedding, len(label2index))
        if args.use_dataParallel:
            student_model = nn.DataParallel(student_model.cuda())
        student_model = student_model.to(device)

        teacher_model = []
        if args.use_distill:
            t_model = get_teacher_model(args,'/opt/hyp/NER/embedding/bert/chinese_L-12_H-768_A-12_pytorch',
                    '/opt/hyp/NER/Cysecurity_pretrain/fine-tune/save_models/cyber/mlm_final_epoch_cys_epoch50_LR5e-5_BATCH_SIZE16',
                    args.label,device)
            teacher_model.append(t_model)
        print('===============================开始训练================================')
        test_result,dev_result, lr, train_loss_step, train_loss_epoch, dev_loss_epoch = train(teacher_model,student_model, train_dataloader, 
                                                    dev_dataloader, test_dataloader,args, device, tb_writer, index2label, tag, train_logger, 
                                                    dev_data_raw,test_data_raw)
        
        with codecs.open(result_dir + '/dev_result.txt', 'w', encoding='utf-8') as f:
            json.dump(dev_result, f, indent=4, ensure_ascii=False)

        with codecs.open(result_dir + '/test_result.txt', 'w', encoding='utf-8') as f:
            json.dump(test_result, f, indent=4, ensure_ascii=False)
        
        with codecs.open(args.model_save_dir + '/learning_rate.txt', 'w', encoding='utf-8') as f:
            json.dump(lr, f, indent=4, ensure_ascii=False)
        with codecs.open(args.model_save_dir + '/train_loss_step.txt', 'w', encoding='utf-8') as f:
            json.dump(train_loss_step, f, indent=4, ensure_ascii=False)
        with codecs.open(args.model_save_dir + '/train_loss_epoch.txt', 'w', encoding='utf-8') as f:
            json.dump(train_loss_epoch, f, indent=4, ensure_ascii=False)
        with codecs.open(args.model_save_dir + '/dev_loss_epoch.txt', 'w', encoding='utf-8') as f:
            json.dump(dev_loss_epoch, f, indent=4, ensure_ascii=False)
        
        print(time.time() - start_time)
        opt = vars(args)  # dict
        # save config
        opt["time'min"] = (time.time() - start_time) / 60
        save_config(opt, args.model_save_dir + '/args_config.json', verbose=True)
        train_logger.info("Train Time cost:{}min".format((time.time() - start_time) / 60))

        del(student_model)
    
    if args.do_test:
        print('=========================测试集==========================')
        start_time = time.time()
        test_model = Bilstm(args, pretrain_word_embedding, len(label2index))
        if args.use_dataParallel:
            test_model = nn.DataParallel(test_model.cuda())
        test_model = test_model.to(device)
        entity_model_save_dir = args.model_save_dir + 'entity_best.pt'

        test_model.load_state_dict(torch.load(entity_model_save_dir))
        test_metric, y_pred = evaluate(test_dataloader, test_model, index2label, tag, args, train_logger, device, 'test',-1,test_data_raw)

        end_time = time.time()
        print('预测Time Cost:{}s'.format(end_time - start_time))
        train_logger.info('预测Time Cost:{}s'.format(end_time - start_time))

        with codecs.open(result_dir + '/test_result_entity.txt', 'w', encoding='utf-8') as f:
            json.dump(test_metric, f, indent=4, ensure_ascii=False)

        assert len(y_pred) == len(test_data_raw)
        results = []
        for i, (text, label) in enumerate(test_data_raw):
            res = []
            res.append(text)
            res.append(label)
            res.append(' '.join(y_pred[i]))
            results.append(res)

        with codecs.open(result_dir + '/test_pred_entity.txt', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)