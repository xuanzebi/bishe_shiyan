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

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from util import get_logger, compute_spans_bio, compute_spans_bieos, compute_instance_f1,compute_f1
from utils_ner import get_labels

from dice_loss import SelfAdjDiceLoss
from distill_loss import kd_mse_loss,kd_ce_loss
from distill_util import str2bool,save_config,seed_everything,get_cyber_data,get_bert_word2id,load_and_cache_examples
from distill_model import Bilstm,get_teacher_model,FewLayerBertForNER

def evaluate_entity(y_true, y_pred, tag):
    if tag == 'BIO':
        gold_sentences = [compute_spans_bio(i) for i in y_true]
        pred_sentences = [compute_spans_bio(i) for i in y_pred]
    elif tag == 'BIEOS':
        gold_sentences = [compute_spans_bieos(i) for i in y_true]
        pred_sentences = [compute_spans_bieos(i) for i in y_pred]
    metric = compute_f1(gold_sentences, pred_sentences)
    return metric

def evaluate(data, model, label_map, tag, args, train_logger, device, mode,epoch):
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

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    metric = evaluate_entity(out_label_list, preds_list, tag)
    metric['test_loss'] = test_loss / nb_eval_steps
    metric['epoch'] = epoch
    if mode == 'test':
        return metric, preds_list
    else:
        return metric

def train(teacher_model: list,student_model,train_dataloader, dev_dataloader, test_dataloader,args, device, tb_writer, label_map, tag, train_logger):
    """
        teacher_model: 教师模型，放入列表，可以多个教师模型进行蒸馏 
        student_model: 蒸馏到学生的模型
    """

    train_loss_step = {}
    train_loss_epoch = {}
    dev_loss_epoch = {}

    no_decay = ["bias", "LayerNorm.weight"]
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in student_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in student_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    warmup_steps = int(args.warmup_proportion * t_total)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

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
            input_ids, input_mask, segment_ids, label_ids = _batch
            loss, s_logits = student_model(input_ids, input_mask, segment_ids, labels=label_ids)
            custom_loss = loss.item()
            # 蒸馏
            if len(teacher_model) > 0 :
                t_logits = torch.zeros_like(s_logits)
                for t_model in teacher_model:
                    t_model = t_model.to(device)
                    t_logit = t_model(input_ids, input_mask, segment_ids, labels=label_ids)[1]
                    t_logits += t_logit
                t_logits = t_logits / len(teacher_model)
                if args.loss_func == 'mse':
                    loss_t_s = kd_mse_loss(s_logits,t_logits,temperature=args.temperature)
                elif args.loss_func == 'ce':
                    loss_t_s = kd_ce_loss(s_logits,t_logits,temperature=args.temperature)
                loss += loss_t_s * args.loss_func_alpha

            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)  # 梯度裁剪
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                student_model.zero_grad()
                global_step += 1
    
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

            if len(teacher_model) > 0 :
                epoch_iterator.set_postfix(train_loss=loss.item(),ditill_loss=loss_t_s.item(),custom_loss=custom_loss)
            else:
                epoch_iterator.set_postfix(train_loss=loss.item())
        
        tq.set_postfix(avg_loss=avg_loss)
        train_loss_epoch[epoch] = avg_loss

        metric = evaluate(dev_dataloader, student_model, label_map, tag, args, train_logger, device, 'dev',epoch)
        test_metric = evaluate(test_dataloader, student_model, label_map, tag, args, train_logger, device,
                                    'dev',epoch)

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
    parser.add_argument("--model_type", default='bert', type=str, help="Model type selected in the list")
    parser.add_argument('--model_save_dir', type=str, default='/opt/hyp/NER/Cysecurity_pretrain/fine-tune/distill_save_models/test/',
                        help='Root dir for saving models.')
    parser.add_argument('--teacher_model_path', type=str, 
                        default='/opt/hyp/NER/Cysecurity_pretrain/fine-tune/save_models/cyber/bert_cys_epoch30_LR3e-5_BATCH_SIZE16',
                        help='Root dir for saving models.')
    parser.add_argument('--data_path', default='/opt/hyp/NER/NER-model/data/Cybersecurity/json_data', type=str,help='数据路径')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'], type=str)
    parser.add_argument("--model_name_or_path",
                        default='/opt/hyp/NER/Cysecurity_pretrain/mlm_dapt/save_model/final_mlm_data_epoch_10_LR5e-5_BATCH_SIZE8_GAS8', type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: ")

    # parser.add_argument('--data_type', default='conll', help='数据类型 -conll - cyber')
    parser.add_argument("--use_bieos", default=True, type=str2bool, help="True:BIEOS False:BIO")
    parser.add_argument('--token_level_f1', default=False, type=str2bool, help='Sequence max_length.')
    parser.add_argument('--do_lower_case', default=False, type=str2bool, help='False 不计算token-level f1，true 计算')
    parser.add_argument('--use_crf', default=False, type=str2bool, help='是否使用crf')
    parser.add_argument('--gpu', default=torch.cuda.is_available(), type=str2bool)
    parser.add_argument('--use_number_norm', default=False, type=str2bool)
    parser.add_argument('--use_dataParallel', default=False, type=str2bool, help='是否使用dataParallel并行训练')
    parser.add_argument('--use_scheduler', default=True, type=str2bool, help='学习率是否下降')
    parser.add_argument('--use_highway', default=False, type=str2bool)
    parser.add_argument('--loss_func', default='mse', type=str, help='mse/ce')
    parser.add_argument('--loss_func_alpha', default=1, type=float, help='蒸馏loss的权重')
    parser.add_argument('--use_distill', default=True, type=str2bool, help='是否使用蒸馏')
    parser.add_argument('--few_bertlayer_num', default=3, type=int, help='BERT的层数')
    parser.add_argument('--temperature', default=1, type=int, help='蒸馏temperature')

    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=5, type=int, help="Total number of training epochs to perform.")
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size.')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_seq_length', default=200, type=int, help='Sequence max_length.')
    parser.add_argument('--logging_steps', default=50, type=int)
    parser.add_argument('--momentum', default=0.9, type=float, help="0 or 0.9")
    parser.add_argument('--min_count', default=1, type=int)
    parser.add_argument('--dropout', default=0.2, type=float, help='词向量后的dropout')
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Linear warmup over warmup_steps.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some. 0/0.01")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

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

    label2index = get_labels(new_data)
    print('该数据集的label为:',label2index)
    index2label = {j: i for i, j in label2index.items()}
    pad_token_label_id = CrossEntropyLoss().ignore_index

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    student_config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=len(label2index))
    config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=len(label2index))

    print("Let's use", torch.cuda.device_count(), "GPUs!")
    train_logger.info("Let's use{}GPUS".format(torch.cuda.device_count()))
    tb_writer = SummaryWriter(args.tensorboard_dir)

    if args.do_train:
        student_config.num_hidden_layers = args.few_bertlayer_num 
        student_model = FewLayerBertForNER.from_pretrained(args.model_name_or_path, config=student_config)
        if args.use_dataParallel:
            student_model = nn.DataParallel(student_model.cuda())
        student_model = student_model.to(device)
        param_optimizer = list(student_model.named_parameters())

        train_dataset = load_and_cache_examples(train_data_raw, args, tokenizer, label2index, pad_token_label_id, 'train', train_logger)
        dev_dataset = load_and_cache_examples(dev_data_raw, args, tokenizer, label2index, pad_token_label_id, 'dev',train_logger)
        test_dataset = load_and_cache_examples(test_data_raw, args, tokenizer, label2index, pad_token_label_id, 'test',train_logger)
        
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)


        teacher_model = []
        if args.use_distill:
            t_model = FewLayerBertForNER.from_pretrained(args.teacher_model_path, config=config)
            teacher_model.append(t_model)
        print('===============================开始训练================================')
        test_result,dev_result, lr, train_loss_step, train_loss_epoch, dev_loss_epoch = train(teacher_model,student_model, train_dataloader, 
                                                    dev_dataloader, test_dataloader,args, device, tb_writer, index2label, tag, train_logger)
        
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
        test_model = FewLayerBertForNER.from_pretrained(args.model_name_or_path, config=student_config)
        if args.use_dataParallel:
            test_model = nn.DataParallel(test_model.cuda())
        test_model = test_model.to(device)
        entity_model_save_dir = args.model_save_dir + '/pytorch_model.bin'

        test_model.load_state_dict(torch.load(entity_model_save_dir))
        for param in test_model.parameters():
            param.requires_grad = False
        test_metric, y_pred = evaluate(test_dataloader, test_model, index2label, tag, args, train_logger, device, 'test',-1)

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