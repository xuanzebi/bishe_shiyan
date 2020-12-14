import json
import argparse
import torch
import os
import random
import numpy as np
import pickle

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def save_config(config, path, verbose=True):
    with open(path, 'w', encoding='utf-8') as outfile:
        json.dump(config, outfile, indent=2)
    if verbose:
        print("Config saved to file {}".format(path))
    return config

def seed_everything(SEED):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

# TODO 腾讯词向量 没有中文的， 《, 考虑将其换成中文的, 英文大写转小写
def build_vocab(data, min_count):
    """
        Return: vocab 词表各词出现的次数
                word2Idx 词表顺序
                label2index 标签对应序列
    """
    unk = '</UNK>'
    pad = '</PAD>'
    label2index = {}
    vocab = {}
    index = 1
    # label2index[pad] = 0
    label2index['O'] = 0

    word2Idx = {}

    for i, line in enumerate(data):
        text = line[0].split(' ')
        label = line[1].split(' ')
        assert len(text) == len(label)
        for te, la in zip(text, label):
            word = te.strip()
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1

            if la not in label2index:
                label2index[la] = index
                index += 1

    index2label = {j: i for i, j in label2index.items()}

    word2Idx[pad] = len(word2Idx)
    word2Idx[unk] = len(word2Idx)

    vocab = {i: j for i, j in vocab.items() if j >= min_count}

    for idx in vocab:
        if idx not in word2Idx:
            word2Idx[idx] = len(word2Idx)
    idx2word = {j: i for i, j in word2Idx.items()}

    return vocab, word2Idx, idx2word, label2index, index2label

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    with open(path) as f:
        next(f)
        return dict(get_coefs(*line.strip().split(' ')) for line in f)


def build_pretrain_embedding(args, word_index):
    word_dim = args.word_emb_dim
    # word_dim = 200
    # load = False
    embedding_matrix = np.zeros((len(word_index), word_dim))
    alphabet_size = len(word_index)
    scale = np.sqrt(3 / word_dim)
    # index:0 padding
    for index in range(1, len(word_index)):
        embedding_matrix[index, :] = np.random.uniform(-scale, scale, [1, word_dim])

    perfect_match = 0
    case_match = 0
    not_match = 0

    if args.pred_embed_path == None or args.pred_embed_path == '':
        print('================不加载词向量================')
        return embedding_matrix, 0
    else:

        if not args.load:
            print('===============加载预训练词向量===================')
            embedding_index = load_embeddings(args.pred_embed_path)
            # embedding_index,word_dim = load_pretrain_emb(embedding_path)
            unknown_words = []
            for word, i in word_index.items():
                if word in embedding_index:
                    embedding_matrix[i] = embedding_index[word]
                    perfect_match += 1
                elif word.lower() in embedding_index:
                    embedding_matrix[i] = embedding_index[word.lower()]
                    case_match += 1
                elif word.title() in embedding_index:
                    embedding_matrix[i] = embedding_index[word.title()]
                    case_match += 1
                else:
                    unknown_words.append(word)
                    not_match += 1

            unkword_set = set(unknown_words)
            pretrained_size = len(embedding_index)
            print("unk words 数量为{},unk 的word（set）数量为{}".format(len(unknown_words), len(unknown_words)))
            print("Embedding:\n     pretrain word:%s, vocab: %s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s" % (
                pretrained_size, alphabet_size, perfect_match, case_match, not_match, (not_match + 0.) / alphabet_size))
            if args.dump_embedding:
                pickle.dump(embedding_matrix, open(args.save_embed_path, 'wb'))  # cc.zh.300.vec
        else:
            print('===============加载事先保存好的预训练词向量===================')
            embedding_matrix = pickle.load(open(args.save_embed_path, 'rb'))  # cc.zh.300.vec

        return embedding_matrix

def get_cyber_data(data, args):
    vocab, word2idx, idx2word, label2index, index2label = build_vocab(data, args.min_count)
    pretrain_word_embedding = build_pretrain_embedding(args, word2idx)
    return pretrain_word_embedding, vocab, word2idx, idx2word, label2index, index2label


def pregress(data, word2idx, label2idx, max_seq_lenth):
    INPUT_ID = []
    INPUT_MASK = []
    LABEL_ID = []
    for text, label in data:
        input_mask = []
        input_id = []
        label_id = []
        text = text.split(' ')
        label = label.split(' ')
        for te, la in zip(text, label):
            te = te.strip()
            if te in word2idx:
                input_id.append(word2idx[te])
            else:
                input_id.append(word2idx['</UNK>'])
            label_id.append(label2idx[la])
            input_mask.append(1)

        if len(input_id) > max_seq_lenth:
            input_id = input_id[:max_seq_lenth]
            label_id = label_id[:max_seq_lenth]
            input_mask = input_mask[:max_seq_lenth]

        while len(input_id) < max_seq_lenth:
            input_id.append(0)
            label_id.append(0)
            input_mask.append(0)

        assert len(input_id) == len(label_id) == len(input_mask) == max_seq_lenth
        INPUT_ID.append(input_id)
        LABEL_ID.append(label_id)
        INPUT_MASK.append(input_mask)

    return INPUT_ID, INPUT_MASK, LABEL_ID


def covert_input_rnnids_bertids(rnnids,rnn_idx2word,bert_word2id):
    bert_ids,bert_maskids,bert_segmentids,bert_label_ids = [],[],[],[]
    rnnids = rnnids.cpu().numpy().tolist()
    for batch in rnnids:
        batch_ids,input_mask = [],[]

        # batch_ids.append(bert_word2id['[CLS]'])
        # input_mask.append(1)

        for ids in batch:
            if rnn_idx2word[ids] == '</UNK>':
                batch_ids.append(bert_word2id['[UNK]'])
            elif rnn_idx2word[ids] == '</PAD>':
                batch_ids.append(bert_word2id['[PAD]'])
            elif rnn_idx2word[ids] not in bert_word2id:
                batch_ids.append(bert_word2id['[UNK]'])
            else:
                batch_ids.append(bert_word2id[rnn_idx2word[ids]])

            if rnn_idx2word[ids] == '</PAD>':
                input_mask.append(0)
            else:
                input_mask.append(1)
            
        # batch_ids.append(bert_word2id['[SEP]'])
        # input_mask.append(1)

        segmentid = [0] * len(batch_ids)
        labelid = [0] * len(batch_ids)

        assert len(batch_ids) == len(input_mask) == len(segmentid) == len(labelid)

        bert_ids.append(batch_ids)
        bert_segmentids.append(segmentid)
        bert_label_ids.append(labelid)
        bert_maskids.append(input_mask)

    bert_ids = np.array(bert_ids)
    bert_segmentids = np.array(bert_segmentids)
    bert_maskids = np.array(bert_maskids)
    bert_label_ids = np.array(bert_label_ids)
    return bert_ids,bert_maskids,bert_segmentids,bert_label_ids

def get_bert_word2id(vocab_path):
    word2Idx = {}
    idx = 0
    with open(vocab_path,'r',encoding='utf-8') as fw:
        for line in fw:
            line = line.strip()
            word2Idx[line] = idx
            idx += 1
    
    return word2Idx