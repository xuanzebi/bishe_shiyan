import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from crf import CRF

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


class WordRep(nn.Module):
    """
    词向量：glove/字向量/elmo/bert/flair
    """
    def __init__(self, args, pretrain_word_embedding):
        super(WordRep, self).__init__()
        self.word_emb_dim = args.word_emb_dim
        self.char_emb_dim = args.char_emb_dim
        self.use_char = args.use_char
        self.use_pre = args.use_pre
        self.freeze = args.freeze
        self.drop = nn.Dropout(args.dropout)

        if self.use_pre:
            if self.freeze:
                self.word_embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrain_word_embedding),
                                                                   freeze=True).float()
            else:
                self.word_embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrain_word_embedding),
                                                                   freeze=False).float()
        else:
            self.word_embedding = nn.Embedding(args.vocab_size, 300)

        if self.use_char:
            pass

    def forward(self, word_input):
        word_embs = self.word_embedding(word_input)
        word_represent = self.drop(word_embs)
        return word_represent

class Bilstm(nn.Module):
    def __init__(self, args, pretrain_word_embedding, label_size):
        super(Bilstm, self).__init__()
        self.use_crf = args.use_crf
        self.use_char = args.use_char
        self.gpu = args.gpu
        self.use_pre = args.use_pre
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.rnn_type = args.rnn_type
        self.max_seq_length = args.max_seq_length
        self.use_highway = args.use_highway
        self.dropoutlstmnum = args.dropoutlstm
        self.dropoutlstm = nn.Dropout(args.dropoutlstm)
        self.wordrep = WordRep(args, pretrain_word_embedding)
        self.args = args
        self.word_emb_dim = args.word_emb_dim

        self.lstm = nn.LSTM(self.word_emb_dim, self.rnn_hidden_dim, num_layers=args.num_layers, batch_first=True,
                            bidirectional=True)
        self.gru = nn.GRU(self.word_emb_dim, self.rnn_hidden_dim, num_layers=args.num_layers, batch_first=True,
                          bidirectional=True)

        self.label_size = label_size
        if self.use_crf:
            self.crf = CRF(self.label_size, self.gpu)
            self.label_size += 2
        self.hidden2tag = nn.Linear(args.rnn_hidden_dim * 2, self.label_size)

    def forward(self, word_input, input_mask, labels,guids=None,mode=None):
        # word_input input_mask   FloatTensor
        word_input_id = self.wordrep(word_input)

        input_mask.requires_grad = False
        word_input_id = word_input_id * (input_mask.unsqueeze(-1).float())

        batch_size = word_input_id.size(0)
        if self.rnn_type == 'LSTM':
            output, _ = self.lstm(word_input_id)
        elif self.rnn_type == 'GRU':
            output, _ = self.gru(word_input_id)

        if self.use_highway:
            output = self.highway(output)

        if self.dropoutlstmnum != 0:
            output = self.dropoutlstm(output)
        output = self.hidden2tag(output)
        maskk = input_mask.ge(1)
        if self.use_crf:
            total_loss = self.crf.neg_log_likelihood_loss(output, maskk, labels)
            scores, tag_seq = self.crf._viterbi_decode(output, input_mask)
            return total_loss / batch_size, tag_seq
        else:
            # loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            loss_fct = nn.CrossEntropyLoss()
            active_loss = input_mask.view(-1) == 1
            active_logits = output.view(-1, self.label_size)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            return loss, output



def get_teacher_model(args,bert_model_path,model_save_dir,label2index,device):
    config = BertConfig.from_pretrained(bert_model_path, num_labels=len(label2index))
    entity_model_save_dir = model_save_dir + '/pytorch_model.bin'
    model = BertForTokenClassification.from_pretrained(entity_model_save_dir, config=config)
    model = model.to(device)
    model.load_state_dict(torch.load(entity_model_save_dir))
    for param in model.parameters():
        param.requires_grad = False
    if args.use_dataParallel:
        model = nn.DataParallel(model.cuda())
    return model
