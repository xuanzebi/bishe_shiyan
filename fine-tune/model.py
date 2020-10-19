import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertModel, BertPreTrainedModel
from crf import CRF

class BertCRFForNER(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.crf = CRF(config.num_labels, True)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels + 2)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        batch_size = input_ids.size(0)
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        attention_mask = attention_mask.ge(1)
        total_loss = self.crf.neg_log_likelihood_loss(logits, attention_mask, labels)
        scores, tag_seq = self.crf._viterbi_decode(logits, attention_mask)
        return total_loss / batch_size, tag_seq
