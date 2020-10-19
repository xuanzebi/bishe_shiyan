import argparse
import logging
import shutil
import os
import torch

from transformers import BertConfig, BertForPreTraining, load_tf_weights_in_bert

logging.basicConfig(level=logging.INFO)


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_bert(model, config, tf_checkpoint_path)

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


tf_checkpoint_file_path = '/opt/hyp/NER/embedding/bert/chinese_L-12_H-768_A-12'
pytorch_path = '/opt/hyp/NER/embedding/bert/chinese_L-12_H-768_A-12_pytorch'

if not os.path.exists(pytorch_path):
    os.makedirs(pytorch_path)

convert_tf_checkpoint_to_pytorch(
    os.path.join(tf_checkpoint_file_path, 'bert_model.ckpt'),
    os.path.join(tf_checkpoint_file_path, 'bert_config.json'),
    pytorch_path + '/pytorch_model.bin')
shutil.copyfile(os.path.join(tf_checkpoint_file_path, 'bert_config.json'), pytorch_path + '/config.json')
shutil.copyfile(os.path.join(tf_checkpoint_file_path, 'vocab.txt'), pytorch_path + '/vocab.txt')
