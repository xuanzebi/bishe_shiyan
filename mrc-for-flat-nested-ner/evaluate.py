# encoding: utf-8
import os
import sys
package_dir = "/opt/hyp/NER/Cysecurity_pretrain/pytorch-lightning-0.9.0"
sys.path.insert(0, package_dir)

from pytorch_lightning import Trainer
from trainer import BertLabeling

def evaluate(ckpt, hparams_file):
    """main"""
    trainer = Trainer(gpus=[1], distributed_backend="ddp")
    model = BertLabeling.load_from_checkpoint(
        checkpoint_path=ckpt,
        hparams_file=hparams_file,
        map_location=None,
        batch_size=4,
        max_length=230,
        workers=8
    )
    result = trainer.test(model=model)
    print(result[0]['span_precision'])
    print(result[0]['span_recall'])
    print(result[0]['span_f1'])
    


if __name__ == '__main__':

    CHECKPOINTS = "/opt/hyp/NER/Cysecurity_pretrain/mrc-for-flat-nested-ner/result/lr5e-5_dropout0.2_acgb2/epoch=9_v0.ckpt"
    HPARAMS = "/opt/hyp/NER/Cysecurity_pretrain/mrc-for-flat-nested-ner/result/lr5e-5_dropout0.2_acgb2/lightning_logs/version_0/hparams.yaml"

    evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS)


    # TODO  将val和test中的文本经过BERT分词器后 和输出的start_preds和end_preds对应上 提取出实体。s