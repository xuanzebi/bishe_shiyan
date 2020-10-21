SPAN_WEIGHT=0.1
DROPOUT=0.2
LR=5e-5
MAXLEN=230
accumulate_grad_batches=2
epochs=20

DATA_DIR="/opt/hyp/NER/Cysecurity_pretrain/mrc-for-flat-nested-ner/datasets/cysecurity" 
BERT_DIR="/opt/hyp/NER/embedding/bert/chinese_L-12_H-768_A-12_pytorch"
OUTPUT_DIR="/opt/hyp/NER/Cysecurity_pretrain/mrc-for-flat-nested-ner/result/epoch${epochs}_lr${LR}_dropout${DROPOUT}_acgb${accumulate_grad_batches}"

mkdir -p $OUTPUT_DIR

python ../trainer.py \
--data_dir $DATA_DIR \
--bert_config_dir $BERT_DIR \
--max_length $MAXLEN \
--batch_size 4 \
--gpus="0,1" \
--precision=16 \
--progress_bar_refresh_rate 1 \
--lr ${LR} \
--distributed_backend=ddp \
--val_check_interval 0.5 \
--accumulate_grad_batches $accumulate_grad_batches \
--default_root_dir $OUTPUT_DIR \
--mrc_dropout $DROPOUT \
--max_epochs $epochs \
--weight_span $SPAN_WEIGHT \
--span_loss_candidates "pred_and_gold" \
--warmup_steps 2000 \
--workers 16 \
