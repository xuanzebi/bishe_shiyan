export CUDA_VISIBLE_DEVICES=1
# chinese_L-12_H-768_A-12_pytorch  chinese_roberta_wwm_ext_pytorch  RoBERTa_zh_L12_PyTorch
max_epoch=50 
LR=3e-5
BATCH_SIZE=16
sda_step_kd=2000
alpha=1

OUTPUT_DIR="/opt/hyp/NER/Cysecurity_pretrain/fine-tune/save_models/cyber/dice_loss_sda_step_mlm_cys_step${sda_step_kd}_alpha_1_epoch${max_epoch}_LR${LR}_BATCH_SIZE${BATCH_SIZE}/"

mkdir -p $OUTPUT_DIR

python ../cys_main_Self_Distillation.py  --model_save_dir=$OUTPUT_DIR \
                     --batch_size=$BATCH_SIZE \
                     --data_path='/opt/hyp/NER/NER-model/data/Cybersecurity/json_data' \
                     --model_name_or_path='/opt/hyp/NER/Cysecurity_pretrain/mlm_dapt/save_model/final_mlm_data_epoch_10_LR5e-5_BATCH_SIZE8_GAS8' \
                     --num_train_epochs=$max_epoch \
                     --do_train=True \
                     --do_test=True \
                     --seed=42 \
                     --max_seq_length=200 \
                     --use_bieos=True \
                     --learning_rate=$LR \
                     --use_dataParallel=False \
                     --use_crf=False \
                     --kd_type='sda_step' \
                     --use_dice_loss=True \
                     --sda_step_kd=$sda_step_kd \
                     --warmup_proportion=0.02 \
