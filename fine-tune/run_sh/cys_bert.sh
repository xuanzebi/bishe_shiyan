export CUDA_VISIBLE_DEVICES=1
python ../cys_main.py  --model_save_dir='/opt/hyp/NER/Cysecurity_pretrain/fine-tune/save_models/cyber/robertcrf_result_bright/' \
                     --batch_size=16 \
                     --data_path='/opt/hyp/NER/NER-model/data/Cybersecurity/json_data' \
                     --model_name_or_path='/opt/hyp/NER/embedding/bert/RoBERTa_zh_L12_PyTorch' \
                     --num_train_epochs=10 \
                     --do_train=True \
                     --do_test=True \
                     --seed=42 \
                     --max_seq_length=200 \
                     --use_bieos=True \
                     --learning_rate=5e-5 \
                     --use_dataParallel=False \
                     --use_crf=True \
                     --warmup_proportion=0.1 \

