# 毕设相关实验源码

## 依赖库
1、transformers 2.4.1 

2、PyTorch >= 1.3.1

## 代码逻辑
- fine-tune/
    - cys_main.py -> bert for 安全NER训练代码
    - cys_main_Self_Distillation.py -> bert for 安全NER自蒸馏代码(sda/sdv)
    - distill_train.py -> bert蒸馏bilstm for安全NER代码
    - distillbert_train.py ->  bert蒸馏到k层bert代码