### 主要参数

+ `task_name`: `ner` 模型名称，如 `crf`、`global_pointer`、`span`等


+ `model_name_or_path`: 开源模型的文件所在路径


+ `model_type`: 模型类型，默认值为 `bert`


+ `dataset_name`: `huggingface` 数据集名称或本地数据集文件所在路径


+ `train_file`: 训练集文件所在路径


+ `validation_file`: 验证集文件所在路径


+ `cache_dir`: 数据缓存路径


+ `preprocessing_num_workers`: 多进程处理数据


+ `num_train_epochs`: 训练轮次


+ `per_device_train_batch_size`: 训练集批量大小


+ `per_device_eval_batch_size`: 验证集批量大小


+ `learning_rate`: 学习率


+ `other_learning_rate`: 差分学习率


+ `output_dir`: 模型保存路径


### CRF

```shell
export TASK_NAME=crf
export MODEL_NAME_OR_PATH=hfl/chinese-roberta-wwm-ext
export OUTPUT_DIR=outputs
export DATA_DIR=datasets/cmeee

python train.py \
    --task_name $TASK_NAME \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --model_type bert \
    --dataset_name $DATA_DIR \
    --train_file train.json \
    --validation_file dev.json \
    --cache_dir $DATA_DIR/$TASK_NAME \
    --preprocessing_num_workers 16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --other_learning_rate 2e-3 \
    --output_dir $OUTPUT_DIR/$TASK_NAME
```

### SPAN

```shell
export TASK_NAME=span
export MODEL_NAME_OR_PATH=hfl/chinese-roberta-wwm-ext
export OUTPUT_DIR=outputs
export DATA_DIR=datasets/cmeee

python train.py \
    --task_name $TASK_NAME \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --model_type bert \
    --dataset_name $DATA_DIR \
    --train_file train.json \
    --validation_file dev.json \
    --cache_dir $DATA_DIR/$TASK_NAME \
    --preprocessing_num_workers 16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --other_learning_rate 2e-3 \
    --output_dir $OUTPUT_DIR/$TASK_NAME
```

### GLOBAL POINTER

```shell
export TASK_NAME=global_pointer
export MODEL_NAME_OR_PATH=hfl/chinese-roberta-wwm-ext
export OUTPUT_DIR=outputs
export DATA_DIR=datasets/cmeee

python train.py \
    --task_name $TASK_NAME \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --model_type bert \
    --dataset_name $DATA_DIR \
    --train_file train.json \
    --validation_file dev.json \
    --cache_dir $DATA_DIR/$TASK_NAME \
    --preprocessing_num_workers 16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --other_learning_rate 2e-4 \
    --output_dir $OUTPUT_DIR/$TASK_NAME
```

### LEAR

```shell
export TASK_NAME=lear
export MODEL_NAME_OR_PATH=hfl/chinese-roberta-wwm-ext
export OUTPUT_DIR=outputs
export DATA_DIR=datasets/cmeee

python train.py \
    --task_name $TASK_NAME \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --model_type bert \
    --dataset_name $DATA_DIR \
    --train_file train.json \
    --validation_file dev.json \
    --cache_dir $DATA_DIR/$TASK_NAME \
    --preprocessing_num_workers 16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --other_learning_rate 2e-4 \
    --output_dir $OUTPUT_DIR/$TASK_NAME
```

### MRC

```shell
export TASK_NAME=mrc
export MODEL_NAME_OR_PATH=hfl/chinese-roberta-wwm-ext
export OUTPUT_DIR=outputs
export DATA_DIR=datasets/cmeee

python train.py \
    --task_name $TASK_NAME \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --model_type bert \
    --dataset_name $DATA_DIR \
    --train_file train.json \
    --validation_file dev.json \
    --cache_dir $DATA_DIR/$TASK_NAME \
    --preprocessing_num_workers 16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --other_learning_rate 2e-4 \
    --output_dir $OUTPUT_DIR/$TASK_NAME
```

### TPLINKER

```shell
export TASK_NAME=tplinker
export MODEL_NAME_OR_PATH=hfl/chinese-roberta-wwm-ext
export OUTPUT_DIR=outputs
export DATA_DIR=datasets/cmeee

python train.py \
    --task_name $TASK_NAME \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --model_type bert \
    --dataset_name $DATA_DIR \
    --train_file train.json \
    --validation_file dev.json \
    --cache_dir $DATA_DIR/$TASK_NAME \
    --preprocessing_num_workers 16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --output_dir $OUTPUT_DIR/$TASK_NAME
```

### W2NER

```shell
export TASK_NAME=w2ner
export MODEL_NAME_OR_PATH=hfl/chinese-roberta-wwm-ext
export OUTPUT_DIR=outputs
export DATA_DIR=datasets/cmeee

python train.py \
    --task_name $TASK_NAME \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --model_type bert \
    --dataset_name $DATA_DIR \
    --train_file train.json \
    --validation_file dev.json \
    --cache_dir $DATA_DIR/$TASK_NAME \
    --preprocessing_num_workers 16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --other_learning_rate 2e-4 \
    --output_dir $OUTPUT_DIR/$TASK_NAME
```

### CNN NER

需要安装额外的 `torch_scatter` 包

在 `https://pytorch-geometric.com/whl/` 中找到与 `torch` 版本对应的 `torch_scatter`，下载后使用 `pip` 安装到环境中

```python
import torch

print(torch.__version__)  # 1.12.0
print(torch.version.cuda)  # 11.3
```

```shell
# 以python=3.8, torch=1.12.0, cuda=11.3为例
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_scatter-2.1.0%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
```

```shell
export TASK_NAME=cnn
export MODEL_NAME_OR_PATH=hfl/chinese-roberta-wwm-ext
export OUTPUT_DIR=outputs
export DATA_DIR=datasets/cmeee

python train.py \
    --task_name $TASK_NAME \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --model_type bert \
    --dataset_name $DATA_DIR \
    --train_file train.json \
    --validation_file dev.json \
    --cache_dir $DATA_DIR/$TASK_NAME \
    --preprocessing_num_workers 16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --other_learning_rate 2e-3 \
    --output_dir $OUTPUT_DIR/$TASK_NAME
```

### 模型评估

```shell
python evaluate.py \
    --eval_file datasets/cmeee/dev.json \
    --task_model_name global_pointer \
    --model_name_or_path outputs/global_pointer/bert-global_pointer \
    --device cuda
```

### 模型结果

|       模型       |       backbone model        |   val_f1   | val_precision | val_recall | 
|:--------------:|:---------------------------:|:----------:|:-------------:|:----------:|
|      crf       | hfl/chinese-roberta-wwm-ext |   64.44%   |    64.03%     |   64.85%   | 
| global_pointer | hfl/chinese-roberta-wwm-ext |   65.83%   |    63.82%     | **67.96%** | 
|      lear      | hfl/chinese-roberta-wwm-ext |   64.38%   |    65.89%     |   62.94%   | 
|      mrc       | hfl/chinese-roberta-wwm-ext |   65.02%   |    64.01%     |   66.06%   | 
|      span      | hfl/chinese-roberta-wwm-ext |   64.43%   |    65.82%     |   63.09%   | 
|    tplinker    | hfl/chinese-roberta-wwm-ext |            |               |            | 
|     w2ner      | hfl/chinese-roberta-wwm-ext |   65.22%   |     66.04     |   64.43%   | 
|      cnn       | hfl/chinese-roberta-wwm-ext | **65.87%** |  **66.22%**   |   65.52%   | 
