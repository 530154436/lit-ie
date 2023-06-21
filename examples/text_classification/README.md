### 主要参数

+ `task_name`: `ner` 模型名称，如 `tc` 等


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


### FC

```shell
export TASK_NAME=tc
export MODEL_NAME_OR_PATH=hfl/chinese-roberta-wwm-ext
export OUTPUT_DIR=outputs
export DATA_DIR=datasets/sentiment

python train.py \
    --task_name $TASK_NAME \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --model_type bert \
    --dataset_name $DATA_DIR \
    --train_file train.json \
    --validation_file dev.json \
    --cache_dir $DATA_DIR/$TASK_NAME \
    --preprocessing_num_workers 16 \
    --train_max_length 256 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --output_dir $OUTPUT_DIR/$TASK_NAME
```

### MDP

```shell
export TASK_NAME=tc
export MODEL_NAME_OR_PATH=hfl/chinese-roberta-wwm-ext
export OUTPUT_DIR=outputs
export DATA_DIR=datasets/sentiment

python train_mdp.py \
    --task_name $TASK_NAME \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --model_type bert \
    --dataset_name $DATA_DIR \
    --train_file train.json \
    --validation_file dev.json \
    --cache_dir $DATA_DIR/$TASK_NAME \
    --preprocessing_num_workers 16 \
    --train_max_length 256 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --output_dir $OUTPUT_DIR/$TASK_NAME
```

### 模型评估

```shell
python evaluate.py \
    --eval_file datasets/sentiment/dev.json \
    --task_model_name tc \
    --model_name_or_path outputs/tc/bert-tc \
    --device cuda
```

### 模型结果（DUEE部分数据）

|     模型      |       backbone model        | val_accuracy | val_precision | val_recall | 
|:-----------:|:---------------------------:|:------------:|:-------------:|:----------:|
|     fc      | hfl/chinese-roberta-wwm-ext |    95.53%    |    95.58%     |   95.53%   |
|     mdp     | hfl/chinese-roberta-wwm-ext |    96.19%    |    96.24%     |   96.19%   |
|  fc + fgm   | hfl/chinese-roberta-wwm-ext |  **96.34%**  |  **96.37%**   | **96.34%** |
| fc + r-drop | hfl/chinese-roberta-wwm-ext |    96.07%    |    96.07%     |   96.07%   |
