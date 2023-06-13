### UIE

首先将 `doccano` 格式的数据进行预处理

```shell
python doccano.py \
    --doccano_file datasets/12315/doccano.json \
    --save_dir datasets/12315
```

然后进行模型的微调

```shell
export TASK_NAME=uie
export MODEL_NAME_OR_PATH=uie_base_pytorch
export OUTPUT_DIR=outputs
export DATA_DIR=datasets/12315

python train.py \
    --task_name $TASK_NAME \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name $DATA_DIR \
    --train_file train.json \
    --validation_file dev.json \
    --cache_dir $DATA_DIR/$TASK_NAME \
    --preprocessing_num_workers 16 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --output_dir $OUTPUT_DIR/$TASK_NAME
```
