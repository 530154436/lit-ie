export TASK_NAME=gplinker
export MODEL_NAME_OR_PATH=hfl/chinese-roberta-wwm-ext
export OUTPUT_DIR=outputs
export DATA_DIR=datasets/duee

python train.py \
    --task_name $TASK_NAME \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --model_type bert \
    --dataset_name $DATA_DIR \
    --train_file train.json \
    --validation_file dev.json \
    --train_max_length 128 \
    --cache_dir $DATA_DIR/$TASK_NAME \
    --preprocessing_num_workers 16 \
    --num_train_epochs 200 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --output_dir $OUTPUT_DIR/$TASK_NAME