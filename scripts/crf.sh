export TASK_NAME=crf
export MODEL_NAME_OR_PATH=hfl/chinese-roberta-wwm-ext
export OUTPUT_DIR=outputs

python train.py \
    --task_name $TASK_NAME \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --model_type bert \
    --dataset_name xusenlin/cmeee \
    --cache_dir dataset/$TASK_NAME \
    --preprocessing_num_workers 16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --other_learning_rate 2e-3 \
    --output_dir $OUTPUT_DIR/$TASK_NAME
