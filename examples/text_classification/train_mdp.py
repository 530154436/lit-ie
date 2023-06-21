import os
import sys

from transformers import HfArgumentParser

from litie.arguments import (
    DataTrainingArguments,
    ModelArguments,
    TrainingArguments,
)
from litie.models import AutoTextClassificationModel

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 1. create model
    model = AutoTextClassificationModel(
        model_config_kwargs={"use_mdp": True},
        model_args=model_args,
        training_args=training_args
    )

    # 2. finetune model
    model.finetune(data_args, monitor="val_accuracy")

    os.remove(os.path.join(training_args.output_dir, "best_model.ckpt"))


if __name__ == '__main__':
    main()
