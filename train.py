import os
import sys

from transformers import HfArgumentParser

from litner.arguments import (
    DataTrainingArguments,
    ModelArguments,
    TrainingArguments,
)
from litner.models import AutoNerModel

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 1. create model
    model = AutoNerModel.create(model_args=model_args, training_args=training_args)

    # 2. finetune model
    model.finetune(data_args)


if __name__ == '__main__':
    main()
