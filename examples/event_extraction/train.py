import json
import os
import sys

from transformers import HfArgumentParser

from litie.arguments import (
    DataTrainingArguments,
    ModelArguments,
    TrainingArguments,
)
from litie.models import AutoEventExtractionModel

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

schema_path = "datasets/duee/schema.json"

labels = []
with open("datasets/duee/schema.json") as f:
    for l in f:
        l = json.loads(l)
        t = l["event_type"]
        for r in ["触发词"] + [s["role"] for s in l["role_list"]]:
            labels.append(f"{t}+{r}")


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 1. create model
    model = AutoEventExtractionModel(model_args=model_args, training_args=training_args)

    # 2. finetune model
    model.finetune(data_args, labels=labels)

    os.remove(os.path.join(training_args.output_dir, "best_model.ckpt"))


if __name__ == '__main__':
    main()
