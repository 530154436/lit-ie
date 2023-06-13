import os
import sys

from transformers import HfArgumentParser

from litie.arguments import (
    DataTrainingArguments,
    ModelArguments,
    TrainingArguments,
)
from litie.models import AutoNerModel

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


schema2prompt = {
    "dis": "疾病，主要包括疾病、中毒或受伤和器官或细胞受损",
    "sym": "临床表现，主要包括症状和体征",
    "pro": "医疗程序，主要包括检查程序、治疗或预防程序",
    "equ": "医疗设备，主要包括检查设备和治疗设备",
    "dru": "药物，是用以预防、治疗及诊断疾病的物质",
    "ite": "医学检验项目，是取自人体的材料进行血液学、细胞学等方面的检验",
    "bod": "身体，主要包括身体物质和身体部位",
    "dep": "部门科室，医院的各职能科室",
    "mic": "微生物类，一般是指细菌、病毒、真菌、支原体、衣原体、螺旋体等八类微生物",
}


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 1. create model
    model = AutoNerModel(model_args=model_args, training_args=training_args)

    # 2. finetune model
    model.finetune(data_args, labels=schema2prompt)

    os.remove(os.path.join(training_args.output_dir, "best_model.ckpt"))


if __name__ == '__main__':
    main()
