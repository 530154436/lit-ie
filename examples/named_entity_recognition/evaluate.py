import argparse
import json

from litie.pipelines import NerPipeline
from litie.utils.logger import tqdm, logger

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


def evaluate():
    texts, entities = [], []
    with open(args.eval_file, "r") as f:
        for line in f:
            line = json.loads(line)
            texts.append(line["text"].strip())
            entities.append(set([(e["label"], e["start_offset"], e["end_offset"]) for e in line["entities"]]))

    pipeline = NerPipeline(
        task_model_name=args.task_model_name,
        model_type=args.model_type,
        model_name_or_path=args.model_name_or_path,
        batch_size=args.batch_size,
        use_fp16=args.use_fp16,
        max_seq_len=args.max_seq_len,
        device=args.device,
        schema2prompt=schema2prompt if args.with_mrc else None,
    )

    predictions = pipeline(texts)
    res = []
    for pred in predictions:
        res.append(set([(label, e["start"], e["end"]) for label, ents in pred.items() for e in ents]))

    X, Y, Z = 1e-10, 1e-10, 1e-10
    for R, T in tqdm(zip(entities, res), ncols=100):
        X += len(R & T)
        Y += len(R)
        Z += len(T)

    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z

    logger.info(f"f1 score: {f1}")
    logger.info(f"precision score: {precision}")
    logger.info(f"recall score: {recall}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model_name_or_path",
        type=str,
        required=True,
        help="The path prefix of inference model to be used.",
    )
    parser.add_argument(
        "-t",
        "--task_model_name",
        type=str,
        default="global_pointer",
        help="The task model to be used.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="bert",
        help="The model type to be used.",
    )
    parser.add_argument(
        "--with_mrc",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
    )
    parser.add_argument(
        "--use_fp16",
        action='store_true',
        help="Whether to use fp16 inference, only takes effect when deploying on gpu.",
    )
    parser.add_argument(
        "--max_seq_len",
        default=512,
        type=int,
        help="The maximum input sequence length. Sequences longer than this will be split automatically.",
    )
    parser.add_argument(
        "-D",
        "--device",
        choices=['cpu', 'cuda'],
        default="cpu",
        help="Select which device to run model, defaults to gpu."
    )

    args = parser.parse_args()

    evaluate()
