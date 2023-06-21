import argparse
import json

from sklearn.metrics import classification_report

from litie.pipelines import TextClassificationPipeline
from litie.utils.logger import logger


def evaluate():
    texts, labels = [], []
    with open(args.eval_file, "r") as f:
        for line in f:
            line = json.loads(line)
            texts.append(line["text"].strip())
            labels.append(line["label"].strip())

    pipeline = TextClassificationPipeline(
        task_model_name=args.task_model_name,
        model_type=args.model_type,
        model_name_or_path=args.model_name_or_path,
        batch_size=args.batch_size,
        use_fp16=args.use_fp16,
        max_seq_len=args.max_seq_len,
        device=args.device,
    )

    predictions = pipeline(texts)

    logger.info(classification_report(labels, predictions))


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
        default="tc",
        help="The task model to be used.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="bert",
        help="The model type to be used.",
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
