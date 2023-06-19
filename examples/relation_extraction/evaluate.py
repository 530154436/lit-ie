import argparse
import json

from litie.pipelines import RelationExtractionPipeline
from litie.utils.logger import tqdm, logger


def evaluate():
    texts, spoes = [], []
    with open(args.eval_file, "r") as f:
        for line in f:
            line = json.loads(line)
            texts.append(line["text"].strip())
            spoes.append(set([(spo["predicate"], spo["subject"], spo["object"]) for spo in line["spo_list"]]))

    pipeline = RelationExtractionPipeline(
        task_model_name=args.task_model_name,
        model_type=args.model_type,
        model_name_or_path=args.model_name_or_path,
        batch_size=args.batch_size,
        use_fp16=args.use_fp16,
        max_seq_len=args.max_seq_len,
        device=args.device,
    )

    predictions = pipeline(texts)
    res = []
    for pred in predictions:
        res.append(set([(label, e["subject"], e["object"]) for label, ents in pred.items() for e in ents]))

    X, Y, Z = 1e-10, 1e-10, 1e-10
    for R, T in tqdm(zip(spoes, res), ncols=100):
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
        default="gplinker",
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
