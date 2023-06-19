import argparse
import json

from litie.pipelines import EventExtractionPipeline
from litie.utils.logger import tqdm, logger


class DedupList(list):
    """定义去重的list"""

    def append(self, x):
        if x not in self:
            super(DedupList, self).append(x)


def evaluate():
    texts, event_list = [], []
    with open(args.eval_file, "r") as f:
        for line in f:
            line = json.loads(line)
            texts.append(line["text"].strip())
            events = []
            for e in line["event_list"]:
                events.append([])
                for a in e["arguments"]:
                    events[-1].append(
                        (
                            e["event_type"],
                            a["role"],
                            a["argument"],
                        )
                    )
            event_list.append(events)

    pipeline = EventExtractionPipeline(
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
        events = []
        for p in pred:
            events.append([])
            for a in p["arguments"]:
                events[-1].append(
                    (
                        p["event_type"],
                        a["role"],
                        a["argument"],
                    )
                )
        res.append(events)

    ex, ey, ez = 1e-10, 1e-10, 1e-10  # 事件级别
    ax, ay, az = 1e-10, 1e-10, 1e-10  # 论元级别

    for events, pred_events in tqdm(zip(event_list, res), ncols=100):
        R, T = DedupList(), DedupList()
        # 事件级别
        for event in pred_events:
            R.append(list(sorted(event)))
        for event in events:
            T.append(list(sorted(event)))
        for event in R:
            if event in T:
                ex += 1

        ey += len(R)
        ez += len(T)

        # 论元级别
        R, T = DedupList(), DedupList()
        for event in pred_events:
            for argu in event:
                R.append(argu)
        for event in events:
            for argu in event:
                T.append(argu)
        for argu in R:
            if argu in T:
                ax += 1

        ay += len(R)
        az += len(T)

    e_f1, e_pr, e_rc = 2 * ex / (ey + ez), ex / ey, ex / ez
    a_f1, a_pr, a_rc = 2 * ax / (ay + az), ax / ay, ax / az

    logger.info("Event level metrics:")
    logger.info(f"f1 score: {e_f1}")
    logger.info(f"precision score: {e_pr}")
    logger.info(f"recall score: {e_rc}")

    logger.info("Argument level metrics:")
    logger.info(f"f1 score: {a_f1}")
    logger.info(f"precision score: {a_pr}")
    logger.info(f"recall score: {a_rc}")


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
