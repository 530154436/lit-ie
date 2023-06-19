import time

import gradio as gr

from ..pipelines import EventExtractionPipeline


def load_pipeline(model_path, model_type, model_name, device, max_seq_len, use_fp16=False):
    return EventExtractionPipeline(
        task_model_name=model_name,
        model_type=model_type,
        model_name_or_path=model_path,
        device=device,
        max_seq_len=max_seq_len,
        use_fp16=use_fp16,
    )


def extract(text, model_path, model_type, model_name, max_seq_len, device, use_fp16):
    pipeline = load_pipeline(model_path, model_type, model_name, device, max_seq_len, use_fp16)
    start = time.time()
    res = pipeline(text)
    running_time = time.time() - start

    return running_time, res


class EventExtractionPlayground:
    def __init__(self, server_name="0.0.0.0", server_port=7860, title=None, **kwargs):
        self.server_name = server_name
        self.server_port = server_port

        self.title = title
        self.kwargs = kwargs

        if self.title is None:
            self.title = "Event Extraction Demo"

    def launch(self) -> None:
        demo = gr.Interface(
            extract,
            [
                gr.Textbox(
                    placeholder="Enter sentence here...",
                    lines=5
                ),
                gr.Textbox(
                    placeholder="Enter model path here..."
                ),
                gr.Radio(
                    ["bert", "ernie", "roformer", "nezha", "chinese-bert"],
                    value="bert"
                ),
                gr.Radio(
                    ["gplinker"],
                    value="gplinker"
                ),
                gr.Slider(
                    0,
                    512,
                    value=256,
                    interactive=True,
                ),
                gr.Radio(
                    ["cpu", "cuda"],
                    value="cpu",
                ),
                gr.Checkbox(
                    label="use fp16 speed strategy?"
                ),
            ],
            [gr.Number(label="Run Time"), gr.Json(label="Result")],
            examples=[
                ["油服巨头哈里伯顿裁员650人 因美国油气开采活动放缓"],
            ],
            title=self.title,
        )

        demo.launch(server_name=self.server_name, server_port=self.server_port, **self.kwargs)
