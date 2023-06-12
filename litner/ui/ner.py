import time

import gradio as gr

from .utils import make_color_palette, visualize_ner
from ..pipelines import NerPipeline


def load_pipeline(model_path, model_type, model_name, device, max_seq_len, split_sentence=False, use_fp16=False):
    return NerPipeline(
        task_model_name=model_name,
        model_type=model_type,
        model_name_or_path=model_path,
        device=device,
        max_seq_len=max_seq_len,
        split_sentence=split_sentence,
        use_fp16=use_fp16,
    )


def extract(text, model_path, model_type, model_name, max_seq_len, device, split_sentence, use_fp16):
    # reset pipeline
    pipeline = load_pipeline(model_path, model_type, model_name, device, max_seq_len, split_sentence, use_fp16)

    start = time.time()
    res = pipeline(text)
    running_time = time.time() - start

    colors = make_color_palette(pipeline.inference_backend.model.config.label2id.keys())
    html = visualize_ner(text, res[0], colors)
    html = (
        ""
        + html
        + ""
    )

    return running_time, res, html


class NerPlayground:
    def __init__(self, server_name="0.0.0.0", server_port=7860, title=None, **kwargs):
        self.server_name = server_name
        self.server_port = server_port
        self.title = title
        self.kwargs = kwargs

        if self.title is None:
            self.title = "Named Entity Recognition for Chinese Medical Corpus"

    def launch(self) -> None:
        demo = gr.Interface(
            extract,
            [
                gr.Textbox(placeholder="Enter sentence here...", lines=5),
                gr.Textbox(placeholder="Enter model path here..."),
                gr.Radio(["bert", "ernie", "roformer", "nezha", "chinese-bert"], value="bert"),
                gr.Radio(["crf", "cascade_crf", "softmax", "span", "global_pointer",
                          "mrc", "tplinker", "lear", "w2ner", "cnn"], value="crf"),
                gr.Slider(0, 512, value=256),
                gr.Radio(["cpu", "cuda"], value="cpu"),
                gr.Checkbox(label="smart split sentence?"),
                gr.Checkbox(label="use fp16 speed strategy?"),
            ],
            [gr.Number(label="Run Time"), gr.Json(label="Result"), gr.HTML(label="Visualize")],
            examples=[
                ["可伴发血肿或脑梗死而出现局灶性神经体征，如肢体瘫痪及颅神经异常等。"],
                ["房室结消融和起搏器植入作为反复发作或难治性心房内折返性心动过速的替代疗法。"],
                ["如非肺炎病例，宜用宽大胶布条紧缠患部以减少其呼吸动作或给镇咳剂抑制咳嗽。"],
            ],
            title=self.title,
        )

        demo.launch(server_name=self.server_name, server_port=self.server_port, **self.kwargs)
