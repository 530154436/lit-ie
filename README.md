# Lit-NER

<p align="center">
    <a href="https://github.com/xusenlinzy/lit-ner"><img src="https://img.shields.io/github/license/xusenlinzy/lit-ner"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.8+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/pytorch-%3E=1.12-red?logo=pytorch"></a>
    <a href="https://github.com/xusenlinzy/lit-ner"><img src="https://img.shields.io/github/last-commit/xusenlinzy/lit-ner"></a>
    <a href="https://github.com/xusenlinzy/lit-ner"><img src="https://img.shields.io/github/issues/xusenlinzy/lit-ner?color=9cc"></a>
    <a href="https://github.com/xusenlinzy/lit-ner"><img src="https://img.shields.io/github/stars/xusenlinzy/lit-ner?color=ccf"></a>
    <a href="https://github.com/xusenlinzy/lit-ner"><img src="https://img.shields.io/badge/langurage-py-brightgreen?style=flat&color=blue"></a>
</p>

此项目为开源**命名实体识别**模型的训练和推理提供统一框架


## 📢 News 

+ 2023.6.12 使用 [m3e](https://huggingface.co/moka-ai/m3e-base) 中文嵌入模型（在中文文本分类和文本检索上都优于 `openai-ada-002`）


+ 2023.6.10 新增 [OpenBuddy](https://github.com/OpenBuddy/OpenBuddy) 仓库训练的中文模型

---

## 🔨 安装

1. 搭建好 `pytorch` 深度学习环境

```bash
conda create -n pytorch python=3.8
conda activate pytorch
conda install pytorch cudatoolkit -c pytorch
```

2. 安装 `litner`

```bash
pip install litner
```

3. 在 `https://pytorch-geometric.com/whl/` 中找到与 `torch` 版本对应的 `torch_scatter`，下载后使用 `pip` 安装到环境中 

```python
import torch

print(torch.__version__)  # 1.12.0
print(torch.version.cuda)  # 11.3
```

```bash
# 以python=3.8, torch=1.12.0, cuda=11.3为例
wget https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_scatter-2.1.0%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
pip install torch_scatter-2.1.0+pt112cu113-cp38-cp38-linux_x86_64.whl
```


## 🐼 模型

支持多种开源实体抽取模型

| 模型                                                | 论文                                                                                                                                                                            | 备注                                                                                                                                            |
|---------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| [softmax](litner/nn/ner/crf.py)                   |                                                                                                                                                                               | 全连接层序列标注并使用 `BIO` 解码                                                                                                                          |
| [crf](litner/nn/ner/crf.py)                       | [Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers) | 全连接层+条件随机场，并使用 `BIO` 解码                                                                                                                       |
| [cascade-crf](litner/nn/ner/crf.py)               |                                                                                                                                                                               | 先预测实体再预测实体类型                                                                                                                                  |
| [span](litner/nn/ner/span.py)                     |                                                                                                                                                                               | 使用两个指针网络预测实体起始位置                                                                                                                              |
| [global-pointer](litner/nn/ner/global_pointer.py) |                                                                                                                                                                               | [GlobalPointer：用统一的方式处理嵌套和非嵌套NER](https://spaces.ac.cn/archives/8373)、[Efficient GlobalPointer：少点参数，多点效果](https://spaces.ac.cn/archives/8877) |
| [mrc](litner/nn/ner/mrc.py)                       | [A Unified MRC Framework for Named Entity Recognition.](https://aclanthology.org/2020.acl-main.519.pdf)                                                                       | 将实体识别任务转换为阅读理解问题，输入为实体类型模板+句子，预测对应实体的起始位置                                                                                                     |
| [tplinker](litner/nn/ner/tplinker.py)             | [TPLinker: Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking.](https://aclanthology.org/2020.coling-main.138.pdf)                            | 将实体识别任务转换为表格填充问题                                                                                                                              |
| [lear](litner/nn/ner/lear.py)                     | [Enhanced Language Representation with Label Knowledge for Span Extraction.](https://aclanthology.org/2021.emnlp-main.379.pdf)                                                | 改进 `MRC` 方法效率问题，采用标签融合机制                                                                                                                      |
| [w2ner](litner/nn/ner/w2ner.py)                   | [Unified Named Entity Recognition as Word-Word Relation Classification.](https://arxiv.org/pdf/2112.10070.pdf)                                                                | 统一解决嵌套实体、不连续实体的抽取问题                                                                                                                           |
| [cnn](litner/nn/ner/cnn.py)                       | [An Embarrassingly Easy but Strong Baseline for Nested Named Entity Recognition.](https://arxiv.org/abs/2208.04534)                                                           | 改进 `W2NER` 方法，采用卷积网络提取实体内部token之间的关系                                                                                                          |


## 📚 数据

将数据集处理成以下 `json` 格式


```json
{
  "text": "结果上周六他们主场0：3惨败给了中游球队瓦拉多利德，近7个多月以来西甲首次输球。", 
  "entities": [
    {
      "id": 0, 
      "entity": "瓦拉多利德", 
      "start_offset": 20, 
      "end_offset": 25, 
      "label": "organization"
    }, 
    {
      "id": 1, 
      "entity": "西甲", 
      "start_offset": 33, 
      "end_offset": 35, 
      "label": "organization"
    }
  ]
}
```

字段含义：

+ `text`: 文本内容


+ `entities`: 该文本所包含的所有实体

    + `id`: 实体 `id`

    + `entity`: 实体名称
  
    + `start_offset`: 实体开始位置

    + `end_offset`: 实体结束位置的下一位

    + `label`: 实体类型


## 🚀 模型训练

```python
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


parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# 1. create model
model = AutoNerModel.create(model_args=model_args, training_args=training_args)

# 2. finetune model
model.finetune(data_args)
```

训练脚本详见 [scripts](./scripts)


## 📊 模型推理


```python
from litner.pipelines import NerPipeline

task_model = "crf"
model_name_or_path = "path of crf model"
pipeline = NerPipeline(task_model, model_name_or_path=model_name_or_path)

print(pipeline("结果上周六他们主场0：3惨败给了中游球队瓦拉多利德，近7个多月以来西甲首次输球。"))
```
  

## 📜 License

此项目为 `Apache 2.0` 许可证授权，有关详细信息，请参阅 [LICENSE](LICENSE) 文件。
