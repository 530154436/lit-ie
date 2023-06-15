# Lit-IE

<p align="center">
    <a href="https://github.com/xusenlinzy/lit-ner"><img src="https://img.shields.io/github/license/xusenlinzy/lit-ner"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.8+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/pytorch-%3E=1.12-red?logo=pytorch"></a>
    <a href="https://github.com/xusenlinzy/lit-ner"><img src="https://img.shields.io/github/last-commit/xusenlinzy/lit-ner"></a>
    <a href="https://github.com/xusenlinzy/lit-ner"><img src="https://img.shields.io/github/issues/xusenlinzy/lit-ner?color=9cc"></a>
    <a href="https://github.com/xusenlinzy/lit-ner"><img src="https://img.shields.io/github/stars/xusenlinzy/lit-ner?color=ccf"></a>
    <a href="https://github.com/xusenlinzy/lit-ner"><img src="https://img.shields.io/badge/langurage-py-brightgreen?style=flat&color=blue"></a>
</p>

此项目为开源**实体抽取和关系抽取**模型的训练和推理提供统一的框架，具有以下特性


+ ✨ 支持多种开源实体抽取模型


+ 🙌 支持多种开源关系抽取模型


+ 👑 支持百度 [UIE](https://github.com/PaddlePaddle/PaddleNLP) 模型的训练和推理


+ 🚀 统一的训练和推理框架


+ 🎯 集成对抗训练方法，简便易用


## 📢 News 

+ 【2023.6.15】 增加对抗训练功能和示例、增加 `onerel` 关系抽取模型


+ 【2023.6.14】 新增 `UIE` 模型代码示例


+ 【2023.6.13】 增加实体抽取和关系抽取代码示例

---

## 📦 安装

### 环境依赖

+ python >= 3.7

+ torch >= 1.12

### pip 安装

```shell
pip install --upgrade litie
```


## 🐼 模型

### 实体抽取

| 模型                                               | 论文                                                                                                                                                                            | 备注                                                                                                                                            |
|--------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| [softmax](litie/nn/ner/crf.py)                   |                                                                                                                                                                               | 全连接层序列标注并使用 `BIO` 解码                                                                                                                          |
| [crf](litie/nn/ner/crf.py)                       | [Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers) | 全连接层+条件随机场，并使用 `BIO` 解码                                                                                                                       |
| [cascade-crf](litie/nn/ner/crf.py)               |                                                                                                                                                                               | 先预测实体再预测实体类型                                                                                                                                  |
| [span](litie/nn/ner/span.py)                     |                                                                                                                                                                               | 使用两个指针网络预测实体起始位置                                                                                                                              |
| [global-pointer](litie/nn/ner/global_pointer.py) |                                                                                                                                                                               | [GlobalPointer：用统一的方式处理嵌套和非嵌套NER](https://spaces.ac.cn/archives/8373)、[Efficient GlobalPointer：少点参数，多点效果](https://spaces.ac.cn/archives/8877) |
| [mrc](litie/nn/ner/mrc.py)                       | [A Unified MRC Framework for Named Entity Recognition.](https://aclanthology.org/2020.acl-main.519.pdf)                                                                       | 将实体识别任务转换为阅读理解问题，输入为实体类型模板+句子，预测对应实体的起始位置                                                                                                     |
| [tplinker](litie/nn/ner/tplinker.py)             | [TPLinker: Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking.](https://aclanthology.org/2020.coling-main.138.pdf)                            | 将实体识别任务转换为表格填充问题                                                                                                                              |
| [lear](litie/nn/ner/lear.py)                     | [Enhanced Language Representation with Label Knowledge for Span Extraction.](https://aclanthology.org/2021.emnlp-main.379.pdf)                                                | 改进 `MRC` 方法效率问题，采用标签融合机制                                                                                                                      |
| [w2ner](litie/nn/ner/w2ner.py)                   | [Unified Named Entity Recognition as Word-Word Relation Classification.](https://arxiv.org/pdf/2112.10070.pdf)                                                                | 统一解决嵌套实体、不连续实体的抽取问题                                                                                                                           |
| [cnn](litie/nn/ner/cnn.py)                       | [An Embarrassingly Easy but Strong Baseline for Nested Named Entity Recognition.](https://arxiv.org/abs/2208.04534)                                                           | 改进 `W2NER` 方法，采用卷积网络提取实体内部token之间的关系                                                                                                          |


### 关系抽取

| 模型                                  | 论文                                                                                                                                                 | 备注                                                                  |
|-------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------|
| [casrel](litie/nn/re/casrel.py)     | [A Novel Cascade Binary Tagging Framework for Relational Triple Extraction.](https://aclanthology.org/2020.acl-main.136.pdf)                       | 两阶段关系抽取，先抽取出句子中的主语，再通过指针网络抽取出主语对应的关系和宾语                             |
| [tplinker](litie/nn/re/tplinker.py) | [TPLinker: Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking.](https://aclanthology.org/2020.coling-main.138.pdf) | 将关系抽取问题转换为主语-宾语的首尾连接问题                                              |
| [spn](litie/nn/re/spn.py)           | [Joint Entity and Relation Extraction with Set Prediction Networks.](http://xxx.itp.ac.cn/pdf/2011.01675v2)                                        | 将关系抽取问题转为为三元组的集合预测问题，采用 `Encoder-Decoder` 框架                        |
| [prgc](litie/nn/re/prgc.py)         | [PRGC: Potential Relation and Global Correspondence Based Joint Relational Triple Extraction.](https://aclanthology.org/2021.acl-long.486.pdf)     | 先抽取句子的潜在关系类型，然后对于给定关系抽取出对应的主语-宾语对，最后通过全局对齐模块过滤                      |
| [pfn](litie/nn/re/pfn.py)           | [A Partition Filter Network for Joint Entity and Relation Extraction.](https://aclanthology.org/2021.emnlp-main.17.pdf)                            | 采用类似  `LSTM`  的分区过滤机制，将隐藏层信息分成实体识别、关系识别和共享三部分，对与不同的任务利用不同的信息        |
| [grte](litie/nn/re/grte.py)         | [A Novel Global Feature-Oriented Relational Triple Extraction Model based on Table Filling.](https://aclanthology.org/2021.emnlp-main.208.pdf)     | 将关系抽取问题转换为单词对的分类问题，基于全局特征抽取模块循环优化单词对的向量表示                           |
| [gplinker](litie/nn/re/gplinker.py) |                                                                                                                                                    | [GPLinker：基于GlobalPointer的实体关系联合抽取](https://kexue.fm/archives/8888) |


## 📚 数据

### 实体抽取

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


### 关系抽取

将数据集处理成以下 `json` 格式

```json
{
  "text": "查尔斯·阿兰基斯（Charles Aránguiz），1989年4月17日出生于智利圣地亚哥，智利职业足球运动员，司职中场，效力于德国足球甲级联赛勒沃库森足球俱乐部", 
  "spo_list": [
    {
      "predicate": "出生地",
      "object": "圣地亚哥", 
      "subject": "查尔斯·阿兰基斯"
    }, 
    {
      "predicate": "出生日期",
      "object": "1989年4月17日",
      "subject": "查尔斯·阿兰基斯"
    }
  ]
}
```

字段含义：

+ `text`: 文本内容

+ `spo_list`: 该文本所包含的所有关系三元组

    + `subject`: 主体名称

    + `object`: 客体名称
  
    + `predicate`: 主体和客体之间的关系

  
## 🚀 模型训练

### 实体抽取

```python
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


parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# 1. create model
model = AutoNerModel(model_args=model_args, training_args=training_args)

# 2. finetune model
model.finetune(data_args)
```

训练脚本详见 [named_entity_recognition](./examples/named_entity_recognition)

### 关系抽取

```python
import os
import sys

from transformers import HfArgumentParser

from litie.arguments import (
    DataTrainingArguments,
    ModelArguments,
    TrainingArguments,
)
from litie.models import AutoRelationExtractionModel

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# 1. create model
model = AutoRelationExtractionModel(model_args=model_args, training_args=training_args)

# 2. finetune model
model.finetune(data_args, num_sanity_val_steps=0)

os.remove(os.path.join(training_args.output_dir, "best_model.ckpt"))
```

训练脚本详见 [relation_extraction](./examples/relation_extraction)


## 📊 模型推理

### 实体抽取

```python
from litie.pipelines import NerPipeline

task_model = "crf"
model_name_or_path = "path of crf model"
pipeline = NerPipeline(task_model, model_name_or_path=model_name_or_path)

print(pipeline("结果上周六他们主场0：3惨败给了中游球队瓦拉多利德，近7个多月以来西甲首次输球。"))
```

web demo

```python
from litie.ui import NerPlayground

NerPlayground().launch()
```


### 关系抽取

```python
from litie.pipelines import RelationExtractionPipeline

task_model = "gplinker"
model_name_or_path = "path of gplinker model"
pipeline = RelationExtractionPipeline(task_model, model_name_or_path=model_name_or_path)

print(pipeline("查尔斯·阿兰基斯（Charles Aránguiz），1989年4月17日出生于智利圣地亚哥，智利职业足球运动员，司职中场，效力于德国足球甲级联赛勒沃库森足球俱乐部"))
```

web demo

```python
from litie.ui import RelationExtractionPlayground

RelationExtractionPlayground().launch()
```

## 🍭 通用信息抽取

+ [UIE(Universal Information Extraction)](https://arxiv.org/pdf/2203.12277.pdf)：Yaojie Lu等人在ACL-2022中提出了通用信息抽取统一框架 `UIE`。

+ 该框架实现了实体抽取、关系抽取、事件抽取、情感分析等任务的统一建模，并使得不同任务间具备良好的迁移和泛化能力。

+ [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)借鉴该论文的方法，基于 `ERNIE 3.0` 知识增强预训练模型，训练并开源了首个中文通用信息抽取模型 `UIE`。

+ 该模型可以支持不限定行业领域和抽取目标的关键信息抽取，实现零样本快速冷启动，并具备优秀的小样本微调能力，快速适配特定的抽取目标。

### 模型训练

模型训练脚本详见 [uie](./examples/uie)

### 模型推理

<details>
<summary>👉 命名实体识别</summary>

```python
from pprint import pprint
from litie.pipelines import UIEPipeline

# 实体识别
schema = ['时间', '选手', '赛事名称'] 
# uie-base模型已上传至huggingface，可自动下载，其他模型只需提供模型名称将自动进行转换
uie = UIEPipeline("xusenlin/uie-base", schema=schema)
pprint(uie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！")) # Better print results using pprint
```

output: 

```json
[
  {
    "时间": [
      {
        "end": 6,
        "probability": 0.98573786,
        "start": 0,
        "text": "2月8日上午"
      }
    ],
    "赛事名称": [
      {
        "end": 23,
        "probability": 0.8503085,
        "start": 6,
        "text": "北京冬奥会自由式滑雪女子大跳台决赛"
      }
    ],
    "选手": [
      {
        "end": 31,
        "probability": 0.8981544,
        "start": 28,
        "text": "谷爱凌"
      }
    ]
  }
]
```
</details>

<details>
<summary>👉 实体关系抽取</summary>

```python
from pprint import pprint
from litie.pipelines import UIEPipeline

# 关系抽取
schema = {'竞赛名称': ['主办方', '承办方', '已举办次数']}
# uie-base模型已上传至huggingface，可自动下载，其他模型只需提供模型名称将自动进行转换
uie = UIEPipeline("xusenlin/uie-base", schema=schema)
pprint(uie("2022语言与智能技术竞赛由中国中文信息学会和中国计算机学会联合主办，百度公司、中国中文信息学会评测工作委员会和中国计算机学会自然语言处理专委会承办，已连续举办4届，成为全球最热门的中文NLP赛事之一。")) # Better print results using pprint
```

output:

```json
[
  {
    "竞赛名称": [
      {
        "end": 13,
        "probability": 0.78253937,
        "relations": {
          "主办方": [
            {
              "end": 22,
              "probability": 0.8421704,
              "start": 14,
              "text": "中国中文信息学会"
            },
            {
              "end": 30,
              "probability": 0.75807965,
              "start": 23,
              "text": "中国计算机学会"
            }
          ],
          "已举办次数": [
            {
              "end": 82,
              "probability": 0.4671307,
              "start": 80,
              "text": "4届"
            }
          ],
          "承办方": [
            {
              "end": 55,
              "probability": 0.700049,
              "start": 40,
              "text": "中国中文信息学会评测工作委员会"
            },
            {
              "end": 72,
              "probability": 0.61934763,
              "start": 56,
              "text": "中国计算机学会自然语言处理专委会"
            },
            {
              "end": 39,
              "probability": 0.8292698,
              "start": 35,
              "text": "百度公司"
            }
          ]
        },
        "start": 0,
        "text": "2022语言与智能技术竞赛"
      }
    ]
  }
]
```
</details>


<details>
<summary>👉  事件抽取</summary>

```python
from pprint import pprint
from litie.pipelines import UIEPipeline

# 事件抽取
schema = {"地震触发词": ["地震强度", "时间", "震中位置", "震源深度"]}
# uie-base模型已上传至huggingface，可自动下载，其他模型只需提供模型名称将自动进行转换
uie = UIEPipeline("xusenlin/uie-base", schema=schema)
pprint(uie("中国地震台网正式测定：5月16日06时08分在云南临沧市凤庆县(北纬24.34度，东经99.98度)发生3.5级地震，震源深度10千米。")) # Better print results using pprint
```

output:

```json
[
  {
    "地震触发词": [
      {
        "end": 58,
        "probability": 0.99774253,
        "relations": {
          "地震强度": [
            {
              "end": 56,
              "probability": 0.9980802,
              "start": 52,
              "text": "3.5级"
            }
          ],
          "时间": [
            {
              "end": 22,
              "probability": 0.98533,
              "start": 11,
              "text": "5月16日06时08分"
            }
          ],
          "震中位置": [
            {
              "end": 50,
              "probability": 0.7874015,
              "start": 23,
              "text": "云南临沧市凤庆县(北纬24.34度，东经99.98度)"
            }
          ],
          "震源深度": [
            {
              "end": 67,
              "probability": 0.9937973,
              "start": 63,
              "text": "10千米"
            }
          ]
        },
        "start": 56,
        "text": "地震"
      }
    ]
  }
]
```
</details>

<details>
<summary>👉 评论观点抽取</summary>

```python
from pprint import pprint
from litie.pipelines import UIEPipeline

# 评论观点抽取
schema = {'评价维度': ['观点词', '情感倾向[正向，负向]']}
# uie-base模型已上传至huggingface，可自动下载，其他模型只需提供模型名称将自动进行转换
uie = UIEPipeline("xusenlin/uie-base", schema=schema)
pprint(uie("店面干净，很清静，服务员服务热情，性价比很高，发现收银台有排队")) # Better print results using pprint
```

output:

```json
[
  {
    "评价维度": [
      {
        "end": 20,
        "probability": 0.98170394,
        "relations": {
          "情感倾向[正向，负向]": [
            {
              "probability": 0.9966142773628235,
              "text": "正向"
            }
          ],
          "观点词": [
            {
              "end": 22,
              "probability": 0.95739645,
              "start": 21,
              "text": "高"
            }
          ]
        },
        "start": 17,
        "text": "性价比"
      },
      {
        "end": 2,
        "probability": 0.9696847,
        "relations": {
          "情感倾向[正向，负向]": [
            {
              "probability": 0.9982153177261353,
              "text": "正向"
            }
          ],
          "观点词": [
            {
              "end": 4,
              "probability": 0.9945317,
              "start": 2,
              "text": "干净"
            }
          ]
        },
        "start": 0,
        "text": "店面"
      }
    ]
  }
]
```
</details>


<details>
<summary>👉 情感分类</summary>


```python
from pprint import pprint
from litie.pipelines import UIEPipeline

# 事件抽取
schema = '情感倾向[正向，负向]'
# uie-base模型已上传至huggingface，可自动下载，其他模型只需提供模型名称将自动进行转换
uie = UIEPipeline("xusenlin/uie-base", schema=schema)
pprint(uie("这个产品用起来真的很流畅，我非常喜欢")) # Better print results using pprint
```

output:

```json
[
  {
    "情感倾向[正向，负向]": [
      {
        "probability": 0.9990023970603943,
        "text": "正向"
      }
    ]
  }
]
```
</details>


## 📜 License

此项目为 `Apache 2.0` 许可证授权，有关详细信息，请参阅 [LICENSE](LICENSE) 文件。
