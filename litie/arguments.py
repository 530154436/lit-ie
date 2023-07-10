from dataclasses import dataclass, field
from typing import Optional, Union

from transformers import SchedulerType


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our gpt2-chinese for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the dataset to use (via the datasets library)."
        }
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        }
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "A csv or a json file containing the training data."
        }
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "A csv or a json file containing the validation data."
        }
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "A csv or a json file containing the test data."
        }
    )
    train_max_length: int = field(
        default=256,
        metadata={
            "help": (
                "The maximum total input sequence length of train dataset after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    validation_max_length: int = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length of valid dataset after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(
        default=False,
        metadata={
            "help": "Enable streaming mode."
        }
    )
    is_chinese: bool = field(
        default=True,
        metadata={
            "help": "Whether the language is Chinese."
        }
    )
    is_sparse: bool = field(
        default=False,
        metadata={
            "help": "Whether to use sparse loss."
        }
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={
            "help": "Overwrite the cached training and evaluation sets"
        }
    )
    validation_split_percentage: Optional[int] = field(
        default=None,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of processes to use for the preprocessing."
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


@dataclass
class ModelArguments:
    task_name: str = field(
        default=None,
        metadata={
            "help": (
                "The task name for fine-tune a specific dataset and task."
            )
        },
    )
    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": (
                "The gpt2-chinese checkpoint for weights initialization.Don't set if you want to train a gpt2-chinese from scratch."
            )
        },
    )
    model_type: str = field(
        default="bert",
        metadata={
            "help": (
                    "The model type of task model, eg bert, ernie et al."
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )


@dataclass
class TrainingArguments:
    learning_rate: float = field(
        default=2e-5,
        metadata={
            "help": (
                "The initial learning rate for fine-tune."
            )
        },
    )
    other_learning_rate: float = field(
        default=0.0,
        metadata={
            "help": (
                "The initial learning rate for other layers."
            )
        },
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": (
                "Number of updates steps to accumulate before performing a backward/update pass."
            )
        },
    )
    weight_decay: float = field(
        default=0.01,
        metadata={
            "help": (
                "Weight decay for fine-tune if we apply some."
            )
        },
    )
    warmup_ratio: float = field(
        default=0.0,
        metadata={
            "help": (
                "Linear warmup over warmup_ratio fraction of total steps."
            )
        },
    )
    warmup_steps: int = field(
        default=0,
        metadata={
            "help": (
                "Linear warmup over warmup_steps."
            )
        },
    )
    num_train_epochs: float = field(
        default=3.0,
        metadata={
            "help": (
                "Total number of training epochs to perform."
            )
        },
    )
    per_device_train_batch_size: int = field(
        default=8,
        metadata={
            "help": (
                "Batch size per GPU/TPU core/CPU for training."
            )
        },
    )
    per_device_eval_batch_size: int = field(
        default=8,
        metadata={
            "help": (
                "Batch size per GPU/TPU core/CPU for evaluation."
            )
        },
    )

    lr_scheduler_type: Union[SchedulerType, str] = field(
        default="linear",
        metadata={
            "help": (
                "The scheduler type to use."
            )
        },
    )
    val_check_interval: float = field(
        default=0.5,
        metadata={
            "help": (
                "Verbose every val_check_interval of epoch."
            )
        },
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={
            "help": (
                "Max gradient norm.")
        },
    )
    adafactor: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to replace AdamW by Adafactor.")
        },
    )
    output_dir: str = field(
        default="outputs",
        metadata={
            "help": (
                "The output directory where the model predictions and checkpoints will be written.")
        },
    )
    do_adv: bool = field(
        default=False,
        metadata={
            "help": "Whether to run adversarial training."
        }
    )
    adv_mode: str = field(
        default="fgm",
        metadata={
            "help": "Adversarial training method."
        }
    )
    adv_embedding_name: str = field(
        default="word_embeddings",
        metadata={
            "help": "Adversarial training embedding name."
        }
    )
    adv_epsilon: float = field(
        default=1.,
        metadata={
            "help": "Adversarial training epsilon."
        }
    )
    adv_alpha: float = field(
        default=0.3,
        metadata={
            "help": "Adversarial training alpha."
        }
    )
    num_adv_attacks: int = field(
        default=3,
        metadata={
            "help": "Adversarial training nums of attacks."
        }
    )
    seed: int = field(
        default=42,
        metadata={
            "help": "Random seed."
        }
    )
    loss_type: str = field(
        default="cross_entropy",
        metadata={
            "help": "Loss type for computing object."
        }
    )
