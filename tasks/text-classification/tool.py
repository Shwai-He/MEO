import os
import sys

sys.path = ['/mnt/petrelfs/dongdaize.d/workspace/sh/MEO', 
            os.path.abspath(os.path.join(os.getcwd(), "../.."))] + sys.path

import transformers
print(transformers)
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

config = AutoConfig.from_pretrained(
    "bert-base-uncased", 
    num_labels=2,
    finetuning_task="sst2",
    cache_dir='/mnt/petrelfs/dongdaize.d/workspace/sh/MEO/cache_dir', 
    revision='main',
    use_auth_token=None
    )
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased", 
    cache_dir='/mnt/petrelfs/dongdaize.d/workspace/sh/MEO/cache_dir', 
    use_fast=True,
    revision='main',
    use_auth_token=None
    )
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    cache_dir='/mnt/petrelfs/dongdaize.d/workspace/sh/MEO/cache_dir', 
    revision='main',
    use_auth_token=None
    )