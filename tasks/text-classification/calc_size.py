import sys
import os

sys.path = [os.path.abspath(os.path.join(os.getcwd(), "../.."))] + sys.path

import torch
from typing import Optional
from dataclasses import dataclass, field
from transformers import (
    AutoModelForSequenceClassification,
    HfArgumentParser,
    AutoConfig,
)
import numpy as np
import math



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    k: int = field(
        default=8,
        metadata={
            "help": "the top-k of experts in moe"
        },
    )
    n_experts: int = field(
        default=32,
        metadata={
            "help": "the number of experts in moe"
        },
    )
    use_moe: str = field(
        default='MEO',
    )
    moe_level: str = field(
        default='token',
    )
    description_size: int = field(
        default=128,
    )

def main():

    # ost = sys.stdout

    num_labels = 3
    task_name = 'cola'
    parser = HfArgumentParser((ModelArguments))
    model_args = parser.parse_args_into_dataclasses()[0]
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    setattr(config, 'k', model_args.k)
    setattr(config, 'use_moe', model_args.use_moe)
    setattr(config, 'n_experts', model_args.n_experts)
    setattr(config, 'moe_level', model_args.moe_level)
    setattr(config, 'description_size', model_args.description_size)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # sparsity = 0.3
    params_all = 0


    for name, params in model.named_parameters():
        # if len(params.size()) == 5:
        #     scale = (params.size()[0] * sparsity + 1 - sparsity)
        #     temp = np.prod(params.size()) / params.size()[0] * scale
        #     # temp = np.prod(params.size())
        # elif 'attention' in name:
        #     # continue
        #     temp = np.prod(params.size()) * sparsity
        #     # temp = np.prod(params.size())
        # else:
        #     # print(name, params.size())
        #     temp = np.prod(params.size())
        # params_all += temp
        params_all += np.prod(params.size())
    n = 1000
    print(params_all / math.pow(n, 2))

    # flops, params = get_model_complexity_info(model, (128, 128),
    #                         as_strings=True,
    #                         print_per_layer_stat=True,
    #                         ost=ost,
    #                         output_precision=3)

    # print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

if __name__ == "__main__":
    main()

