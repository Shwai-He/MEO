# Merging Experts into One: Improving Computational Efficiency of Mixture of Experts
The source code of "Merging Experts into One: Improving Computational Efficiency of Mixture of Experts
 (EMNLP 2023)":

 ```
Merging Experts into One: Improving Computational Efficiency of Mixture of Experts
Shwai He, Run-Ze Fan, Liang Ding, Li Shen, Tianyi Zhou, Dacheng Tao
EMNLP 2023 Main Conference. 
```

## Requirements
- torch==1.13.1
- transformers==4.17.0
- tokenizers==0.10.1
- nltk==3.5

## Usage

You should add your local path of transformers in run_glue.py. \
To train MoE on GLUE, run: 

`./dyconv/main.py` 

To train DY-Conv on ImageNet, run: 

`./moe/tasks/text-classification/run_glue_pad.py` 

## Citation

```
@misc{he2023merging,
      title={Merging Experts into One: Improving Computational Efficiency of Mixture of Experts}, 
      author={Shwai He and Run-Ze Fan and Liang Ding and Li Shen and Tianyi Zhou and Dacheng Tao},
      year={2023},
      eprint={2310.09832},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```