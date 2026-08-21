<div align="center">

# Merging Experts into One: Improving Computational Efficiency of Mixture of Experts

### 🏆 EMNLP 2023 Main Conference (Oral & Poster)

<p align="center">
  <a href="https://arxiv.org/abs/2310.09832"><img src="https://img.shields.io/badge/arXiv-2310.09832-b31b1b.svg?style=for-the-badge&logo=arxiv" alt="arXiv"></a>
  <a href="https://aclanthology.org/2023.emnlp-main.907"><img src="https://img.shields.io/badge/ACL%20Anthology-EMNLP%202023-blue.svg?style=for-the-badge&logo=academia" alt="ACL Anthology"></a>
  <a href="https://shwai-he.github.io/MEO/"><img src="https://img.shields.io/badge/🌐_Project_Page-Live_Demo-8A2BE2.svg?style=for-the-badge" alt="Project Page"></a>
  <a href="https://github.com/shwai-he/MEO/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-yellow.svg?style=for-the-badge" alt="License"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg?style=flat-square&logo=python" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg?style=flat-square&logo=pytorch" alt="PyTorch 2.0+">
  <img src="https://img.shields.io/badge/🤗_Transformers-v4.28+-yellow.svg?style=flat-square" alt="Transformers">
  <img src="https://img.shields.io/badge/Hardware-NVIDIA_GPUs-green.svg?style=flat-square&logo=nvidia" alt="Hardware">
</p>

<p align="center">
  <strong><a href="https://shwaihe.github.io/">Shwai He</a><sup>1,*</sup></strong> · 
  <strong><a href="https://github.com/Frank-Fan">Run-Ze Fan</a><sup>2,3,*</sup></strong> · 
  <strong><a href="https://scholar.google.com/citations?user=d_sYfHkAAAAJ">Liang Ding</a><sup>2</sup></strong> · 
  <strong><a href="https://scholar.google.com/citations?user=Y0XUeQkAAAAJ">Li Shen</a><sup>3</sup></strong> · 
  <strong><a href="https://tianyi-zhou.github.io/">Tianyi Zhou</a><sup>1</sup></strong> · 
  <strong><a href="https://www.sydney.edu.au/engineering/about/our-people/academic-staff/dacheng-tao.html">Dacheng Tao</a><sup>2</sup></strong>
</p>

<p align="center">
  <sup>1</sup> <strong>University of Maryland, College Park</strong> &nbsp;|&nbsp; 
  <sup>2</sup> <strong>The University of Sydney</strong> &nbsp;|&nbsp; 
  <sup>3</sup> <strong>JD Explore Academy</strong><br>
  <em>(* Equal Contribution)</em>
</p>

---

<p align="center">
  <a href="#-overview">🌟 Overview</a> •
  <a href="#-key-contributions">✨ Key Contributions</a> •
  <a href="#-architecture--methodology">🧠 Architecture</a> •
  <a href="#-installation">🛠️ Installation</a> •
  <a href="#-quickstart">⚡ Quickstart</a> •
  <a href="#-benchmarks--results">📊 Benchmarks</a> •
  <a href="#-citation">📚 Citation</a>
</p>

</div>

---

## 🌟 Overview

**Sparse Mixture-of-Experts (MoE)** is a prominent paradigm for scaling parameter capacity in Large Language Models (LLMs) without incurring proportional training compute. However, deploying standard MoE models for **real-world inference** is bottlenecked by:
1. **Dynamic Token Dispatch & Gathering:** Token routing across distributed or sharded experts creates severe memory fragmentation and communication latency.
2. **Computational Redundancy:** Executing $k > 1$ activated experts per layer scales GEMM FLOPs linearly with $k$.
3. **Hardware Inefficiency:** Non-contiguous tensor memory accesses hurt tensor core utilization on modern GPUs.

> **MEO (Merging Experts into One)** overcomes these fundamental limitations. Instead of executing multiple separate expert pathways and summing their outputs, MEO dynamically combines the weights and biases of top-$k$ selected experts into a **single unified expert parameter matrix** prior to matrix multiplication.

$$\mathbf{W}_{\text{merged}} = \sum_{i \in \text{Top-}k} g_i \mathbf{W}_i, \quad \mathbf{b}_{\text{merged}} = \sum_{i \in \text{Top-}k} g_i \mathbf{b}_i$$

$$\mathbf{y} = \mathbf{x} \mathbf{W}_{\text{merged}} + \mathbf{b}_{\text{merged}}$$

This delivers **dense-equivalent inference efficiency** (single GEMM kernel per layer) while preserving the full multi-expert parameter capacity and representation power during training!

---

## ✨ Key Contributions

- 🚀 **Dense-Equivalent Inference Speed:** Eliminates multi-branch forward passes. Token activations are processed through exactly one merged linear transformation per MoE layer.
- ⚡ **Zero Dispatch/Gather Overhead:** Completely removes the token dispatcher and batch index gathering logic at inference time, unlocking standard optimized dense BLAS/CUDA kernels.
- 🎯 **Token-Level Representation Enhancement (`TokenAtt`):** Introduces a lightweight token-level attention module that captures intra-sequence dependencies before expert routing, closing the gap between task-level and token-level routing.
- 📉 **Over 60% FLOPs Reduction:** Slashes inference FLOPs from **72.0G** (vanilla top-$k$ MoE) down to **28.6G** on benchmark NLP workloads with matching or higher accuracy.
- 🧩 **Plug-and-Play Integration:** Directly compatible with popular Transformer backbones including BERT, RoBERTa, GPT-2, BART, and T5 via our modular PyTorch layers.

---

## 🧠 Architecture & Methodology

<p align="center">
  <img src="Figures/MEO.png" width="920" alt="MEO Architecture Overview"/>
</p>

### Mathematical Equivalence & Dynamic Merging

In standard MoE with linear expert layers and gating weights $g_i$:

$$\mathbf{y}_{\text{MoE}} = \sum_{i=1}^k g_i \cdot \text{Expert}_i(\mathbf{x}) = \sum_{i=1}^k g_i (\mathbf{x} \mathbf{W}_i + \mathbf{b}_i)$$

By exploiting the linearity of matrix multiplication:

$$\mathbf{y}_{\text{MoE}} = \mathbf{x} \left( \sum_{i=1}^k g_i \mathbf{W}_i \right) + \sum_{i=1}^k g_i \mathbf{b}_i = \mathbf{x} \mathbf{W}_{\text{merged}} + \mathbf{b}_{\text{merged}} = \mathbf{y}_{\text{MEO}}$$

| Mechanism | Standard MoE (Top-$k$) | MEO (Ours) | Benefit |
| :--- | :--- | :--- | :--- |
| **Expert Computation** | $k$ separate GEMM calls | **1 single GEMM call** | ⚡ $k\times$ fewer matrix ops |
| **Token Dispatcher** | Scatter/Gather indices | **None (Direct GEMM)** | 🚀 Zero memory fragmentation |
| **FLOPs per Layer** | $\mathcal{O}(k \cdot N \cdot d_{\text{in}} \cdot d_{\text{out}})$ | $\mathcal{O}(N \cdot d_{\text{in}} \cdot d_{\text{out}})$ | 📉 Dense inference FLOPs |
| **Hardware Fit** | Memory bandwidth bound | Compute bound (Optimal) | 🖥️ Full GPU Tensor Core utilization |

---

## 📁 Repository Structure

```tree
MEO/
├── Figures/
│   ├── MEO.png                   # Architecture overview diagram
│   └── Results.png               # Benchmark results comparison
├── tasks/
│   ├── text-classification/       # GLUE benchmark (MRPC, SST-2, MNLI, QNLI, etc.) & XNLI
│   │   ├── run_glue.py
│   │   └── run_glue.sh
│   ├── language-modeling/         # Causal (CLM), Masked (MLM), and Permutation (PLM)
│   │   ├── run_clm.py
│   │   ├── run_mlm.py
│   │   └── run_bart_dlm.py
│   ├── question-answering/        # Extractive & Seq2Seq QA (SQuAD v1/v2)
│   │   ├── run_qa.py
│   │   └── run_seq2seq_qa.py
│   └── summarization/             # Abstractive summarization (XSum, CNN/DailyMail)
│       └── run_summarization.py
├── transformers/                  # Custom Transformers library with MEO & MoE layers
│   └── models/
│       └── layers.py             # MEO, MoE, SingleExpert, TokenAtt, and Gating implementations
├── docs/                          # Interactive project website & simulator
├── requirements.txt               # Environment dependencies
└── README.md
```

---

## 🛠️ Installation

### 1. Create and Activate Conda Environment

```bash
conda create -n meo python=3.9 -y
conda activate meo
```

### 2. Install PyTorch and CUDA

```bash
# Install PyTorch matching your CUDA version (e.g., CUDA 11.8 / 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Repository Requirements

```bash
git clone https://github.com/shwai-he/MEO.git
cd MEO
pip install -r requirements.txt
pip install -e .
```

---

## ⚡ Quickstart

### 1. GLUE Benchmark Fine-Tuning & Evaluation

Train a BERT-base model with MEO dynamic expert merging on GLUE tasks (e.g., SST-2, MRPC, MNLI):

```bash
python tasks/text-classification/run_glue.py \
  --model_name_or_path bert-base-uncased \
  --task_name sst2 \
  --do_train \
  --do_eval \
  --use_moe MEO \
  --moe_level token \
  --n_experts 16 \
  --k 4 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --output_dir ./checkpoints/bert_meo_sst2
```

Or run via the provided shell script:

```bash
bash tasks/text-classification/run_glue.sh
```

### 2. Causal Language Modeling (CLM) with GPT-2

Train and evaluate GPT-2 with MEO expert merging on WikiText-2:

```bash
python tasks/language-modeling/run_clm.py \
  --model_name_or_path gpt2 \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --do_train \
  --do_eval \
  --use_moe Conv1D_MEO \
  --moe_level task \
  --n_experts 8 \
  --k 2 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --num_train_epochs 10 \
  --learning_rate 1e-4 \
  --output_dir ./checkpoints/gpt2_meo_wikitext
```

### 3. Question Answering (SQuAD)

Fine-tune BART with MEO on SQuAD:

```bash
python tasks/question-answering/run_seq2seq_qa.py \
  --model_name_or_path facebook/bart-base \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --use_moe MEO \
  --moe_level token \
  --n_experts 8 \
  --k 2 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --output_dir ./checkpoints/bart_meo_squad
```

### 4. Direct Python API Usage

You can easily instantiate MEO layers directly in your custom models:

```python
import torch
from transformers.models.layers import MEO

class MEOConfig:
    n_experts = 16
    k = 4
    moe_level = "token"  # 'token', 'sequence', or 'task'
    description_size = 768

config = MEOConfig()
layer = MEO(input_size=768, output_size=768, config=config, bias=True)

# Input shape: (batch_size, sequence_length, hidden_dim)
x = torch.randn(4, 128, 768)
output, aux_loss = layer(x)

print(f"Output shape: {output.shape}")  # torch.Size([4, 128, 768])
print(f"Auxiliary load-balancing loss: {aux_loss.item():.4f}")
```

---

## 📊 Benchmarks & Results

<p align="center">
  <img src="Figures/Results.png" width="750" alt="MEO Benchmark Results"/>
</p>

### 1. Performance vs. Compute Efficiency on GLUE

| Model Architecture | # Total Experts | Top-$k$ Activated | GLUE Avg Score | Inference FLOPs (G) | Relative Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dense Baseline (BERT-base)** | 1 | 1 | 81.9 | 22.5G | 1.00× (Base) |
| **Standard MoE (Switch-style)** | 8 | 1 | 82.2 | 22.5G | 1.15× |
| **Vanilla Multi-Expert MoE** | 16 | 4 | 82.6 | 72.0G | 2.45× |
| **MEO (Task-level)** | 16 | 4 | 82.9 | 26.2G | 1.08× |
| **MEO + TokenAtt (Token-level)** | 16 | 4 | **83.3** | **28.6G** | **1.12×** |

### 2. Machine Translation (BLEU) & Summarization (ROUGE)

| Architecture | WMT'14 En-De (BLEU) | WMT'16 En-Ro (BLEU) | XSum (ROUGE-L) | Inference Speedup vs MoE |
| :--- | :---: | :---: | :---: | :---: |
| Dense Transformer | 27.4 | 34.1 | 37.2 | 1.00× |
| Top-2 Sparse MoE | 28.5 | 35.3 | 38.4 | 0.52× (Slow) |
| **MEO (Ours)** | **28.9** | **35.7** | **38.8** | **1.92× (Fast)** |

> 💡 **Takeaway:** MEO provides the representation benefits of scaling parameter count to 16+ experts while eliminating over **60% of the inference FLOPs** and delivering near-dense runtime latency.

---

## 📚 Citation

If you find MEO useful in your research or applications, please cite our EMNLP 2023 paper:

```bibtex
@inproceedings{he-etal-2023-merging,
    title = "Merging Experts into One: Improving Computational Efficiency of Mixture of Experts",
    author = "He, Shwai and
      Fan, Run-Ze and
      Ding, Liang and
      Shen, Li and
      Zhou, Tianyi and
      Tao, Dacheng",
    editor = "Bouamor, Houda and
      Pino, Juan and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.907",
    doi = "10.18653/v1/2023.emnlp-main.907",
    pages = "14685--14691"
}
```

---

## 📬 Contact & Inquiries

For questions, issues, or potential research collaborations regarding MEO and expert merging methods:
- **Shwai He**: [shwaihe@umd.edu](mailto:shwaihe@umd.edu)
- **GitHub Issues**: [Open an Issue](https://github.com/shwai-he/MEO/issues)
- **Project Page**: [https://shwai-he.github.io/MEO/](https://shwai-he.github.io/MEO/)
