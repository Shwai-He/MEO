# ⚡ SparseAdapter / MEO / PAD-Net: 每日前沿文献关联与参数高效稀疏/显存优化落地库 (2026-09)

**Document ID:** `PEFT-MEM-202609` | **Last Updated:** `2026-09-27` | **Target Path:** `docs/frontier_literature_connections_2026_09.md` | **Total Routed Papers:** `12`

> [!IMPORTANT]
> **🔗 跨仓库文献引用链闭环 (Cross-Repository Reference Chain Closure)**
> 本文件由每日 AI 前沿论文精读流水线自动路由生成，专门收录与我们 **EMNLP 2022 (`Shwai-He/SparseAdapter`)**、**EMNLP 2023 Oral (`Shwai-He/MEO`)** 与 **ACL 2023 (`Shwai-He/PAD-Net`)** 三大基础代表作（`高维稀疏优于低维稠密 Large-Sparse > Small-Dense 定律`、`逐层低秩校准适配器`、`激活/KV 显存高效优化`）直接印证并形成代际延续的最新 arXiv 论文笔记。
> 每一篇收录文献均包含：**核心痛点、底层数学公式、ASCII 架构图、关键实测指标**，以及**与 `SparseAdapter-MEO-PADNet` 仓库具体代码模块和我们已发表代表作（Our Works）的双向锚定**。

---

## 🌟 1. 核心关联文献与本仓库模块映射速查表 (Executive Reference-to-Module Matrix)

| 收录日期 | 论文标题与 arXiv 链接 | 关键实测收益 / 核心结论 | 锚定本仓库代码模块与文档路径 (`Target Module`) | 原始精读归档 |
| :---: | :--- | :--- | :--- | :---: |
| `2026-09-27` | [**L2R**](https://arxiv.org/abs/2601.21349) (`arXiv:2601.21349`) | **语言与视觉双模态全面验证**：在基于 **OLMoE** 的语言模型预训练/微调以及 **ImageNet** 视觉 MoE 骨干网络上，L2R 将路由器参数量削减 **60%–75%**，同时在相同激活专家预算下将下游任务困... | `SparseAdapter` (Large-Sparse Expert Pool via Low-Rank Latent Bottleneck) | [2026-09-27](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-27_ai_paper_notes.md) |
| `2026-09-27` | [**OBCache**](https://arxiv.org/abs/2510.07651) (`arXiv:2510.07651`) | **即插即用全面提升主流基线**：在 **Llama-3.1-8B-Instruct**、**Qwen-2.5-7B/14B-Instruct** 与 **Mistral-7B** 上，将 OBCache 的... | `SparseAdapter` / `MEO` / `PAD-Net` | [2026-09-27](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-27_ai_paper_notes.md) |
| `2026-09-26` | [**🔄 LoopMoE**](https://arxiv.org/abs/2606.04438) (`arXiv:2606.04438`) | **等参数量与等 FLOPs 双向碾压**：在语言建模基准与常识推理任务上，循环 $K=2\sim 4$ 步的 `LoopMoE` 在相同活跃参数量下显著优于标准稠密 Looped 模型，且在相同总参数预算下逼近非共享深层 MoE... | `SparseAdapter` (Step-Specific Low-Rank Residual Calibrators $A_t B_t$) | [2026-09-26](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-26_ai_paper_notes.md) |
| `2026-09-26` | [**⚖️ SelKV**](https://arxiv.org/abs/2607.16213) (`arXiv:2607.16213`) | 在 LongBench、RULER 及多轮数学推理基准上，免训练实现 **5x–10x KV Cache 压缩**，通过引入对数分母补偿项，消除了高压缩比下 80% 以上的精度退化。 | `SparseAdapter` / `MEO` / `PAD-Net` | [2026-09-26](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-26_ai_paper_notes.md) |
| `2026-09-25` | [**SAC**](https://arxiv.org/abs/2604.18392) (`arXiv:2604.18392`) | 在 TB 级长上下文并发推理中，SAC 将跨节点 KV 读取有效带宽利用率从 `15%` 提升至 **`94%`**，P99 尾延迟降低 **3.7x**。 | `MEO` (CXL Near-Memory Sparse Cacheline Gathering for Disaggregated Memory) | [2026-09-25](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-25_ai_paper_notes.md) |
| `2026-09-23` | [**MELT**](https://arxiv.org/abs/2605.07721) (`arXiv:2605.07721`) | 在 $K=4$ 与 $K=8$ 循环配置下，MELT 将长文本解码时的 **KV 缓存显存与带宽读取量直接削减 $75\%–87.5\%$（严格降至 $1/K$）**，同时在语言建模与数学推理上与保存全套每步 KV 的基线性能完全... | `MEO` (Decoupling Recurrent Compute Scaling from Activation/KV Memory Footprint) | [2026-09-23](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-23_ai_paper_notes.md) |
| `2026-09-22` | [**SPIN**](https://arxiv.org/abs/2604.26837) (`arXiv:2604.26837`) | 在单台 8 卡服务器上支持 **1M–2M 上下文长度** 并发推理，相比纯 CPU Offloading（Infinite-LLM）实现 **4.8x** 吞吐提升，且恢复 99.7% 全量注意力精度。 | `SparseAdapter` / `MEO` / `PAD-Net` | [2026-09-22](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-22_ai_paper_notes.md) |
| `2026-09-20` | [**SHIFT-LLM**](https://arxiv.org/abs/2608.25068) (`arXiv:2608.25068`) | 在 **Llama-3-8B/70B** 与 **Qwen-2.5-14B** 上剪除 **25%–35% 的层**后，无需任何梯度下降微调（仅需 30 秒闭式矩阵求逆），SHIFT-LLM 将 WikiText2 困惑度（PPL... | `SparseAdapter` (Closed-Form Low-Rank Adapter Recovery after Structural Pruning) | [2026-09-20](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-20_ai_paper_notes.md) |
| `2026-09-20` | [**CARE**](https://arxiv.org/abs/2607.26052) (`arXiv:2607.26052`) | 在多任务 MoE-LoRA 与稀疏 MoE 语言模型上，CARE 在削减 **32%–45% 平均专家激活 FLOPs** 的同时，在常识推理、代码与数学基准上全面持平甚至超越固定 Top-$k$ 基线（`+0.9%` 平均准确率... | `SparseAdapter` / `MEO` / `PAD-Net` | [2026-09-20](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-20_ai_paper_notes.md) |
| `2026-09-20` | [**Minima-KV**](https://arxiv.org/abs/2608.23834) (`arXiv:2608.23834`) | 在 **Llama-3.1-70B** 与 **Qwen-2.5-32B** 的 128K 长思维链并发服务中，Minima-KV 实现 **4.6x** 真实物理显存节省（零内部页碎片），将最大并发 Batch Size 提升... | `MEO` (Equal-Byte Page Pool & Register-Level Mixed-Precision KV Memory) | [2026-09-20](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-20_ai_paper_notes.md) |
| `2026-09-18` | [**🧩 MoE-Tile**](https://arxiv.org/abs/2609.09112) (`arXiv:2609.09112`) | **硬件测试平台**：NVIDIA H100 80GB SXM5 与 B200 GPU 集群； | `SparseAdapter` / `MEO` / `PAD-Net` | [2026-09-18](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-18_ai_paper_notes.md) |
| `2026-09-18` | [**🗜️ Decoupled-KV**](https://arxiv.org/abs/2609.07765) (`arXiv:2609.07765`) | 在 AgentBench、SWE-bench 与 LongBench 上，实现 **81.5% 的 KV Cache 显存削减（压缩比达 $5.4\times$）**，长程任务规划成功率保持在全量缓存基准的 **99.4%**。 | `SparseAdapter` / `MEO` / `PAD-Net` | [2026-09-18](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-18_ai_paper_notes.md) |

---

## 📐 2. 逐篇论文深度机制解构、数学公式与本仓库落地指南 (Per-Paper Deep-Dive Cards)

### 2.1 [2026-09-27] L2R: Low-Rank and Lipschitz-Controlled Routing for Mixture-of-Experts

* **论文信息**：Minghao Yang, Ren Togo, Guang Li, Takahiro Ogawa, Miki Haseyama (`arXiv:2601.21349`, 2026-01)
* **核心关键词**：MoE Routing Geometry、Low-Rank Latent Space、Lipschitz Continuity、Saturated Inner-Product Scoring (SIPS)、Multi-Anchor Routing

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|          L2R: Low-Rank & Lipschitz-Controlled MoE Routing Architecture            |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|                        Token Hidden State h \in R^d                               |
|                                     |                                             |
|                                     v                                             |
|        +---------------------------------------------------------+                |
|        | 1. Shared Low-Rank Latent Projection (低秩路由子空间映射)|                |
|        |    z = P h \in R^r   (r << d, orthogonalized P P^T = I_r)|                |
|        |    Filters out high-dimensional isotropic noise         |                |
|        +---------------------------------------------------------+                |
|                                     |                                             |
|                                     v                                             |
|        +---------------------------------------------------------+                |
|        | 2. Multi-Anchor Expert Prototypes (多锚点专家原型表示)   |                |
|        |    Each Expert e has M low-rank anchors: {u_{e,m}}_{m=1}^M               |
|        +---------------------------------------------------------+                |
|                                     |                                             |
|                                     v                                             |
|        +---------------------------------------------------------+                |
|        | 3. Saturated Inner-Product Scoring (SIPS Lipschitz 控制) |                |
|        |    s_{e,m}(z) = \tau \cdot \tanh( <z, u_{e,m}> / (\tau \|z\|_\gamma) )   |
|        |    Explicitly bounds || \nabla_h s_e(h) ||_2 <= L_lip    |                |
|        +---------------------------------------------------------+                |
|                                     |                                             |
|                                     v                                             |
|             SoftMax / Top-k Selection ---> Stable Expert Dispatch                 |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **高维线性路由的三大几何病态**：标准稀疏 MoE 普遍采用单层线性投影 $s(h) = W_r h \in \mathbb{R}^N$ 作为路由器（Router）。作者从表示几何角度指出高维空间 $d \gg N$ 中的线性内积路由存在三大固有缺陷：
  1. **维度失配与噪声过拟合（Representation Mismatch）**：Token 隐状态 $h \in \mathbb{R}^d$ 包含了大量与任务路由无关的词法/位置高频噪声，全维内积导致路由决策极易受正交噪声方向干扰。
  2. **高维角度集中现象（Angular Concentration）**：随着层深增加，Transformer 隐状态落入狭窄的各向异性锥（Anisotropic Cone），不同专家路由向量与 $h$ 的余弦相似度高度趋同，导致门控分布扁平化或赢家通吃。
  3. **范数敏感与 Lipschitz 失控（Scale Sensitivity）**：当隐状态范数 $\|h\|_2$ 在深层或长序列中剧烈膨胀时，未受控的内积 $w_e^\top h$ 会使 Softmax 进入指数饱和区，微小输入扰动即可引发离散 Top-$k$ 路由集合翻转（Routing Instability）。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **共享低秩潜空间路由投影（Low-Rank Latent Routing Space）**：
   引入行正交低秩投影矩阵 $P \in \mathbb{R}^{r \times d}$（$r \ll d$，例如 $d=2048, r=64$），将隐状态 $h$ 压缩至低秩判别子空间：
   $$z = P h \in \mathbb{R}^r, \qquad \mathcal{L}_{\text{orth}} = \| P P^\top - I_r \|_F^2$$
2. **饱和内积打分与显式 Lipschitz 边界控制（Saturated Inner-Product Scoring, SIPS）**：
   为消除隐状态径向范数 $\|h\|_2$ 暴涨导致的路由震荡，L2R 设计了带阻尼范数归一化与双曲正切饱和的打分算子：
   $$\phi_{\text{SIPS}}(z, u_e) = \tau \cdot \tanh\left( \frac{\langle z, u_e \rangle}{\tau \left(\sqrt{\|z\|_2^2 + \epsilon^2}\right)^\gamma \left(\sqrt{\|u_e\|_2^2 + \epsilon^2}\right)^\gamma} \right)$$
   其中 $\tau > 0$ 控制饱和软边界，$\gamma \in [0, 1]$ 控制径向尺度不变性强度（当 $\gamma=1$ 时退化为受控余弦路由）。利用 $\text{sech}^2(x) \le 1$ 及正交投影 $\|P\|_2 = 1$，可严格证明打分函数对原始输入 $h$ 的梯度范数（即局部 Lipschitz 常数）存在显式解析上界：
   $$\left\| \nabla_h \phi_{\text{SIPS}}(P h, u_e) \right\|_2 \le \|P\|_2 \cdot \frac{\|u_e\|_2^{1-\gamma}}{\epsilon^\gamma} = L_{\text{lip}}$$
   从而从数学上保证了有界输入扰动 $\|\delta h\|_2 \le \delta$ 不会引发路由分数的剧烈跳变。
3. **多锚点专家表达（Multi-Anchor Routing）**：
   由于单个专家往往需要处理多模态或多子类语义簇，在低秩空间 $\mathbb{R}^r$ 中为每个专家分配 $M$ 个子锚点 $\{u_{e,m}\}_{m=1}^M \subset \mathbb{R}^r$（参数量仅为 $N \times M \times r \ll N \times d$），通过 Log-Sum-Exp 软聚合计算专家总得分：
   $$s_e(h) = \frac{1}{\beta} \log \sum_{m=1}^M \exp\Big( \beta \cdot \phi_{\text{SIPS}}(P h, u_{e,m}) \Big)$$

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* **语言与视觉双模态全面验证**：在基于 **OLMoE** 的语言模型预训练/微调以及 **ImageNet** 视觉 MoE 骨干网络上，L2R 将路由器参数量削减 **60%–75%**，同时在相同激活专家预算下将下游任务困惑度（PPL）降低 `0.42–0.68`，ImageNet Top-1 准确率提升 `+1.3%`。
* **路由稳定性与负载均衡双升**：在对抗性高斯扰动测试下，L2R 的 Top-$k$ 路由翻转率（Routing Flip Rate）比标准线性 Router 降低 **47%**，专家负载熵（Routing Entropy）更加接近理想均匀分布，无需强依赖破坏主任务梯度的大权重 Load-Balancing 辅助损失。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
1. **与 *Router-Tuning* (EMNLP 2025) & *Capacity-Aware Inference* (ICLR 2026) 的直接耦合**：
   * 我们在 *Router-Tuning* 中提出仅微调轻量路由器即可解锁深层稀疏网络潜力，但在极低资源或长上下文微调中，全维线性路由器容易过拟合表面范数特征。将 L2R 的 **SIPS + 低秩多锚点路由** 作为 *Router-Tuning* 的参数化形式，不仅能将可训练参数再降一个数量级，还能利用 Lipschitz 边界防止微调过程中的路由坍缩。
2. **与 *Transformer-Geometry* (`arXiv:2609.15975`, EMNLP 2026) & `MerA` SVD 初始化的深刻同构**：
   * L2R 发现的“径向范数敏感性（Scale Sensitivity）”与我们在 *Transformer-Geometry* 及 `ads-rsi`（定律 ADS-RSI-1：Scale-Cancellation）中揭示的**“深层残差流径向范数 $\|h\|_2$ 掩盖切向语义方向 $h / \|h\|_2$”**完全一致！此外，在将稠密模型或预训练线性路由器 $W_r \in \mathbb{R}^{N \times d}$ 转化为 L2R 路由器时，无需随机初始化 $P$，可直接调用我们的 **`MerA` 数据感知激活协方差 SVD（Activation-Covariance SVD）** 提取前 $r$ 个主奇异方向初始化 $P$，实现零冷启动抖动的低秩 Lipschitz 路由升级。

---

> [!TIP]
> **🎯 `SparseAdapter-MEO-PADNet` 仓库代码级落地点 (`Target Module`)**：`SparseAdapter` (Large-Sparse Expert Pool via Low-Rank Latent Bottleneck)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-27_ai_paper_notes.md`


---

### 2.2 [2026-09-27] OBCache: Optimal Brain KV Cache Pruning for Efficient Long-Context LLM Inference

* **论文信息**：Yuzhe Gu, Xiyu Liang, Jiaojiao Zhao, Enmao Diao (`arXiv:2510.07651`, **ICML 2026**)
* **核心关键词**：KV Cache Eviction、Optimal Brain Damage (OBD)、Second-Order Taylor Perturbation、Output-Aware Saliency、Joint KV Pruning

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|         OBCache: Optimal Brain Damage (OBD) Layer-Wise KV Cache Pruning           |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Prefill / Decoding Step: Queries Q \in R^{S_q x d_k}, Cached K, V \in R^{S_k x d}|
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 1. Attention Output Perturbation Objective (层输出二阶泰勒扰动建模)         |  |
|  |    Target: Minimize || O - \tilde{O}(\mathcal{M}) ||_F^2 where O = A V       |  |
|  |    Instead of heuristic \sum_i A_{i,j}, expand \Delta O w.r.t. masked K_j,V_j|  |
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|           +----------------------------+----------------------------+             |
|           v                            v                            v             |
|  +-----------------+          +-----------------+          +-------------------+  |
|  | Isolated Value  |          |  Isolated Key   |          | Joint KV Saliency |  |
|  | Score \Omega_j^V|          |  Score \Omega_j^K|         | Score \Omega_j^{KV}| |
|  | ||A_{:,j}||_2^2 |          | Softmax Jacobian|          | Exact Rank-1      |  |
|  | * ||V_j||_2^2   |          | Coupling Term   |          | Softmax Renorm    |  |
|  +-----------------+          +-----------------+          +-------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 2. Plug-and-Play Eviction Gate (即插即用淘汰门控: 兼容 SnapKV / PyramidKV)  |  |
|  |    Evict tokens with minimal \Omega_j^{KV} -> Retain top-B KV budget        |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **启发式注意力权重累加的理论缺陷**：主流长上下文 KV 缓存淘汰算法（如 H2O、SnapKV、PyramidKV）均使用累积注意力分数 $s_j = \sum_{i} A_{i,j}$ 作为 Token $j$ 的重要性指标。然而，注意力层真正传递给后续残差流的是加权输出矩阵 $O = A V \in \mathbb{R}^{S_q \times d_v}$：
  1. **忽略 Value 向量范数与方向抵消**：若某个历史 Token $j$ 的注意力权重 $A_{i,j}$ 较高，但其对应的 Value 向量范数 $\|V_j\|_2 \approx 0$，或者其 $V_j$ 与当前上下文均值方向完全重合，驱逐它对注意力输出 $O$ 的实际影响极小；反之，注意力权重中等但 $\|V_j\|_2$ 极大且承载正交关键信息的 Token 被驱逐后会造成严重的输出畸变。
  2. **忽略 Softmax 分母重归一化效应（Denominator Renormalization）**：驱逐第 $j$ 个 Key 相当于将注意力得分 $Z_{i,j} \to -\infty$，这不仅移除了 $A_{i,j} V_j$，还会通过 Softmax 分母缩放将其余所有保留 Token 的注意力权重放大 $\frac{1}{1 - A_{i,j}}$ 倍。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **基于 Optimal Brain Damage (OBD) 的二阶输出扰动构建**：
   设某注意力头在查询窗口 $Q \in \mathbb{R}^{S_q \times d_k}$ 下的注意力概率矩阵为 $A = \text{Softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{S_q \times S_k}$，输出为 $O = A V \in \mathbb{R}^{S_q \times d_v}$。定义驱逐准则为最小化层输出矩阵的 Frobenius 范数平方误差 $\mathcal{E} = \frac{1}{2} \| O - \tilde{O} \|_F^2$。
2. **单 Value、单 Key 与联合 KV 对的闭式显著性公式（Closed-Form Saliency Scores）**：
   * **孤立 Value 剪枝显著性（Isolated Value Saliency $\Omega_j^V$）**：
     当将第 $j$ 个 Token 的 Value 向量置零（$V_j \leftarrow 0$）时，$\mathcal{E}$ 对 $V_j$ 的海森矩阵（Hessian）为 $\mathbf{H}_{V_j} = \frac{\partial^2 \mathcal{E}}{\partial V_j \partial V_j^\top} = \left(\sum_{i=1}^{S_q} A_{i,j}^2\right) I_{d_v}$。根据二阶泰勒展开，孤立 Value 显著性得分为：
     $$\Omega_j^V = \frac{1}{2} V_j^\top \mathbf{H}_{V_j} V_j = \frac{1}{2} \| A_{:, j} \|_2^2 \cdot \| V_j \|_2^2$$
     注意此处注意力权重是**平方和 $\|A_{:,j}\|_2^2$**（二阶能量）而非启发式的线性求和 $\|A_{:,j}\|_1$，且显式乘上了 Value 范数平方 $\|V_j\|_2^2$！
   * **联合 KV 剪枝与 Softmax 重归一化修正（Joint KV Saliency $\Omega_j^{KV}$）**：
     当真正从缓存中移除第 $j$ 个 KV 对（即令未归一化 logit $Z_{i,j} \to -\infty$）时，剩余 Token $k \neq j$ 的注意力权重精确变为 $\tilde{A}_{i,k} = \frac{A_{i,k}}{1 - A_{i,j}}$。因此，移除第 $j$ 个 KV 对在第 $i$ 个查询位置引起的**精确输出残差**为：
     $$\Delta O_i^{(-j)} = O_i - \tilde{O}_i^{(-j)} = O_i - \frac{O_i - A_{i,j} V_j}{1 - A_{i,j}} = \frac{A_{i,j}}{1 - A_{i,j}} \big( V_j - O_i \big)$$
     对该精确残差在所有查询位置 $i \in \{1, \dots, S_q\}$ 上求二阶能量，即得到极其优雅的**联合 KV 闭式显著性得分**：
     $$\Omega_j^{KV} = \frac{1}{2} \sum_{i=1}^{S_q} \left( \frac{A_{i,j}}{1 - A_{i,j}} \right)^2 \big\| V_j - O_i \big\|_2^2$$

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* **即插即用全面提升主流基线**：在 **Llama-3.1-8B-Instruct**、**Qwen-2.5-7B/14B-Instruct** 与 **Mistral-7B** 上，将 OBCache 的 $\Omega_j^{KV}$ 闭式打分直接替换 H2O、SnapKV 与 PyramidKV 的启发式打分（零额外超参），在 **LongBench**（16 个长文本任务）与 **RULER**（128K 极限大海捞针与多跳追踪）上，在仅保留 **5%–10% KV 缓存预算**下将平均准确率提升 **`+2.8%` 至 `+6.4%`**。
* **计算开销近乎为零**：$\|V_j - O_i\|_2^2 = \|V_j\|_2^2 - 2 \langle V_j, O_i \rangle + \|O_i\|_2^2$ 可直接复用 FlashAttention 已经算出的输出向量 $O_i$，无需显式物化完整的 $S_q \times S_k$ 矩阵，Prefill 延迟增加小于 `1.2%`。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
1. **对我们 `vla-dtr` & `Efficient Ads / HisTrim` 中 `Exclude-Self Value-Space Perpendicular KV Pruning` 的精确二阶理论证明！**
   * 请仔细对比 OBCache 的核心公式 $\Omega_j^{KV} = \frac{1}{2}\sum_i \left(\frac{A_{i,j}}{1 - A_{i,j}}\right)^2 \|V_j - O_i\|_2^2$ 与我们在 `vla-dtr`（定律 5）和 `ads-rsi` 中独立提出的 **`Exclude-Self Value-Space Perpendicular VLM KV Pruning`**：
     * 其中的因子 $\frac{A_{i,j}}{1 - A_{i,j}}$ 正是**排除自身注意力权重后的重归一化系数（Exclude-Self Renormalization）**！
     * 其中的 $\|V_j - O_i\|_2^2$ 度量的正是第 $j$ 个 Token 的 Value 向量相对于当前聚合输出均值 $O_i$ 的**偏离能量（即正交/非共线奇异度）**！如果 $V_j \approx O_i$（即该 Token 的 Value 与上下文均值完全共线/冗余），即便 $A_{i,j}$ 再大，$\|V_j - O_i\|_2^2 \approx 0$，驱逐它也完全不改变注意力输出！
2. **落地融合方案（Perp-OBCache）**：
   * 在我们的论文撰写与代码实现中，可以直接引用 ICML 2026 的 OBCache 作为二阶泰勒理论背书，并指出我们进一步将 $\|V_j - O_i\|_2^2$ 投影到了输出投影矩阵 $W_O$ 之后的残差切空间 $\|(V_j - O_i) W_O P_\perp(h_i)\|_2^2$，从而构成了比 OBCache 更进一层的**流形正交切空间二阶最优脑缓存剪枝（Manifold-Orthogonal OBCache）**。

---

## 🔥 板块二：全球前沿热点精选 (Trending Frontier)

> **赛道锚点**：前沿研发智能体递归自我改进（Agent Harness RSI）、抗过拟合正则化进化、可执行代码物理世界模型（Code as Worlds）。

---

> [!TIP]
> **🎯 `SparseAdapter-MEO-PADNet` 仓库代码级落地点 (`Target Module`)**：`SparseAdapter` / `MEO` / `PAD-Net`  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-27_ai_paper_notes.md`


---

### 2.3 [2026-09-26] 🔄 *LoopMoE: Unifying Iterative Computation with Mixture-of-Experts for Language Modeling*
> **聚焦领域**：Looped Transformers · Mixture of Experts (MoE) · Iterative Depth Scaling · Weight Sharing  
> **arXiv**：[`arXiv:2606.04438`](https://arxiv.org/abs/2606.04438)

```
  输入表征 h^{(0)} ──► [ 循环步 t = 1..K : IterAdaLN(h, t) 轮次特征调制 ]
                                       │
                                       ▼
                     [ 共享 MoE 路由层: Top-k 稀疏专家激活 + 跨循环容量均衡 ]
                                       │
                                       ▼
                     [ 解耦总参数量 P 与单 Token 算力 FLOPs (同参数量 PPL 显著降低) ]
```

#### 🎯 背景与痛点剖析 (Problem Statement)
* **权重复用与轮次角色分化的矛盾**：在 Looped Transformer 中，直接将同一组 Transformer 块重复循环 $K$ 次，虽然能以 $O(1)$ 参数开销换取 $O(K)$ 的等效推理深度，但会导致两个严重退化：（1）不同循环步 $t \in \{1, \dots, K\}$ 缺乏步间身份区分，引发梯度震荡与隐状态平行分量 $\Delta h_\parallel$ 爆炸；（2）若将循环架构直接与 MoE 结合，不同循环步会争抢同一批头部 Expert，导致严重的跨循环路由坍缩（Cross-Loop Routing Collapse）。

#### 💡 核心方法与底层数学实现 (Mathematical Formulations)
1. **迭代步自适应层归一化 (Iteration-Adaptive LayerNorm, `IterAdaLN`)**：
   - 为第 $t$ 次循环引入轻量级步间嵌入向量 $e_t \in \mathbb{R}^d$，对共享主干的归一化层施加轮次特异性的仿射缩放与偏移调制：
     $$\text{IterAdaLN}(h^{(t)}, t) = \big(1 + \gamma(e_t)\big) \odot \frac{h^{(t)} - \mu}{\sigma} + \beta(e_t)$$
   - 通过仅占总参数量 $<0.1\%$ 的步间条件调制参数，赋予共享 MoE 块在不同循环深度下截然不同的几何变换角色。
2. **跨循环容量感知负载均衡 (Iteration-Aware Capacity Balancing)**：
   - 设第 $t$ 步第 $i$ 个专家的路由门控概率为 $p_i^{(t)}(x)$，论文将辅助负载均衡损失扩展至循环时间轴与批次维度的联合分布上，防止特定专家在连续多次循环中被重复饱和激活。

#### 📊 关键实验与结论 (Experiments & Findings)
* **等参数量与等 FLOPs 双向碾压**：在语言建模基准与常识推理任务上，循环 $K=2\sim 4$ 步的 `LoopMoE` 在相同活跃参数量下显著优于标准稠密 Looped 模型，且在相同总参数预算下逼近非共享深层 MoE 模型的困惑度（PPL）上限。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Relevance & Synergy with Our Works)
* **🎯 锚定代表作与在研主线**：
  * [Paper #16: *Disentangling Representation Evolution in Transformers through Directional Decomposition* (EMNLP 2026, `arXiv:2609.15975`)]
  * [Paper #11: *Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts* (ICLR 2026)]
  * [Paper #10: *Router-Tuning for Dynamic Mixture of Experts* (EMNLP 2025)]
  * [Active Line: *Physical AI / VLA-Loop (Stage-Wise Multi-LoRA Residual Boost & Adaptive Layer Looping)*]
* **🔬 机理对比与技术演进**：
  * `LoopMoE` 采用 `IterAdaLN`（逐通道对角缩放 $\gamma(e_t)$）来区分不同循环轮次；而我们在 `VLA-Loop`（见 W39 研发笔记 9/22–9/23）中提出**用极小秩的 Stage-Wise LoRA 去编辑共享主干的每一次循环**，并进一步推进到了**逐层自适应决定是否 Loop**；
  * 从我们 *Transformer-Geometry (EMNLP 26)* 的正交方向分解视角来看，`IterAdaLN` 仅在归一化后施加坐标轴缩放，主要调节平行缩放分量 $\Delta h_\parallel$；而我们的 **共享主干 + 轮次轻量 LoRA ($\Delta W_t = B_t A_t$)** 则能直接在子空间中引入低秩正交旋转分量 $\Delta h_\perp$，在表达能力上严格包含 `IterAdaLN`！
* **💡 下一阶段研究（Next Research Directions）落地启发**：
  * 在撰写 `Physical AI` (MLSys) 论文的 Loop 章节时，可将 `LoopMoE` 的 `IterAdaLN` 作为轻量轮次调制的文献对照基准，用实验展示我们 **“共享主干 + MERA 初始化的轮次小 LoRA + 逐层自适应 Loop 路由”** 相比单纯 LayerNorm 调制的显著几何表达优势。

---

> [!TIP]
> **🎯 `SparseAdapter-MEO-PADNet` 仓库代码级落地点 (`Target Module`)**：`SparseAdapter` (Step-Specific Low-Rank Residual Calibrators $A_t B_t$)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-26_ai_paper_notes.md`


---

### 2.4 [2026-09-26] ⚖️ *SelKV: Selective KV Cache Merging with Per-Token Merge-or-Drop and Attention Compensation*
> **聚焦领域**：KV Cache Compression · Softmax Denominator Compensation · Token Merging vs. Dropping  
> **arXiv**：[`arXiv:2607.16213`](https://arxiv.org/abs/2607.16213)

```
  待压缩历史 Token 序列 ──► [ 软余弦门控 (Soft Cosine Gate) 评估 Value 流形相似度 ]
                                       │
                        ┌──────────────┴──────────────┐
                        ▼                             ▼
             [ 高相似度: 加权合并 KV ]        [ 低相似度低重要度: 直接丢弃 ]
                        └──────────────┬──────────────┘
                                       ▼
               [ 注意力比率补偿 (Attention-Ratio Logit Compensation) ]
               消除 Softmax 分母塌陷 (Attention Sag) ──► 免训练高压缩保真
```

#### 🎯 背景与痛点剖析 (Problem Statement)
* **为什么免训练剪枝/合并会导致“注意力塌陷（Attention Sag）”**：当我们在推理期丢弃或合并大量历史 Token 后，参与 Softmax 计算的 Key 数量从 $N$ 锐减至 $M$（$M \ll N$）。若直接对剩余 $M$ 个 Token 的内积得分做标准 Softmax 归一化，原本被大量被删 Token 分担的分母配分函数质量消失，导致剩余 Token（或合并簇）的注意力权重被人为膨胀或失衡，深层表征模长发生剧烈偏移。

#### 💡 核心方法与数学推导 (Mathematical Formulations)
1. **软余弦门控决定“合并还是丢弃” (Soft Cosine Gate for Merge-or-Drop)**：
   - 给定被淘汰候选 Token $i$ 及其在保留集合中的最近邻锚点 $j^*$，计算其 Value 向量的余弦相似度 $s_i = \cos(v_i, v_{j^*})$；
   - 通过平滑门控函数 $g(s_i) = \sigma(\alpha (s_i - \tau))$ 动态决定将其特征并入锚点 $j^*$（当 $s_i > \tau$）还是直接丢弃（当 $s_i \le \tau$）。
2. **注意力比率对数补偿 (Attention-Ratio Compensation)**：
   - 若锚点 $j^*$ 吸收了等效计数为 $c_{j^*}$ 的历史 Token 质量，则在计算注意力 Logits 时显式加上对数质量补偿项：
     $$\tilde{a}_{q, j^*} = \frac{q^\top k_{j^*}}{\sqrt{d_k}} + \ln(c_{j^*})$$
   - 从而保证合并/剪枝前后的 Softmax 分母配分函数 $Z = \sum_j \exp(\tilde{a}_{q,j})$ 严格守恒！

#### 📊 关键实验与结论 (Experiments & Findings)
* 在 LongBench、RULER 及多轮数学推理基准上，免训练实现 **5x–10x KV Cache 压缩**，通过引入对数分母补偿项，消除了高压缩比下 80% 以上的精度退化。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Relevance & Synergy with Our Works)
* **🎯 锚定代表作与在研主线**：
  * [Active Line: *Efficient Ads & VLA `HisTrim` (Hierarchical Progressive Token Drop + Softmax Denominator Mass Compensation)*]
  * [Paper #15: *Demystifying When Pruning Works via Representation Hierarchies* (ICML 2026)]
  * [Paper #16: *Transformer-Geometry* (EMNLP 2026, `arXiv:2609.15975`)]
* **🔬 机理对比与技术演进**：
  * **这篇工作独立验证了我们本周在 `Efficient Ads` 与 `axon` FlashAttention 推导中发现的核心机制！** 我们在 W39 周记（9/21）中明确指出：**当丢弃 Token 后，若直接把剩余保留 Token 的注意力权重重新归一化到 100%，会引发 $>1\times$ 的权重膨胀（分母偏差 / Denominator Bias）**，并推导出了 FlashAttention LSE（$L_i = m_i + \ln \ell_i$）下的 `$+\ln(M)$` 对数配分函数补偿与特殊 Token（Attention Sink）保留机制；
  * `SelKV` 在免训练 KV 合并场景下观测到了完全相同的现象（其命名为 *Attention Sag*），并用 $+\ln(c_{j^*})$ 予以修正。
* **💡 下一阶段研究（Next Research Directions）落地启发**：
  * 在正在撰写的 `Efficient Ads`（冲刺 NAACL）正文中，可将 `SelKV` 与我们的分母偏差修正共同作为**“Token 稀疏化中的 Softmax 配分函数守恒定律”**的双向佐证，进一步强化我们把“分母偏差 $\leftrightarrow$ 位置编码与 Attention Sink”作为核心机制贡献（而非工程补丁）的理论厚度！

---

> [!TIP]
> **🎯 `SparseAdapter-MEO-PADNet` 仓库代码级落地点 (`Target Module`)**：`SparseAdapter` / `MEO` / `PAD-Net`  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-26_ai_paper_notes.md`


---

### 2.5 [2026-09-25] SAC: Disaggregated KV Cache Architecture for Sparse Attention Serving over CXL

* **论文信息**：`arXiv:2604.18392` (2026-04)
* **核心关键词**：CXL 3.0 Memory Pooling、Disaggregated KV Cache、Sparse Attention Sub-Page Gather

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|       SAC: CXL-Disaggregated KV Cache Architecture for Sparse Attention           |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  GPU Compute Nodes <--- CXL 3.0 Fabric ---> Shared CXL Memory Pool (TB-Scale KV)  |
|                                                       |                           |
|                                                       v                           |
|  +-----------------------------------------------------------------------------+  |
|  | Near-Memory Sparse Gather Engine on CXL Type-2/3 Controller                 |  |
|  |    Receives Top-k sparse token indices from GPU -> Packs only selected      |  |
|  |    cachelines into dense CXL flits -> 6.5x effective bandwidth amplification|  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **稀疏注意力在 PCIe/CXL 远端内存读取时的粒度放大（Granularity Amplification）**：当稀疏注意力仅需读取分散在不同物理页中的少量关键 Token 时，传统 DMA 以 4KB 页为单位搬运会导致高达 85% 的无效带宽浪费。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **CXL 控制器端近存稀疏聚集与头维度转置存储**：
   在 CXL 内存池侧按缓存行（64B Cacheline）对齐存储单头量化 KV 向量，由 CXL 控制器根据 GPU 下发的稀疏索引列表 $\mathcal{I}_{\text{top-}k}$ 在远端完成紧密打包（Dense Packing）后再经 CXL.mem 链路回传：
   $$\text{BW}_{\text{eff}} = \text{BW}_{\text{CXL}} \cdot \frac{d_{\text{head}} \cdot b_{\text{quant}}}{\lceil d_{\text{head}} \cdot b_{\text{quant}} / 64\text{B} \rceil \cdot 64\text{B}} \approx 0.94 \cdot \text{BW}_{\text{CXL}}$$

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在 TB 级长上下文并发推理中，SAC 将跨节点 KV 读取有效带宽利用率从 `15%` 提升至 **`94%`**，P99 尾延迟降低 **3.7x**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **为我们的 SelKV / OBCache 稀疏缓存算法在大规模分布式机架上的部署提供了硬件近存聚集蓝图**。

---

> [!TIP]
> **🎯 `SparseAdapter-MEO-PADNet` 仓库代码级落地点 (`Target Module`)**：`MEO` (CXL Near-Memory Sparse Cacheline Gathering for Disaggregated Memory)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-25_ai_paper_notes.md`


---

### 2.6 [2026-09-23] MELT: Memory-Efficient Looped Transformer — Decoupling Compute from Memory

* **论文信息**：`arXiv:2605.07721` (2026-05)
* **核心关键词**：Memory-Efficient Looped Transformer、Shared Cross-Loop KV Cache、Compute-Memory Decoupling

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|         MELT: Memory-Efficient Looped Transformer (Shared KV Cache Pool)          |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Standard Looped Transformer (K Loops):                                           |
|    Stores separate KV^{(1)}, KV^{(2)}, ..., KV^{(K)} -> K x Memory Footprint!     |
|                                                                                   |
|  MELT Architecture:                                                               |
|    Single Physical KV Cache Buffer \mathcal{C}_{KV} in HBM                        |
|    Loop k=1..K reads & refines \mathcal{C}_{KV} via gated EMA update:             |
|    \mathcal{C}_{KV}^{(k)} = (1 - \alpha_k) \mathcal{C}_{KV}^{(k-1)} + \alpha_k \text{Proj}_{KV}(h^{(k)})|
|    ===> O(K) Compute Depth with strictly O(1) KV Cache Memory!                    |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **循环 Transformer 的“隐性 KV 缓存倍增陷阱”**：虽然 Looped Transformer 通过复用层权重将模型参数显存压缩为 $1/K$，但在自回归生成时，如果第 $t$ 个 Token 在第 $k$ 次循环时需要 Attend 到前序 Token $1 \dots t-1$ 在第 $k$ 次循环时的键值状态，就必须为全部 $K$ 次循环分别缓存独立的 $K^{(k)}, V^{(k)}$，导致 KV 缓存显存依然随循环步数 $K$ 线性增长！

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **跨循环指数移动平均共享 KV 缓存（Cross-Loop EMA Shared KV Cache）**：
   对于历史已生成的上下文位置 $1 \dots t-1$，仅在显存中维护唯一一份最终收敛态的键值缓存 $(K_{\text{shared}}, V_{\text{shared}})$（即每个历史 Token 完成第 $K$ 次循环后的稳态 KV）。在当前位置 $t$ 执行第 $k \in \{1, \dots, K\}$ 次内部循环时，当前查询 $q_t^{(k)}$ 统一读取历史稳态缓存 $K_{\text{shared}, 1:t-1}$ 并结合当前步自键值 $(k_t^{(k)}, v_t^{(k)})$：
   $$\text{Attn}_t^{(k)} = \text{Softmax}\left( \frac{q_t^{(k)} \big[ K_{\text{shared}, 1:t-1}; \; k_t^{(k)} \big]^\top}{\sqrt{d_k}} \right) \begin{bmatrix} V_{\text{shared}, 1:t-1} \\ v_t^{(k)} \end{bmatrix}$$
   当第 $t$ 个 Token 完成全部 $K$ 步循环后，仅将其终端稳态 $(k_t^{(K)}, v_t^{(K)})$ 写入共享缓存池！

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在 $K=4$ 与 $K=8$ 循环配置下，MELT 将长文本解码时的 **KV 缓存显存与带宽读取量直接削减 $75\%–87.5\%$（严格降至 $1/K$）**，同时在语言建模与数学推理上与保存全套每步 KV 的基线性能完全持平（差异 `<0.2%`）。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **直接印证我们 `vla-loop` 定律 v19/v20（1-Pass Backbone + Multi-Step LoRA-Only Cascade & Shared KV Grounding）**：在 Looped VLA 中，历史观测与前缀只需保存唯一一份稳态 KV 缓存，多步循环仅更新当前动作查询状态，从而将循环推理的内存带宽开销降到最低。

---

> [!TIP]
> **🎯 `SparseAdapter-MEO-PADNet` 仓库代码级落地点 (`Target Module`)**：`MEO` (Decoupling Recurrent Compute Scaling from Activation/KV Memory Footprint)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-23_ai_paper_notes.md`


---

### 2.7 [2026-09-22] SPIN: Unifying Sparse Attention with Hierarchical Memory for Scalable Long-Context LLM Serving

* **论文信息**：`arXiv:2604.26837` (2026-04)
* **核心关键词**：Sparse Attention Serving、Hierarchical GPU-CPU Memory、Asynchronous Layer-Ahead Prefetching

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|       SPIN: Unifying Sparse Attention with Hierarchical Memory Serving            |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  GPU HBM: [Compact Page Indices + Hot Anchor KV Cache (10%)]                      |
|  CPU DRAM: [Full Cold KV Cache Pool (100%)]                                       |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | Layer l-1 Hidden State Speculative Index Prediction                         |  |
|  |    Predict Top-K sparse pages needed by Layer l BEFORE Layer l starts       |  |
|  |    Overlap PCIe/NVLink DMA prefetch of missing cold pages with Layer l-1 FFN|  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **动态稀疏注意力的 PCIe 按需拉取延迟陷阱**：若将全量 KV 缓存卸载至 CPU 内存并在每层动态选出 Top-$k$ 页面后才通过 PCIe 搬运回 GPU，PCIe 传输延迟将远超稀疏注意力节省的计算时间。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **跨层隐状态余弦惯性预取（Cross-Layer Speculative Page Prefetching）**：
   利用相邻层查询向量高度相似的几何惯性（$\cos(Q^{(l-1)}, Q^{(l)}) > 0.9$），在第 $l-1$ 层计算注意力的同时，使用轻量级页中心内积 $\hat{s}_p^{(l)} = Q^{(l-1)} \bar{K}_p^{(l)\top}$ 提前预测第 $l$ 层所需的冷页集合 $\mathcal{P}_{\text{miss}}^{(l)}$，实现计算与 PCIe DMA 搬运的完美流水线掩盖：
   $$T_{\text{step}}^{(l)} = \max\Big( T_{\text{FFN}}^{(l-1)} + T_{\text{QKV}}^{(l)}, \; \frac{|\mathcal{P}_{\text{miss}}^{(l)}| \cdot B_{\text{page}}}{\text{BW}_{\text{PCIe}}} \Big)$$

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在单台 8 卡服务器上支持 **1M–2M 上下文长度** 并发推理，相比纯 CPU Offloading（Infinite-LLM）实现 **4.8x** 吞吐提升，且恢复 99.7% 全量注意力精度。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **与我们 *Transformer-Geometry* (EMNLP 2026) 的层间方向平稳性定理天然契合**：正是因为深层残差流中平行分量占主导、层间角度旋转平缓，才保证了跨层提前 1–2 层预取稀疏 KV 页的高命中率！

---

> [!TIP]
> **🎯 `SparseAdapter-MEO-PADNet` 仓库代码级落地点 (`Target Module`)**：`SparseAdapter` / `MEO` / `PAD-Net`  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-22_ai_paper_notes.md`


---

### 2.8 [2026-09-20] SHIFT-LLM: Distribution Shift Correction in Depth-Pruned LLMs

* **论文信息**：`arXiv:2608.25068` (2026-08)
* **核心关键词**：Depth Pruning、Distribution Shift Correction、Linear Residual Adapters (LRA)、Closed-Form Ridge Regression、Weight Folding

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|          SHIFT-LLM: Closed-Form Distribution Shift Correction at Cut Sites        |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Original Stack:  h^{(l-1)} ---> [Pruned Block l..l+m] ---> h_{\text{orig}}^{(l+m)}|
|  Pruned Stack:    \tilde{h}^{(l-1)} -----(Identity Skip)---> \tilde{h}^{(l-1)}    |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 1. Covariate Shift Diagnosis at Pruning Cut Site (剪枝切口协变量偏移诊断)   |  |
|  |    \Delta \mu = \mathbb{E}[h_{\text{orig}}^{(l+m)} - \tilde{h}^{(l-1)}],    |  |
|  |    Angular & norm mismatch causes downstream RMSNorm / Attention saturation |  |
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 2. Closed-Form Linear Residual Adapter (LRA) via Woodbury/Ridge             |  |
|  |    \hat{h}^{(l+m)} = \tilde{h}^{(l-1)} + U_r V_r^\top \tilde{h}^{(l-1)} + b |  |
|  |    Solved in closed form on 128 calibration sequences (Training-Free)       |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **层剪枝切口处的“流形断裂（Manifold Fracture）”**：当直接移除 Transformer 中的第 $l$ 至 $l+m$ 层时，第 $l-1$ 层的输出隐状态 $\tilde{h}^{(l-1)}$ 被直接送入原本期望接收 $h_{\text{orig}}^{(l+m)}$ 的第 $l+m+1$ 层。由于缺失了中间层的残差漂移与旋转，输入分布的一阶均值 $\mu$ 与二阶协方差矩阵 $\Sigma$ 发生剧烈跳变，导致紧随其后的注意力层 Q/K 点积失真并沿着深层指数级放大。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **剪枝切口处的最小二乘残差重构**：
   设剪枝段输入隐状态矩阵为 $X = \tilde{H}^{(l-1)} \in \mathbb{R}^{N \times d}$，原始未剪枝模型在该切口输出的目标残差增量为 $\Delta Y = H_{\text{orig}}^{(l+m)} - \tilde{H}^{(l-1)} \in \mathbb{R}^{N \times d}$。SHIFT-LLM 在切口处插入一个低秩线性残差适配器（LRA）$W_{\text{LRA}} = U_r V_r^\top + \mathbf{1} b^\top$，通过带 Tikhonov 正则化的岭回归闭式求解全秩最优映射 $W^*$：
   $$W^* = \arg\min_{W \in \mathbb{R}^{d \times d}} \big\| \Delta Y - (X - \bar{X}) W \big\|_F^2 + \lambda \| W \|_F^2 = \Big( \tilde{X}^\top \tilde{X} + \lambda I_d \Big)^{-1} \tilde{X}^\top \Delta \tilde{Y}$$
2. **激活协方差加权奇异值截断（Covariance-Weighted Truncated SVD）**：
   为保证适配器自身的计算开销可忽略（或直接折叠进下一层权重），对预测输出空間执行白化 SVD 分解：
   $$\tilde{X} W^* = \hat{U} \hat{\Sigma} \hat{V}^\top \implies U_r = (\tilde{X}^\top \tilde{X} + \lambda I_d)^{-1/2} \hat{U}_{:, 1:r} \hat{\Sigma}_{1:r}^{1/2}, \quad V_r = \hat{V}_{:, 1:r} \hat{\Sigma}_{1:r}^{1/2}$$

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在 **Llama-3-8B/70B** 与 **Qwen-2.5-14B** 上剪除 **25%–35% 的层**后，无需任何梯度下降微调（仅需 30 秒闭式矩阵求逆），SHIFT-LLM 将 WikiText2 困惑度（PPL）从 `28.4` 恢复至 **`9.1`**，零样本常识与数学推理平均精度恢复 **`+7.9%`**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **与我们 `modellesion-compression-scaffold`、`vla-dtr` (Ortho-MerA) 及 *Layer Dropping* (TMLR 2025) 的直接印证**：
  * SHIFT-LLM 的闭式岭回归校正算子 $W^* = (\tilde{X}^\top \tilde{X} + \lambda I)^{-1} \tilde{X}^\top \Delta \tilde{Y}$ 与我们在 `modellesion-compression-scaffold` 中使用的 **Depth SVD-LoRA / Woodbury KKT 闭式残差补偿** 数学形式完全一致！更进一步，结合我们的 `vla-dtr`（Ortho-MerA），我们只需对正交切空间残差 $\Delta Y_\perp = \Delta Y \cdot P_\perp(X)$ 进行低秩 SVD 拟合，而将平行分量 $\Delta Y_\parallel$ 简化为标量增益 $\alpha \in \mathbb{R}$，即可用一半的秩恢复更高的几何保真度。

---

> [!TIP]
> **🎯 `SparseAdapter-MEO-PADNet` 仓库代码级落地点 (`Target Module`)**：`SparseAdapter` (Closed-Form Low-Rank Adapter Recovery after Structural Pruning)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-20_ai_paper_notes.md`


---

### 2.9 [2026-09-20] CARE: Spend Experts Where You Are Unsure — Confidence-Adaptive Routing for MoE-LoRA

* **论文信息**：`arXiv:2607.26052` (2026-07)
* **核心关键词**：Confidence-Adaptive Routing、MoE-LoRA、Nucleus Expert Activation、Router Uncertainty Entropy

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|            CARE: Confidence-Adaptive Routing for Mixture-of-Experts               |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Token Hidden State h_t ---> Router Probabilities p_t = Softmax(W_r h_t) \in \Delta^E|
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 1. Router Uncertainty Quantification (路由分布置信度/不确定性度量)          |  |
|  |    Sort probabilities: p_{t,(1)} >= p_{t,(2)} >= ... >= p_{t,(E)}           |  |
|  |    High confidence (peaked p_t) -> K_t = 1; High entropy -> K_t = K_{\max}  |  |
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 2. Nucleus & Margin-Gated Dynamic Top-K(t) Selection                        |  |
|  |    K_t = \min \{ k \in [K_{\min}, K_{\max}] : \sum_{i=1}^k p_{t,(i)} >= \tau_p|
|  |               \text{ or } p_{t,(k)} - p_{t,(k+1)} >= \tau_m \}              |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **静态 Top-$k$ 路由的算力错配**：标准 MoE 对序列中的每一个 Token（无论是标点符号、常见停用词，还是复杂的逻辑转折词）均无差别地激活固定数量 $k$ 个专家。对于路由器高度确信的简单 Token（例如 $p_{t,(1)} > 0.85$），强制拉起第 $2 \dots k$ 个低概率专家不仅浪费算力，还会引入长尾噪声干扰；而对于处于知识边界的模糊 Token，固定 $k$ 个专家又不足以覆盖多维语义假设。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **累积概率核与边际跳变双门控（Nucleus & Margin Gated Dynamic $K_t$）**：
   将排序后的专家门控概率记为 $p_{t,(1)} \ge p_{t,(2)} \ge \dots \ge p_{t,(E)}$。CARE 为每个 Token $t$ 动态分配激活专家个数 $K_t \in [K_{\min}, K_{\max}]$：
   $$K_t = \min \left\{ k \in \{K_{\min}, \dots, K_{\max}\} \;\middle|\; \sum_{i=1}^k p_{t,(i)} \ge \tau_{\text{nuc}} \;\;\lor\;\; \big(p_{t,(k)} - p_{t,(k+1)}\big) \ge \tau_{\text{margin}} \right\}$$
2. **零训练即插即用温度校准（Temperature Calibration under Global FLOPs Target）**：
   给定目标平均激活专家预算 $\bar{K}_{\text{target}}$，在校准集上通过单标量温度 $\beta$ 缩放路由 logits $p_t(\beta) = \text{Softmax}(W_r h_t / \beta)$，满足 $\mathbb{E}_t[K_t(\beta)] = \bar{K}_{\text{target}}$。

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在多任务 MoE-LoRA 与稀疏 MoE 语言模型上，CARE 在削减 **32%–45% 平均专家激活 FLOPs** 的同时，在常识推理、代码与数学基准上全面持平甚至超越固定 Top-$k$ 基线（`+0.9%` 平均准确率）。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **与我们 *Capacity-Aware Inference* (ICLR 2026) & *Router-Tuning* (EMNLP 2025) 的协同**：可将 CARE 的 Token 级置信度核门控（Nucleus Routing）与我们在 ICLR 2026 中提出的硬件容量感知丢弃/重路由（Capacity-Aware Dropping）级联，在软件置信度与硬件队列容量两个维度同时实现最优分配。

---

> [!TIP]
> **🎯 `SparseAdapter-MEO-PADNet` 仓库代码级落地点 (`Target Module`)**：`SparseAdapter` / `MEO` / `PAD-Net`  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-20_ai_paper_notes.md`


---

### 2.10 [2026-09-20] Minima-KV: Mixed-Format Paged Attention for Extreme KV Cache Compression

* **论文信息**：`arXiv:2608.23834` (2026-08)
* **核心关键词**：Mixed-Precision KV Cache、PagedAttention、Sub-Page Bit-Packing、Reasoning Continuity

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|        Minima-KV: Mixed-Format Paged Attention for Extreme KV Compression         |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Incoming KV Tokens ---> Saliency Tiering: [Tier-0: FP16] [Tier-1: INT4] [Tier-2: INT2]|
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 1. Unified Iso-Byte Physical Page Pool (等字节物理页统一内存池)             |  |
|  |    Each Physical Page = 64 KB fixed size:                                   |  |
|  |    * Can store N_0 FP16 tokens OR 4*N_0 INT4 tokens OR 8*N_0 INT2 tokens    |  |
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 2. Warp-Specialized Mixed-Format PagedAttention Kernel                      |  |
|  |    Single CUDA kernel dispatches dequantization per page descriptor header  |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **混合精度 KV 缓存的“页表碎片化与多核启动开销”**：虽然算法层已证明将关键 Token 存为 FP16、次要 Token 存为 INT4/INT2 可逼近无损压缩，但在 vLLM 等生产级 PagedAttention 系统中，传统的物理页（Page Block）按固定 Token 槽位数划分。若不同位宽的 Token 混存，会导致高达 40% 的页内字节对齐浪费（Internal Fragmentation），或被迫拆分为 3 次独立 CUDA Kernel 启动。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **等字节容量物理页抽象（Iso-Byte Physical Page Abstraction）**：
   固定每个物理页的字节容量为 $B_{\text{page}}$（如 64 KB）。对于位宽为 $b \in \{16, 4, 2\}$ 的页类型，其容纳的逻辑 Token 槽位数动态缩放为：
   $$C_{\text{slots}}(b) = \frac{8 \cdot B_{\text{page}}}{2 \cdot H_{kv} \cdot d_h \cdot b + M_{\text{meta}}(b)}$$
   其中 $M_{\text{meta}}(b)$ 为分组量化缩放因子与零点（Scale & Zero-Point）的紧凑页头字节数。
2. **页描述符驱动的单核融合反量化注意力（Single-Kernel Fused Dequant-Attention）**：
   在逻辑页表中增加 2-bit 格式标签 $\text{fmt}(p) \in \{0, 1, 2\}$，CUDA Warp 在读取物理页 $p$ 时根据 $\text{fmt}(p)$ 在寄存器内执行即时位解包（Register-Level Bit Unpacking）：
   $$\hat{K}_p = \text{Unpack}_{\text{fmt}(p)}(Q_p^K) \odot s_p^K + z_p^K, \qquad S_p = Q \hat{K}_p^\top$$

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在 **Llama-3.1-70B** 与 **Qwen-2.5-32B** 的 128K 长思维链并发服务中，Minima-KV 实现 **4.6x** 真实物理显存节省（零内部页碎片），将最大并发 Batch Size 提升 **3.9x**，端到端解码吞吐提升 **2.7x**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **直接解决我们昨日精读的 SelKV 与 `Efficient Ads / HisTrim` 混合位宽生产落地瓶颈**：可将我们的正交价值空间显著性打分（Perp-OBCache）作为 Minima-KV 的三档分层准则（FP16 / INT4 / INT2），直接集成进统一等字节页表内核中。

---

## 🔥 板块二：全球前沿热点精选 (Trending Frontier)

---

> [!TIP]
> **🎯 `SparseAdapter-MEO-PADNet` 仓库代码级落地点 (`Target Module`)**：`MEO` (Equal-Byte Page Pool & Register-Level Mixed-Precision KV Memory)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-20_ai_paper_notes.md`


---

### 2.11 [2026-09-18] 🧩 *MoE-Tile: Warp-Aligned Tensor Slicing for Zero-Overhead Dynamic Sparse Routing on Modern Accelerators*
> **聚焦领域**：Mixture of Experts (MoE) · GPU Kernel Optimization · Warp Divergence · Hardware-Aware Sparsity  
> **arXiv**：[`arXiv:2609.09112`](https://arxiv.org/abs/2609.09112)

```
  动态 Token 路由序列 ──► [ 块级非对齐碎片 (Warp 严重分化) ] ──► 计算利用率 38%
                                      │
                                      ▼
                      [ MoE-Tile 算子: Warp-Aligned 2D Slicing ]
                                      ├── 线程块内 128x128 Tile 对齐填充
                                      └── 零开销 TMA (Tensor Memory Accelerator) 异步流水
                                      ▼
                    [ 算子级 GEMM 吞吐达 89% 理论峰值 (提速 2.34x) ]
```

#### 🎯 背景与硬件级痛点剖析 (Hardware Bottleneck)
* **动态门控与 GPU 线程块的天然矛盾**：MoE 模型的 Top-$k$ 门控路由将不同数量的 Token 动态分发给不同 Expert。在 GPU 底层执行专家 FFN 矩阵乘（GEMM）时，每个 Expert 分配到的实际 Token 数（Batch $M_e$）并非硬件友好的 128 或 256 的倍数，导致大量 Warp 处于空转分化（Warp Divergence）状态，且引发非连续非对齐的显存搬运（Uncoalesced Memory Access），Tensor Core 实际利用率极低。

#### 💡 核心方法与原文底层工程实现 (Detailed System Mechanism)
1. **Warp 对齐二维分块调度器 (Warp-Aligned 2D Tile Slicer)**：
   - 设第 $e$ 个 Expert 接收到的 Token 数量为 $M_e$，隐藏维度为 $K$ 与 $N$；
   - 传统实现采用 Padding 将 $M_e$ 补齐到固定上界（造成显存与计算浪费），或采用 Ragged Batch（引发线程分化）；
   - 原文提出跨 Expert 全局排队与 Tile 重映射机制：将所有专家的计算任务切分为固定大小的硬件微块 $\mathcal{T}_{i,j} \in \mathbb{R}^{128 \times 128}$，将跨 Expert 的边界碎片（Tail Residuals）打包组合进统一的共享微块中执行。
2. **硬件 TMA 异步流水线重叠 (Asynchronous TMA Pipelining)**：
   - 利用现代 GPU（Hopper/Blackwell）的 Tensor Memory Accelerator（TMA），在 Shared Memory 与 Global Memory 之间构建三级流水线缓冲，将 Tile 索引寻址重排序开销完全隐藏在 FFN 计算延迟内部。

#### 📊 关键实验与结论 (Experiments & Findings)
* **硬件测试平台**：NVIDIA H100 80GB SXM5 与 B200 GPU 集群；
* **测试模型**：DeepSeek-V2/V3 (236B/671B)、Mixtral-8x22B；
* **实测性能**：
  * **算子级 GEMM 计算吞吐**：相比标准 Megatron-LM 与 vLLM MoE 算子，计算吞吐提升 **2.34 倍**，Tensor Core 利用率从 38.2% 提升至 **89.1%**；
  * **端到端端 Decode 延迟**：Token 生成阶段延迟降低 **43.5%**，完全消除了动态路由带来的硬件抖动。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Relevance & Synergy with Our Works)
* **🎯 锚定代表作**：
  * [Paper #11: *Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts* (ICLR 2026)]
  * [Paper #10: *Router-Tuning for Dynamic Mixture of Experts* (EMNLP 2025)]
  * [Paper #5: *MEO: Mixture of Experts Optimization* (EMNLP 2023)]
* **🔬 机理对比与技术演进**：
  * 我们在 *Capacity-Aware Inference (ICLR 26)* 中从**分布式全局宏观调度层面**定义了动态 Capacity Factor 与 Token 溢出分配策略；
  * *MoE-Tile* 则在**单卡底层 CUDA 算子与微架构 Tile 粒度**上解决了非规整 Token 批处理的执行开销；
* **💡 下一阶段研究（Next Research Directions）落地启发**：
  * 可将我们 ICLR 26 的全局分布式调度器与 MoE-Tile 底层 Triton/CUDA 算子进行纵向打通：由我们算法在上层输出动态均衡的 Expert 负载约束，下层由 MoE-Tile 执行 128 对齐的极速计算，构建从分布式集群到单卡底层内核的端到端超高效 MoE 推理栈。

---

> [!TIP]
> **🎯 `SparseAdapter-MEO-PADNet` 仓库代码级落地点 (`Target Module`)**：`SparseAdapter` / `MEO` / `PAD-Net`  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-18_ai_paper_notes.md`


---

### 2.12 [2026-09-18] 🗜️ *Decoupled-KV: Low-Rank Residual Decomposition for Multi-Turn Agentic KV Cache Compression*
> **聚焦领域**：KV Cache Compression · Agent Long-Context · Low-Rank Decomposition · Memory Bandwidth  
> **arXiv**：[`arXiv:2609.07765`](https://arxiv.org/abs/2609.07765)

```
  多轮对话/Agent 历史 KV ──► [ 低秩主基底空间 U_base (共享常驻) ] ──► 显存占用极小 (占 10%)
                                              │
                                              ▼
                             [ 动态稀疏时域残差 ΔK, ΔV ] ──► 8-Bit 符号压缩 (占 8.5%)
                                              │
                                              ▼
                              [ 总 KV Cache 显存削减 81.5% (保持 99.4% 精度) ]
```

#### 🎯 背景与痛点 (Problem Statement)
在自主智能体（AI Agents）与多轮长对话场景中，随着工具调用轨迹和历史环境反馈不断延长（达到 64K~256K tokens），KV 缓存占据超过 80% 的 GPU 显存。传统的基于 Token 丢弃的方法会丢失历史工具调用的精确参数信息，导致 Agent 多步规划频繁崩溃。

#### 💡 核心方法与数学公式 (Mathematical Formulations)
1. **低秩基础基底与时域稀疏残差解耦 (Decoupled Representation)**：
   - 将跨轮次的 Key 张量 $K \in \mathbb{R}^{T \times d}$ 解耦为静态系统提示/工具定义的共享低秩基底 $U_{\text{base}} \in \mathbb{R}^{d \times r}$（$r \ll d$）与动态增量残差：
     $$K = Z U_{\text{base}}^T + \Delta K, \quad \text{其中 } \|\Delta K\|_0 \le s \cdot (T \times d)$$
2. **正交残差追踪与高效重构**：
   - 对 $U_{\text{base}}$ 保持全精度常驻显存，对稀疏残差 $\Delta K, \Delta V$ 执行 1.5-bit 量化编码与行稀疏存储，在注意力计算时通过轻量 Fused Kernel 瞬时还原。

#### 📊 关键实验与结论 (Experiments & Findings)
* 在 AgentBench、SWE-bench 与 LongBench 上，实现 **81.5% 的 KV Cache 显存削减（压缩比达 $5.4\times$）**，长程任务规划成功率保持在全量缓存基准的 **99.4%**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Relevance & Synergy with Our Works)
* **🎯 锚定代表作**：
  * [Paper #13: *EffiR: Making Large Language Models Efficient Dense Retrievers* (ACL 2026)]
  * [Paper #3: *SparseAdapter: Parameter-Efficient Fine-Tuning* (EMNLP 2022)]
* **🔬 机理对比与技术演进**：
  * 我们在 *EffiR (ACL 26)* 中探索了低维稠密向量对齐与检索压缩；
  * 本文证明了多轮 Agent 交互中 Key/Value 张量内部存在极强的低秩共享子空间与稀疏残差分离特性；
* **💡 下一阶段研究（Next Research Directions）落地启发**：
  * 可将该低秩残差分解架构应用于我们 Dense Retriever 的长文档向量索引中，将向量库内存开销直接削减 80%。

---

## 🔥 板块二：全球前沿热点精选 (Trending Frontier)

---

> [!TIP]
> **🎯 `SparseAdapter-MEO-PADNet` 仓库代码级落地点 (`Target Module`)**：`SparseAdapter` / `MEO` / `PAD-Net`  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-18_ai_paper_notes.md`


---
