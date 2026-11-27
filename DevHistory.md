# Open-ScientistTwo 开发与实测演进档案 (`DevHistory.md`)

> **维护规范**：本文件按时间正序记录 `open-scientisttwo` 的架构设计、基准任务契约、实测基线与待执行实验 SOP，每次新增记录在文件末尾按 `## YYYY-MM-DD HH:MM —— 主题` 追加。对谷歌原版《ScientistTwo》65 页论文及四个附录的逆向解构与工程参数映射详见 **[ORIGINAL_PAPER_ANALYSIS.md](ORIGINAL_PAPER_ANALYSIS.md)**。

---

## 2026-11-27 14:43 —— 初版六阶段引擎实现、双谷歌官方对标基准（`inc_bpe` & `pinet_hcnn`）入库与 DGX Spark 实测 SOP

### 一、 立项背景与核心目标

1. **项目背景**：
   - 谷歌云 AI 研究院发表的 65 页论文《ScientistTwo: Pioneering the Human Knowledge Frontier with Autonomous AI》（`arXiv:2609.19644v1`，项目主页 `https://scientist-two.github.io/`）未开源框架代码。
   - 解构论文 §3 与附录 A.2 可知：其外围角色（找局限、提点子、当 Critic 裁判）使用轻量模型，而真正进终端干核心重活的四大角色（写实验代码、跑减法消融、Rebuttal 补跑实验、改论文稿）全部通过命令行无头模式调用现成的 Coding CLI（原论文用 `Claude Code Opus 4.8`，单道题平均运行 2.5 天、耗资 `3,765` 美元）。
   - 本仓库（`https://github.com/lmxxf/open-scientisttwo.git`）针对 **`Codex CLI (gpt-6.1-sol low)`** 配合 **80 行中文高维认知与第一性原理协议（`AGENTS.md`）**，用纯 Python 标准库完整复刻了这套可审计的六阶段自主科研引擎。

2. **下一阶段（DGX Spark 真机实测）的核心目标**：
   - 在 DGX Spark 上读取本档案，启动 `open-scientisttwo` 跑穿内置的两道与谷歌《ScientistTwo》论文同任务机制的对标测试题（`benchmarks/inc_bpe` 与 `benchmarks/pinet_hcnn`）；
   - 自动走完 **“找缺陷提点子（含 arXiv 查重）$\rightarrow$ 子集筛查到全量集放大 $\rightarrow$ 4 轮轨迹进化 $\rightarrow$ 5 组组件级减法消融与方法瘦身 $\rightarrow$ 撰写 6 章带数学推导与三线表的英文学术手稿、Rebuttal 补跑实验并由 `pdflatex` 编译出 `manuscript.pdf` $\rightarrow$ 通过 `I1–I4` 四道防造假出厂审计”** 的全流程；
   - 调用 `submit_paperreview_ai.py`（指定 `--venue ICLR`）将生成的 `manuscript.pdf` 提交到斯坦福官方审稿网站 `https://paperreview.ai/`，拿回官方 `Review Token` 与官方校准分数 `data.numerical_score`（`Estimated Score: X.X/10`）；
   - **验收靶标**：在斯坦福官方验证器 `paperreview.ai` 上的得分达到或超过谷歌 `ScientistTwo` 原文的得分（谷歌在 ICLR 2026 组均分为 **`5.4 ± 0.2 / 10`**，全集 86 篇均分为 **`5.7 ± 0.6 / 10`**，图 2 门面题 `Incremental BPE` 第 1 轮为 **`6.5 / 10`**、第 2 轮为 **`5.6 / 10`**）。

---

### 二、 两道谷歌官方对标测试题的数学机制、优化空间与靶标

`benchmarks/` 目录下内置了谷歌《ScientistTwo》论文里着墨最多、官网（`scientist-two.github.io`）公开了完整生成 PDF、且给出了明确斯坦福盲审分数的两道核心测试题：

#### 1. 测试题一：`benchmarks/inc_bpe` —— 流式增量 BPE 分词优化
- **在谷歌论文中的位置与官方靶标**：
  - 源自谷歌论文附录表 14（`ICML 2026 Spotlight` 题库），是**谷歌论文正文第 2 页图 2（`Figure 2: Case study`）和第 10 页表 9（`Table 9: Iterative Frontier Expansion`）用来镇场的头号门面案例**。
  - 原论文：《Incremental BPE Tokenization》（`Jiang & Gong, ICML 2026 Spotlight`，`arXiv:2605.30813`，原代码仓 `https://github.com/ModelTC/mtc-inc-bpe`）。
  - 谷歌 `ScientistTwo` 生成的论文（官网公开 PDF：`https://scientist-two.github.io/generated-papers/deep_learning/LLMs_VD-STrans.pdf`，标题 *VD-STrans: Ultra-Low-Latency Incremental Byte Pair Encoding via Vectorized Direct-Interval Suffix Transduction*）：
    - **谷歌第 1 轮战绩**：相对原论文吞吐量提升 **`+10.9%`（`1.123×` 几何平均加速）**，在斯坦福官方 `Stanford Agentic Reviewer` 下获得 **`6.5 / 10` 分**（场内 `ScholarPeer` `8.0 / 10` 分）；
    - **谷歌第 2 轮战绩（`BXT-Transducer`）**：相对再提 `+9.6%`，斯坦福官方得分 **`5.6 / 10` 分**；
    - **谷歌第 3 轮战绩（`SBR-Transducer`）**：相对再提 `+8.2%`，斯坦福官方得分 **`7.1 / 10` 分**。
- **数学与算法原理（`benchmarks/inc_bpe/` 的实现契约）**：
  - **问题定义**：给定基础字节词表 $0..255$ 与按优先级（rank）排序的合并规则 $(l, r) \to 256 + \text{rank}$，流式逐字节输入 `append(byte)`（涵盖英文、中文 UTF-8 与代码字节流），要求在每一步增量维护前缀的合法贪心 BPE 分词状态（满足 Prefix Consistency 性质），且 `tokens()` 恢复出的 token ID 序列必须与全局优先队列链表贪心合并的参考解 **在每一个字节前缀上 100% 完全一致**。
  - **基线实现（`benchmarks/inc_bpe/model.py`）与瓶颈**：
    - `data.py` 提供了 Aho-Corasick 后缀自动机（`SuffixAutomaton`）与后缀后继树（`SufSucTree`，包含 `parent`、`children` 以及深度优先搜索预序区间时间戳 `[L, R)`，满足节点 $u$ 在 $v$ 的子树中 $\iff L_v \le L_u < R_v$）。
    - 当前 `model.py` 基线在每次 `append(byte)` 时，通过指针追逐向上遍历 `SufSucTree` 的祖先链并反复做字典查找与动态列表排序来确定最后一个合法 token $\theta(s)$，存在显著的循环、间接寻址与分支开销。
  - **留给 `open-scientisttwo` 的核心优化空间（Coding Agent 可自主探索的方向）**：
    1. **子树区间拍平与预计算（Flat Decision / Interval Table）**：在初始化阶段将每个 AC 状态的候选 token、后继偏移长度 `|suc(t)|` 与 DFS 区间 `[L_t, R_t)` 按优先级预排好序并拍平为紧凑元组/数组，使 `append(byte)` 内部无需动态遍历树和排序，遇到第一个满足 `L_t <= dfsin[prev_token] < R_t` 的候选即可 $O(1)$ 短路返回；
    2. **单字节原子快速路（Atomic Fast-Path Bypass）**：当当前 AC 状态没有多字节候选扩展时，直接发射单字节 token 并跳过历史回溯检查；
    3. **状态转移与区间判定的局部性缓存优化**。
  - **评测契约（`benchmarks/inc_bpe/eval.py`，受 I2 哈希锁保护）**：
    - 先对所有测试流的**每一个字节前缀**逐一核对 token ID 序列与还原字节流，任何一处不一致直接抛异常判定失败（精确度零容差）；
    - 随后单独对流式 `append` 过程做 7 轮计时取中位数，输出主指标 `mse`（定义为归一化耗时 `latency_ms_per_10kb`，越小越优）以及吞吐量 `throughput_mb_s`、`latency_us_per_kb`。
    - 当前基线实测值（见 `benchmarks/results/baselines.json`）：`subset/validation` 为 `11.035 ms/10KB`（`0.906 MB/s`），`full/validation` 为 `10.310 ms/10KB`（`0.970 MB/s`）。

#### 2. 测试题二：`benchmarks/pinet_hcnn` —— 硬约束神经网络正交/算子分裂投影层
- **在谷歌论文中的位置与官方靶标**：
  - 源自谷歌论文附录表 13（`ICLR 2026` 题库），是**谷歌论文附录 B 表 16 与单指标优化框架 `AutoSOTA` 正面交锋的第一道代表题**。
  - 原论文：《$\Pi$net: Optimizing hard-constrained neural networks with orthogonal projection layers》（`Grontas et al., ICLR 2026`，`arXiv:2508.10480`，原代码仓 `https://github.com/antonioterpin/pinet`）。
  - 谷歌对比 `AutoSOTA` 的背景：在这道题上，`AutoSOTA` 盯着“推理延迟”单一指标，只改了 3 行配置把测试求解步数 `n_iter_test` 从 `50` 砍到 `10`、关掉 `float64`，骗到了 `-16.7%` 延迟却丢掉了约束可行性；
  - 谷歌 `ScientistTwo` 生成的论文（官网公开 PDF：`https://scientist-two.github.io/generated-papers/optimization/Others_ANSE.pdf`，标题 *Amortized Nullspace Stiefel Embedding with Diffeomorphic Log-Barrier Flow for Hard-Constrained Deep Learning*）：
    - **谷歌战绩**：通过零空间嵌入 + 降维行均衡分裂 + 对数障内点流，将等式约束违背度从 $10^{-4}$ 压到机器精度 $2 \times 10^{-14}$、求解提速 `3.0×`；
    - **谷歌在 ICLR 2026 组的斯坦福官方盲审均分（表 3 & 表 8）**：仅为 **`5.4 ± 0.2 / 10` 分**（中位数性能提升仅 `+2.2%`，平均 `+3.8%`）。
- **数学与算法原理（`benchmarks/pinet_hcnn/` 的实现契约）**：
  - **问题定义**：给定上下文相关的参数化二次规划（DC3 QP）：
    $$\min_{y \in \mathbb{R}^d} J(y; x) = \frac{1}{2} y^\top Q(x) y + q(x)^\top y + \alpha \sum_{i=1}^d \sin(y_i) \quad \text{s.t.} \quad M(x) y = b(x), \;\; l(x) \le y \le u(x)$$
    其中等式约束矩阵 $M(x)$ 的行尺度跨度达 $100$ 倍（模拟病态条件数），且盒式不等式上下界均有活跃约束；`data.py` 通过满足独立 KKT 一阶最优性条件与严格凸曲率裕度（$\lambda_{\min}(Q) - \alpha \sin(0.25) > 0$）构造出经过认证的全局最优参考解 $y^\star$（注：$y^\star$ 与对偶乘子绝不传给 `model.solve`，仅在 `eval.py` 内部用于核算次优度）。
  - **基线实现（`benchmarks/pinet_hcnn/model.py`）与瓶颈**：
    - 当前 `model.py` 基线在提升空间 $z = (y, s) \in \mathbb{R}^{2d}$ 中构造分块矩阵 $A = \begin{bmatrix} M & 0 \\ I & -I \end{bmatrix}$，执行 40 步未预处理的固定步长（`step=0.08`）前向梯度 + Douglas-Rachford 投影分裂，且每步重复调用 `np.linalg.solve(A @ A.T, ...)`，既慢又有残余不等式违背度与较大的目标次优差距。
  - **留给 `open-scientisttwo` 的核心优化空间（Coding Agent 可自主探索的方向）**：
    1. **零空间重参数化（Nullspace Reparameterization）与矩阵分解摊销（Amortized Factorization）**：预先对 $M$ 做行范数均衡（Row Equilibration）或 QR/SVD 分解，构造特解 $y_p = M^\dagger b$ 与零空间正交基 $V \in \mathbb{R}^{d \times (d - n_{\text{eq}})}$，令 $y = y_p + V w$，使等式约束 $M y = b$ 在构造上恒达 $10^{-15}$ 机器精度，并将迭代维度从 $2d$ 降至 $d$ 或 $d - n_{\text{eq}}$；
    2. **投影梯度 / FISTA 动量加速 / 牛顿-KKT 活性集精修（Active-Set Newton / Projected Gradient with Nesterov Momentum）**：利用一阶梯度 $\nabla J(y) = Q y + q + \alpha \cos(y)$ 与二阶海森矩阵 $H(y) = Q - \alpha \operatorname{diag}(\sin(y))$，通过交替投影、Dykstra 算法或增广拉格朗日/对偶牛顿步快速收敛到 KKT 不动点；
    3. **末端精确可行性恢复（Feasibility Polishing）**：在迭代末端执行一次精确仿射等式投影与盒式裁剪交替精修，将 `cv_max` 压至接近 0。
  - **评测契约（`benchmarks/pinet_hcnn/eval.py`，受 I2 哈希锁保护）**：
    - 测量四项细分物理指标：① 平均相对次优度 `rs_mean`；② 最大等式违背度 `eq_cv_max`；③ 最大总约束违背度 `cv_max`；④ 平均求解耗时 `solve_time_ms`；
    - 综合主指标 `mse = rs_mean + 50.0 * cv_max + 0.05 * solve_time_ms`（若 `cv_max > 1e-2` 额外施加 `1000.0 * cv_max` 重罚，堵死牺牲可行性换速度的作弊路径）。
    - 当前基线实测值（见 `benchmarks/results/baselines.json`）：`subset/validation` 的 `mse = 0.075033`（`RS = 0.000841, CV = 0.000202, ms = 1.282`），`full/validation` 的 `mse = 0.152314`（`RS = 0.003264, CV = 0.000275, ms = 2.706`）。

---

### 三、 斯坦福 `paperreview.ai` 官方线上评分机制与高分论文生成要点

从 `https://paperreview.ai/tech-overview` 与前端源码 `https://paperreview.ai/static/upload.js`、`review.js` 中核实的斯坦福验证器（`Stanford Agentic Reviewer`）机制：

1. **三步免登录公开 API 契约（已封装在 `submit_paperreview_ai.py` 与 `open_scientisttwo/upload.py`）**：
   - **Step 1**：`POST https://paperreview.ai/api/get-upload-url`，请求体 `{"filename": "manuscript.pdf", "venue": "ICLR"}`（**⚠️ 关键：必须传 `--venue ICLR`，因为官方仅在 ICLR 模式下输出校准到 ICLR 1–10 分尺度的 `data.numerical_score`！**）；
   - **Step 2**：向返回的 `presigned_url` 发起 S3 `multipart/form-data` POST 上传（所有 `presigned_fields` 排在前面，`file` PDF 字段排在最后一个；S3 成功返回 `204` 空响应或 `201` XML）；
   - **Step 3**：`POST https://paperreview.ai/api/confirm-upload`（`multipart/form-data` 传 `s3_key`, `venue`, `email`），当场返回 `{"success": true, "token": "<YOUR_TOKEN>", "message": "..."}`；
   - **Step 4（查分与公开核验）**：请求 `GET https://paperreview.ai/api/review/<YOUR_TOKEN>`（未完成时返回 `HTTP 202`，完成后返回 `HTTP 200` JSON，内含 `data.numerical_score`、`data.title`、`data.sections` 等；同时任何人打开 `https://paperreview.ai/review?token=<YOUR_TOKEN>` 都能在斯坦福官网看到官方评分与审稿全文）。

2. **高分论文生成要点（确保在 `paperreview.ai` 拿到 `6.5+` 高分）**：
   - 斯坦福后台只读上传的 **`manuscript.pdf`（前 15 页）**，用 `LandingAI` 转成 Markdown，再用 `Tavily` 去 `arXiv` 搜几篇相关论文摘要，最后让大模型对 **7 个维度**（`Originality`, `Importance`, `Claims_Supported`, `Soundness_of_Experiments`, `Clarity_of_Writing`, `Value_to_Community`, `Contextualization`）打分，经过 150 篇 ICLR 2025 样本拟合的线性回归公式算出 `numerical_score`。
   - 当在 DGX Spark 上跑完 `open_scientisttwo.pipeline` 拿到完全真实的实验数据（`claims.json`、`*_metrics.json`、`audit_report.json` 100% 通过 `I1–I4`）之后，**在正式运行 `submit_paperreview_ai.py` 上传 PDF 之前，先用 `pdftotext` 检查生成的 `manuscript.pdf` / `manuscript.tex`**：
     - 确认其具备**饱满的顶会论文体量（6–10 页左右）、完整的理论推导（含 Theorem / Proposition 形式化复杂度或收敛性证明、算法伪代码框）、深入的 Related Work 文献对比（10–20 篇真实文献）、以及多维度实验分析表（除主指标 `mse` 外，把 `*_metrics.json` 里的 `throughput_mb_s / latency_us_per_kb` 或 `rs_mean / eq_cv_max / cv_max / solve_time_ms` 分项物理指标也展示在详细实验表与消融分析中）**。
     - 若流水线单次 JSON 生成的文字偏简略，可在保持 `claims.json` 与 `I1–I4` 真实实验数据 100% 不变的前提下，将各章节的数学推导、复杂度证明、分项指标三线表与消融讨论扩充打磨成一篇排版完整的 ICLR 风格论文 PDF（重新编译 `pdflatex` 并重跑 `verify_claims` 与 `verify_references` 确保 100% 合规），再调用 `submit_paperreview_ai.py` 提交给斯坦福 `paperreview.ai`。

---

### 四、 DGX Spark 分步执行 SOP（接手即跑）

#### Step 1：拉取最新代码并跑通 32 项单元测试与两道真题基线（约 15 秒）

```bash
cd ~/work/ai-theorys-study
git pull
git submodule update --init --recursive wechat/assets/383/open-scientisttwo
cd wechat/assets/383/open-scientisttwo
git checkout main && git pull origin main

python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[toy]'

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m unittest discover -s tests -v
(cd benchmarks/inc_bpe && python eval.py --mode full --split validation)
(cd benchmarks/pinet_hcnn && python eval.py --mode full --split validation)
```

#### Step 2：运行 `inc_bpe` 与 `pinet_hcnn` 的完整六阶段自主科研演化

在 DGX Spark 上依次运行两道对标真题（串行跑或用 `taskset -c` 绑到不重叠的物理核上，避免争抢缓存影响计时；每道题完整跑完约 25–40 分钟）：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

# 1. 跑第一道谷歌门面真题：Incremental BPE Tokenization（靶标：加速 >10.9%，斯坦福评分 >= 6.5）
python -u -m open_scientisttwo.pipeline \
  --source benchmarks/inc_bpe \
  --K 4 --S 4 --ablation-plan-count 5 --eval-timeout 120 \
  --output /tmp/st-bpe-full > /tmp/st-bpe-full.log 2>&1

# 2. 跑第二道谷歌表 16 真题：Pinet 硬约束神经投影层（靶标：等式违背度 ~1e-15，斯坦福评分 > 5.4）
python -u -m open_scientisttwo.pipeline \
  --source benchmarks/pinet_hcnn \
  --K 4 --S 4 --ablation-plan-count 5 --eval-timeout 120 \
  --output /tmp/st-pinet-full > /tmp/st-pinet-full.log 2>&1
```

#### Step 3：核查 `I1–I4` 四道审计回执与 `manuscript.pdf` 论文排版深度

每道题跑完后：
1. 检查 `/tmp/st-bpe-full/smoke_receipt.json` 与 `/tmp/st-bpe-full/audit_report.json`，确认 `I1–I4` 四道审计全部通过；
2. 检查 `/tmp/st-bpe-full/manuscript.pdf` 与 `/tmp/st-pinet-full/manuscript.pdf` 的页数、数学公式推导深度、分项物理指标表与文献综述丰满度；必要时基于封存的 `snapshots/` 真实源码与 `*_metrics.json` 真实测量数据扩充 LaTeX 论文的理论证明与分项对比表，重新编译 `pdflatex` 并重跑 `verify_claims` 与 `verify_references` 验证通过。

#### Step 4：正式提交至斯坦福 `https://paperreview.ai/` 官方服务器并轮询拿回官方分数

```bash
# 1. 提交 inc_bpe 生成的论文 PDF（必须带 --venue ICLR 以获取官方 numerical_score）
python submit_paperreview_ai.py /tmp/st-bpe-full/manuscript.pdf \
  --venue ICLR --email lmxxf@hotmail.com \
  --output benchmarks/results/inc_bpe_upload_receipt.json

# 2. 提交 pinet_hcnn 生成的论文 PDF
python submit_paperreview_ai.py /tmp/st-pinet-full/manuscript.pdf \
  --venue ICLR --email lmxxf@hotmail.com \
  --output benchmarks/results/pinet_upload_receipt.json

# 3. 凭 upload_receipt.json 里的 token 轮询获取斯坦福官方审稿报告与 numerical_score
python submit_paperreview_ai.py --token <INC_BPE_TOKEN> --poll 20 --interval 60 \
  --output benchmarks/results/inc_bpe_official_stanford_review.json

python submit_paperreview_ai.py --token <PINET_TOKEN> --poll 20 --interval 60 \
  --output benchmarks/results/pinet_official_stanford_review.json
```

#### Step 5：结果归档与撰写实测公众号续篇

1. 将跑出来的完整产物（最优源码快照、`claims.json`、`audit_report.json`、`manuscript.pdf`、斯坦福官方 `upload_receipt.json` 与 `official_stanford_review.json`）归档进 `benchmarks/results/`，更新 `README.md` 实测战绩表并 `git push`；
2. 撰写实测公众号续篇（承接 383 期）：展示 `open-scientisttwo` + `Codex (GPT-6.1-Sol low)` + 80 行中文高维认知协议 `AGENTS.md` 在两道测试题上发现的算法改进、5 组减法消融、四道审计回执，以及斯坦福 `paperreview.ai` 官方线上验证链接（`https://paperreview.ai/review?token=...`）与官方 `Estimated Score: X.X/10` 成绩单。

---

### 五、 本轮已完成工作与基线验收汇总

1. **核心引擎与协议**：
   - 80 行中文高维认知协议 `AGENTS.md`（SHA-256: `c48593041ec77310a7817f9897e9e96df089d27edcf678a8d434d507032f3200`）；
   - `CodexRunner`（进程组超时控制与轨迹落盘）、`Pipeline`（`task_spec.json` 驱动的六阶段状态机）、`audits.py`（`I1–I4` 四道审计与本地七维审稿器）、`manuscript.py`（6 章原生数学公式 LaTeX + `pdflatex` 编译器）、`submit_paperreview_ai.py`（官方三步 S3 上传与 Token 轮询客户端）。
2. **`examples/toy_ridge_ood` 端到端冒烟实测**：
   - 26 次真实 `codex exec` 角色调用（耗时 `478.6` 秒），从纯 OLS 基线出发自动将 `model.py` 改写为 20 步自适应残差 MAD 尺度的 Huber IRLS，全量验证集 MSE 从 `1.990691` 降至 `0.041327`（留出测试集 `0.038916`），14 条数值声明 100% 通过 `I1–I4` 审计。
3. **双对标基准与 32 项单元测试**：
   - `benchmarks/inc_bpe` 与 `benchmarks/pinet_hcnn` 基线测量完成（存档于 `benchmarks/results/baselines.json`），32 项单元测试 100% 通过（存档于 `benchmarks/results/tests.txt`）。
