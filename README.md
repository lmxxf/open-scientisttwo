# open-scientisttwo

基于 **`Codex CLI (gpt-6.1-sol low)`** 与**高维认知协议（[`AGENTS.md`](AGENTS.md)）**的可审计六阶段自主科研引擎。

本项目参照论文《ScientistTwo: Pioneering the Human Knowledge Frontier with Autonomous AI》（[arXiv:2609.19644](https://arxiv.org/abs/2609.19644)，§3、附录 A.2 及 Table 7）公开的多阶段状态机，通过声明式任务配置（`task_spec.json`）驱动任意领域的算法改进任务，内置组件级减法消融、四道防造假出厂审计（`I1–I4`）、六章节原生数学公式 LaTeX/PDF 编译器以及斯坦福七维评审接口（[paperreview.ai](https://paperreview.ai/tech-overview)）。

详细基准说明、指标公式与新任务扩展教程请参见 **[BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md)**，全量开发与基线验证档案见 **[DevHistory.md](DevHistory.md)**。

> 🔥 **必读深度解构**：既然谷歌捂着一行框架代码都不开源，我们是怎么纯靠 65 页论文正文与四个附录里的蛛丝马迹（包括附录 A.2“谷歌搭台、Claude Code 唱戏”、表 8 临时换自家 `Antigravity CLI` 当备胎掉分、借清华 `AutoSOTA` 题库再拿人家开源代码当靶子、以及表 5 二轮改稿过拟合掉分），把它的六阶段工程图纸一比一反推并复刻出来的？请直接阅读 **[ORIGINAL_PAPER_ANALYSIS.md](ORIGINAL_PAPER_ANALYSIS.md)**！

---

## 一、 三步安装与单元测试（32 项全覆盖）

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[toy]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

- **环境要求**：`Python >= 3.10`、`NumPy`；编译 `manuscript.pdf` 需系统已安装 `pdflatex`（含 `amsmath, amssymb, booktabs, longtable, geometry, hyperref`）；运行端到端科研循环需已登录 `codex` CLI（默认调用 `gpt-6.1-sol` + `low` 推理档，亦支持 `--backend claude` 或 `--backend custom`）。

---

## 二、 核心模块架构

- **[`AGENTS.md`](AGENTS.md)**：自动部署到每个隔离实验工作区的 80 行高维认知与第一性原理协议（第一人称责任锚点、出刀前四步瞬时自检、连续两次未改善触发二阶递归制动、强制减法消融与方法—代码 100% 逐行对齐）。
- **[`open_scientisttwo/pipeline.py`](open_scientisttwo/pipeline.py)**：六阶段闭环状态机——
  1. **Stage 1（局限提取与种子假设生成）**：提取基线瓶颈并经 Verifier 核验，实时调用 `arXiv API` 检索文献并按新颖度排序候选队列；
  2. **Stage 2（子集筛查 → 全量集放大）**：从封存父快照切出独立 Git 分支修改 `model.py`，在 `--mode subset` 与 `--mode full` 上依次执行 `Good / Bad / Engineer` 裁决（最多 2 轮工程微调）；
  3. **Stage 3（基于执行轨迹 `traces` 的点子进化）**：汇总历史成功与失败日志演化新候选，按全量验证集实测指标与 `higher_is_better` 方向选出最优方案；
  4. **Stage 4（组件级减法消融与方法瘦身）**：自动规划并执行 3–5 组减法/控制消融，剥离无增益花架子模块（最多 1 轮瘦身，实测不劣于原方案才替换）；
  5. **Stage 5（六章论文生成与 Rebuttal 终端补实验）**：生成含完整数学公式推导与消融表的论文手稿，当模拟审稿得分 `< 8` 时退回终端跑 `--extra` 扩展验证负载并更新手稿（默认 1 轮，支持 `--max-peer-rounds 2`）；
  6. **Stage 6（总编终审与四道 CoE 完整性审计）**：在留出测试集（`--split test`）上冻结评测一次，并执行 `I1–I4` 四维合规审计与七维审稿打分。
- **[`open_scientisttwo/audits.py`](open_scientisttwo/audits.py)**：实现论文 Table 7 的四道出厂审计——`I1`（独立子进程从不可变快照重算全部表格行数值）、`I2`（`eval.py / data.py / task_spec.json` 哈希锁防篡改）、`I3`（解析手稿参考文献并实时请求 `Crossref / arXiv Atom API` 核对标题与 DOI）、`I4`（对照最终选中源码快照与手稿 Method 公式逐行对账）。
- **[`open_scientisttwo/manuscript.py`](open_scientisttwo/manuscript.py)**：输出完整 6 章英文学术手稿（`Abstract`、`Introduction`、`Related Work`、`Methodology`、`Experiments`、`Limitations` + `References`），原生渲染 `$...$` / `$$...$$` 数学公式与 `booktabs` 三线表，双遍运行 `pdflatex` 编译生成 `manuscript.pdf`。
- **[`submit_paperreview_ai.py`](submit_paperreview_ai.py)**：对接斯坦福 `https://paperreview.ai/` 官方线上三步上传（`/api/get-upload-url` $\rightarrow$ S3 预签名上传 $\rightarrow$ `/api/confirm-upload` 拿 `token`）与 `/api/review/<token>` 轮询查分客户端。

---

## 三、 如何测试与在 DGX Spark 上运行谷歌同款对标实验

仓库内置三个完整可运行的基准任务（无需下载外部大体积数据集，全部由确定性随机种子生成评测负载）：
1. **`benchmarks/inc_bpe`** —— 对标谷歌论文图 2 ＋ 表 9 头号案例：**流式增量 BPE 分词优化**（`Jiang & Gong, ICML 2026 Spotlight`, `arXiv:2605.30813`；谷歌 `ScientistTwo` 第一轮 `VD-STrans` 靶标：相对加速 `+10.9%` / `1.123×`，斯坦福盲审 **`6.5` 分**）；
2. **`benchmarks/pinet_hcnn`** —— 对标谷歌论文附录 B 表 16 第一题：**硬约束神经网络投影层优化**（`Grontas et al., ICLR 2026`, `arXiv:2508.10480`；谷歌 `ScientistTwo` `ANSE` 靶标：等式违背度压至 `$10^{-14}$`，ICLR 2026 组斯坦福盲审均分 **`5.4` 分**）；
3. **`examples/toy_ridge_ood`** —— 8 分钟快速冒烟测试任务（带 12% 离群响应污染与协变量偏移的鲁棒回归基线）。

### 1. 第一步：秒级验证两道对标真题的本地基线（不调用 LLM）

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
(cd benchmarks/inc_bpe && python eval.py --mode subset --split validation)
(cd benchmarks/inc_bpe && python eval.py --mode full --split validation)
(cd benchmarks/pinet_hcnn && python eval.py --mode subset --split validation)
(cd benchmarks/pinet_hcnn && python eval.py --mode full --split validation)
```

- `inc_bpe` 会先对流式输入的**每一个字节前缀** 100% 校验增量分词 token ID 序列与标准贪心 BPE 参考解完全一致，再测量 `latency_ms_per_10kb`（主指标 `mse`，越低越优）与吞吐量 `throughput_mb_s`；
- `pinet_hcnn` 会对照经独立 KKT 条件认证的全局最优参考解，测量相对次优度 `rs_mean`、等式违背度 `eq_cv_max`、总约束违背度 `cv_max`、求解耗时 `solve_time_ms` 及综合损失主指标 `mse`（基线实测回执见 [`benchmarks/results/baselines.json`](benchmarks/results/baselines.json)）。

### 2. 第二步：运行快速端到端闭环（`--fast-demo`，1 轮进化 + 完整 6 阶段审计与 PDF 生成）

```bash
python -u -m open_scientisttwo.pipeline --source benchmarks/inc_bpe --fast-demo --output /tmp/st-bpe-quick
python -u -m open_scientisttwo.pipeline --source benchmarks/pinet_hcnn --fast-demo --output /tmp/st-pinet-quick
```

### 3. 第三步：在家里 DGX Spark 上用 `tmux` 挂机跑完整 4 轮演化 + 5 组减法消融

建议在 `tmux` 会话中逐个任务串行运行（避免多任务同时争抢 CPU 影响计时精度）：

```bash
tmux new -s scientist-run
. .venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

# 1) 完整跑穿流式增量 BPE 对标实验（K=4 轮进化，S=4 个成功解，5 组减法消融）
python -u -m open_scientisttwo.pipeline \
  --source benchmarks/inc_bpe \
  --K 4 --S 4 --ablation-plan-count 5 --eval-timeout 120 \
  --output /tmp/st-bpe-full > /tmp/st-bpe-full.log 2>&1

# 2) 完整跑穿 Pinet 硬约束投影对标实验
python -u -m open_scientisttwo.pipeline \
  --source benchmarks/pinet_hcnn \
  --K 4 --S 4 --ablation-plan-count 5 --eval-timeout 120 \
  --output /tmp/st-pinet-full > /tmp/st-pinet-full.log 2>&1
```

跑完后在输出目录（如 `/tmp/st-bpe-full/`）下直接查看：
- `smoke_receipt.json`：基线得分、最佳方案得分、留出测试集得分与本地七维审稿总分汇总；
- `audit_report.json`：`I1–I4` 四道完整性审计回执；
- `manuscript.pdf` / `manuscript.tex` / `manuscript.md`：自动生成的 6 章完整学术论文 PDF 与源码；
- `stanford_review.json`：本地七维审稿打分单与详细评语。

### 4. 第四步：一键将生成的 `manuscript.pdf` 提交至斯坦福 `paperreview.ai` 官方服务器拿分

```bash
# 先用 --dry-run 本地校验 PDF 大小与表单参数
python submit_paperreview_ai.py /tmp/st-bpe-full/manuscript.pdf --venue ICLR --email lmxxf@hotmail.com --dry-run --output /tmp/upload-dry.json

# 正式提交到斯坦福 paperreview.ai，当场获取官方 Review Token 并写入 upload.json
python submit_paperreview_ai.py /tmp/st-bpe-full/manuscript.pdf --venue ICLR --email lmxxf@hotmail.com --output /tmp/upload.json

# 凭返回的 token 轮询拉取斯坦福官方审稿报告
python submit_paperreview_ai.py --token YOUR_TOKEN --poll 20 --interval 30 --output /tmp/official-review.json
```
