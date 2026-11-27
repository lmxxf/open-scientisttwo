# 对标基准、测量与扩展完整指南

## 1. 哪些是本地事实，哪些是论文靶标

本仓两道任务对应 ScientistTwo 图2/表9 的 Incremental BPE 和表16的 Pinet。
它们是**同任务数学机制、独立可运行实现**，不是作者官方代码或原数据全量复刻。
BPE 使用自训练的72条合并规则与中英代码流；Pinet 使用确定性参数化 QP，
是投影/求解层内核，不含神经网络训练或反向传播。不能据此宣称超过原论文。

来源：
- ScientistTwo: https://arxiv.org/abs/2609.19644
- Incremental BPE: https://arxiv.org/abs/2605.30813
- Pinet: https://arxiv.org/abs/2508.10480

任务提供的生成论文称 VD-STrans 使用 AVX2、12套生产词表，几何平均加速1.123x；
ScientistTwo图2的10.9%与表9后续9.6%/8.2%是不同轮次/分母，不可互换。
官方案例斯坦福6.5分也不能当成本仓评分。本地 ARM64/Python 成绩不等于
x86 AVX2 内核。ANSE生成论文称训练快2.9–3.2倍、约束残差1e-13–1e-14；
本仓只测 forward CPU projection，无法验证其训练/梯度成绩。AutoSOTA是论文
比较对象，未核实其特定表格行的数字不抄作本地验收线。

## 2. 安装与立即验证（DGX Spark / Linux CPU）

从克隆仓库根目录执行，Python>=3.10，NumPy唯一算法依赖；PDF需要pdflatex及
amsmath/amssymb/booktabs/geometry/hyperref/longtable等标准TeX包。

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[toy]'
```

设置线程以减少小矩阵线程调度噪声；Spark统一内存不要求GPU/CUDA。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m unittest discover -s tests -v
(cd benchmarks/inc_bpe && python eval.py --mode subset --split validation)
(cd benchmarks/inc_bpe && python eval.py --mode full --split validation)
(cd benchmarks/pinet_hcnn && python eval.py --mode subset --split validation)
(cd benchmarks/pinet_hcnn && python eval.py --mode full --split validation)
```

两者均支持 `--split test --extra` 和 `--config config.json`；单独运行eval时，
必须在对应目录（或显式提供config路径）执行。时间单位和硬约束不可更改。
本次已测结果与源码SHA在 `benchmarks/results/baselines.json`，是当前机器的
一次基线回执，不是跨平台成绩承诺。

## 3. 增量 BPE：测什么与怎么改

给定原子字节ID0..255、有序规则 `(left,right)->256+rank`。按最低rank、
最左位置反复合并，直至无可合并对。独立参考解使用优先队列和链表。
基线使用Aho-Corasick识别后缀、SufSucTree successor指针检查可达性和优先级，
历史theta数组恢复任意前缀的完整token序列。DFS区间满足：
`node belongs to subtree(root) iff L[root]<=L[node]<R[root]`。

基线循环里是指针追逐与候选排序，可改为紧凑数组、区间判断、缓存转移和
单字节快速路；不能变成长词优先、改变tie-break、跳过前缀检查。eval先对
**每一个字节前缀**核对token ID及字节恢复，任何不一致直接异常；精确性无容差。
然后单独计时append（不含词表/自动机构造和tokens()结果物化），7次中位数。
`mse=latency_ms_per_10kb=seconds/bytes*1e7`，越低越好；输出另含MB/s与us/kB。
词表构建成本若被优化必须另报，不能把计时边界混着比较。

## 4. Pinet：硬等式、盒约束与参考解

`J=0.5*y.T@Q@y+q.T@y+alpha*sum(sin(y))`；约束`M@y=b, lower<=y<=upper`。
上下文改变Q/q/M/b；M行尺度跨度100倍。subset维数12/24，full24/48/64，
每个维数含alpha=0/.8。下上界各有活跃分量。参考解通过独立KKT条件生成并核验：
`grad J + M.T@lambda + normal=0`，normal符号与活跃上下界一致。
`lambda_min(Q)-alpha*sin(.25)>0`证明在盒内严格凸，因而该参考解为全局最优。
正弦目标在盒外可非凸；此设计不支持“解决任意非凸全局优化”的结论。

提升空间z=(y,s)，A=[[M,0],[I,-I]]；在s上clip，在y上作目标梯度步，
经lifted affine投影更新forward Douglas-Rachford / Davis-Yin迭代。
baseline每次重解AA.T，40步、step=.08。可研究零空间重参数化、行均衡、
缓存分解、加速/可行性恢复。不能把未收敛点的低目标当成优于可行最优解。

- `rs_mean=mean(max(0,(J-J_ref)/max(1,abs(J_ref))))`；另留signed gap。
- `eq_cv_max=max ||M@y-b||inf`。
- `cv_max`涵盖等式及盒违约。
- `solve_time_ms`：每上下文3次中位数，再均值。
- `mse=rs_mean+50*cv_max+0.05*solve_time_ms`；cv>1e-2再加1000*cv。
- 非有限/错shape/参考KKT失败抛异常；不可行结果显式大罚，不静默修复。

参考向量和乘子不传给model.solve。不要让Coding Agent读取或拟合参考解。

## 5. 周末真实LLM循环（本次升级没有运行）

先检查 `codex --version`、登录、`pdflatex --version`，从仓库根运行。
每个output必须不存在，禁止复用上次产物。三个快速档：

```bash
python -m open_scientisttwo.pipeline --source examples/toy_ridge_ood --fast-demo --output /tmp/st-toy-weekend
python -m open_scientisttwo.pipeline --source benchmarks/inc_bpe --fast-demo --output /tmp/st-bpe-quick
python -m open_scientisttwo.pipeline --source benchmarks/pinet_hcnn --fast-demo --output /tmp/st-pinet-quick
```

toy历史约8分钟，只是测量记录，不能保证5分钟。fast仍会跑真实Codex角色。
完整档（另开终端进入tmux后运行，不要并发争CPU测时）：

```bash
tmux new -s scientist-bpe
```

进入tmux后：

```bash
. .venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u -m open_scientisttwo.pipeline --source benchmarks/inc_bpe --K 4 --S 4 --ablation-plan-count 5 --eval-timeout 120 --output /tmp/st-bpe-full > /tmp/st-bpe-full.log 2>&1
```

另一道题单独运行：

```bash
python -u -m open_scientisttwo.pipeline --source benchmarks/pinet_hcnn --K 4 --S 4 --ablation-plan-count 5 --eval-timeout 120 --output /tmp/st-pinet-full > /tmp/st-pinet-full.log 2>&1
```

`Ctrl-b d`脱离，`tmux attach -t scientist-bpe`返回，`tail -f /tmp/st-bpe-full.log`
看控制台。每个角色回执/时长在隔离工作区trace；最终或失败时会复制到output。
CLI进程超时杀进程组；增加`--role-timeout`与`--eval-timeout`是不同预算。
默认1轮peer review，最多2轮，避免第二轮过拟合反馈；零成功明确失败终止。

## 6. 论文、审计与官方投稿

Writer JSON：title/abstract/introduction/related_work/methodology/experiments/limitations。
Markdown包含摘要及5章正文，合计6个学术部分，另列Reference；精确主指标表
由引擎生成，Agent不能填写数字。LaTeX保留`$...$`/`$$...$$`公式，使用标准数学
包与booktabs表，pdflatex跑两遍。公式坏掉时显式记录native失败日志，退回安全
转义版再编译；`pdf_build.json`注明fallback，不能当作原生数学版成功。
篇幅由研究内容决定，不靠塞空白页凑“多页”；测试用足量六部分文本验证多页排版。

- I1：独立子进程重算claims并核对Markdown每行值。默认确定性指标1e-9绝对误差；
  两道计时基准显式50%相对容差，避免宣称时间能逐位复现。应检查细分指标并做
  多次同机配对计时，50%容差不是速度提升的统计显著性证据。
- I2：eval.py/data.py/task_spec.json与每个封存快照内容哈希不变。
- I3：最终参考表每条标题/DOI在线核验；Crossref DOI走Crossref，arXiv DOI走Atom API。
- I4：审稿Agent读取选中源代码和配置与公式对账，不是形式化正确性证明。
- `stanford_review.json`是本地七维未校准均分代理，非官网结果；Reject照实保存。

官方投稿（真正执行会上传PDF，本次未执行）：

```bash
python submit_paperreview_ai.py /tmp/st-bpe-full/manuscript.pdf --venue ICLR --email YOUR_EMAIL --dry-run --output /tmp/st-upload-dry.json
python submit_paperreview_ai.py /tmp/st-bpe-full/manuscript.pdf --venue ICLR --email YOUR_EMAIL --output /tmp/st-upload.json
```

从回执读取token，再查询：

```bash
python submit_paperreview_ai.py --token YOUR_TOKEN --poll 20 --interval 30 --output /tmp/st-official-review.json
```

三步：POST `/api/get-upload-url` JSON filename/venue；S3 multipart签名字段先、PDF最后；
POST `/api/confirm-upload` multipart s3_key/venue/email得到token。GET `/api/review/TOKEN`
202表示处理中，429停止不重投；继续查token即可。回执在轮询前落盘，避免丢token。
官方仅ICLR目标展示分数，是否成功/具体字段以返回JSON为准，不伪造换算官方总分。

## 7. 新任务接入与维护

复制一个基准目录：必需model.py/data.py/eval.py/baseline_paper.md，建议config.json。
评测CLI统一`--mode subset|full --split validation|test --extra --config`；只输出一个
JSON对象，主指标必须有限数值；任务所需额外依赖先装好。

```json
{
  "title": "A research question",
  "primary_metric": "speedup",
  "higher_is_better": true,
  "arxiv_query": "all:\"algorithm\"",
  "seed_dois": [],
  "coding_guidance": "Edit only model.py; preserve solve(problem,config). Never alter protected files.",
  "eval_timeout": 120,
  "metric_atol": 1e-9,
  "metric_rtol": 0
}
```

无task_spec默认中性研究任务、mse最小化、无凭空引用。领域知识只放task_spec与
baseline_paper，不能塞回pipeline/audits。任意基准仍须符合上述文件/CLI契约；
复杂多文件外部仓需先提供适配器，不能假设未经适配的仓库可直接运行。
每次改算法跑测试、subset/full基线，留源码hash、机器、线程、命令、原始JSON。
发布结论须分开：本地可运行、对标同任务、复现作者全负载、超越其成绩，证据逐级要求。
