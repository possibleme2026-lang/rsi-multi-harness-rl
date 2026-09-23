# rsi-multi-harness-rl

**面向 agentic RL 的跨 harness 泛化。**

让同一个策略在多种 agent harness 中训练，再测量它有多少能迁移到一个从未见过的 harness 上。

> **当前状态：基础设施完成，主实验尚未运行。**
> 本仓库交付的是一套经过验证的测量装置和一份预注册的协议。跨 harness gap 的数字
> **待补**，原因是实测出来的而非假设的——扫描到的训练单元格中有 52% 不含任何梯度。
> 见[实验结果](#实验结果)。

[English](README.md)

---

## 问题

一个 agentic 策略是在某个 *harness* 内部训练的：一段 system prompt、一组工具、一套
提交协议。通常人们把 harness 当成训练之后的产品决策——先把模型训好，再套上 Claude
Code，或者 Codex，或者自己写的脚手架。

这个视角掩盖了一种失败模式。如果一个策略只在一个 harness 里训练过，它可以靠记住那个
harness 的接口来满足奖励，而不是靠理解任务本身。奖励上升了，能力没有迁移。然后脚手架
一换，策略就得重训。

小米的 MiMo-V2.6 技术报告直接把这一点讲了出来。§4.2.5 采用多 harness 训练，理由是开源
用户会各自搭建自己的 harness、而不会收敛到同一个；发布说明则指出多 harness 训练提升了
*"the model's generalization ability across different frameworks, including unseen ones"*。
报告的效果：把 harness 换成训练中从未见过的那些（Codex、Claude Code、mini-swe-agent），
平均通过率从约 50% 升到 66%。

本仓库是这个实验的一个小型、可完整复现的实例。一块笔记本 GPU、一个 0.5B 模型、一套小到
可以逐条人工检查的任务集——但测量方法做扎实。

## 测量方法

主指标是一个差分之差：

```
gap = mean(reward | train harnesses) − mean(reward | held-out harness)
```

把它当成一个绝对水平来读是没有价值的：基座模型本来就有 gap。所以本仓库在同一个进程、
同一个随机种子、共享同一批 harness 实例的条件下测三条臂：

| 臂 | 训练所用 | 它能告诉你什么 |
| --- | --- | --- |
| baseline | 不训练 | 基座模型本来就有的 gap |
| single | 单个 harness | 过拟合单一脚手架要付出多少代价 |
| multi | 四个 harness | 混合训练是否缩小了 held-out gap |

消融是 `single` 对 `multi`，两者都对照 `baseline` 来读。一个不带 baseline 臂的原始 gap
数字只是一个水平值，不是结论。

被留出的是两条轴，不是一条。只切分 harness 会让模型背下任务却仍显得"泛化了"；只切分任务
则让 harness 这条轴完全没被检验。所以任务集按 16/8 切分，而 `codex_style` 从头到尾不出现在
训练里。

## 设计取舍

真正起作用的是四个决定。每一个的存在都是因为"天真的写法"给出过错误答案。

**verifier 与 harness 无关。** 判分只读 `<workdir>/answer.txt`。判分器无法知道一份答案是哪个
脚手架产出的，所以分数差异不可能来自判分环节。各 harness 的区别在于答案**怎么提交**——
`react_tools` 有 `finish` 工具，`json_strict` 有 `submit`——一个只学会"用 bash 写文件"的模型
不做出适配就无法在这些 harness 上得分。这正是被测的信号。

**harness 之间是结构性差异，不是装饰性差异。** 五个 harness，在工具集和提交协议上不同：

| harness | 工具 | 提交方式 | 角色 |
| --- | --- | --- | --- |
| `bash_minimal` | `bash` | 写文件 | 训练 |
| `react_tools` | `bash`、`read_file`、`write_file`、`finish` | `finish()` 工具 | 训练 |
| `json_strict` | `bash`、`submit` | `submit()` 工具 | 训练 |
| `longctx_summary` | `bash`、`read_file`、`replace_in_file` | 写文件 | 训练 |
| `codex_style` | `bash`、`apply_patch` | 写文件 | **留出** |

**死单元格要测出来，不能拿来训练。** 当一个组内所有 rollout 得分相同时，二值奖励没有梯度。
在 `G` 次生成、通过概率 `p` 的条件下，一个组不含信号的概率是 `p^G + (1−p)^G`——所以
`p ≤ 0.05` 或 `p ≥ 0.95` 的单元格是可证明的浪费。pipeline 用一次难度扫描测出每个单元格，
并在训练前丢掉死掉的那些。这是前置条件，不是优化项。

**能力探测把住 GPU 花费这道关。** 训练之前，四道门先确认基座模型能否驱动这套 harness。
如果不能，之后测出的任何 gap 都是噪声，诚实的输出是"没有结论"而不是一个数字。

| 门 | 阈值 | 实测 | 判定 |
| --- | --- | --- | --- |
| G1 tool-call 率 | ≥ 50% | **76.6%** | 通过 |
| G2 可达单元格（pass@8） | ≥ 1 | **64 个中 32 个** | 通过 |
| G3 跨 harness 差值 | > 0 | **0.75** | 通过 |
| G4 多轮采纳率 | ≥ 50% | 392 条调用工具的 rollout 中 **100%** | 通过 |

**Go/No-Go：GO。** Qwen2.5-0.5B-Instruct 能驱动全部五个 harness。平均轮数 1.77。

## 实验结果

**主消融实验尚未运行。** 已测的是：n=8，4 个训练 harness × 16 个训练任务。

通过率矩阵（行 = harness，列 = 任务）：

| 任务 | `bash_minimal` | `react_tools` | `json_strict` | `longctx_summary` |
| --- | --- | --- | --- | --- |
| t1-01 | 0.875 | 0.375 | 0.750 | 0.125 |
| t1-02 | 0.500 | 0.375 | 0.125 | 0.000 |
| t1-03 | 0.375 | 0.625 | 0.250 | 0.250 |
| t1-04 | 0.125 | 0.250 | 0.250 | 0.125 |
| t1-05 | 0.750 | 0.625 | 0.250 | 0.250 |
| t1-06 | 0.500 | 0.375 | 1.000 | 0.250 |
| t1-07 | 0.875 | 0.375 | 0.500 | 0.125 |
| t1-08 | 0.250 | 0.000 | 0.625 | 0.000 |
| t2-01 … t2-04 | 0.000 | 0.000 | 0.000 | 0.000 |
| t3-01 | 0.000 | 0.250 | 0.000 | 0.250 |
| t3-02 | 0.000 | 0.000 | 0.000 | 0.000 |
| t3-03 | 0.000 | 0.000 | 0.000 | 0.000 |
| t3-04 | 0.000 | 0.125 | 0.000 | 0.000 |

**64 个单元格里有 33 个（52%）是死的**——通过率 ≤ 0.05 或 ≥ 0.95，因此没有梯度。分布本身
就是结论，而且比单个数字所显示的更糟：

| 层 | 活单元格 | 死单元格 |
| --- | --- | --- |
| T1 抄写 | 28 | 4 |
| T2 读取后写出 | **0** | **16** |
| T3 引号压力 | 3 | 13 |

T2 在**全部 16 个单元格**上都是死的：0.5B 模型无法足够可靠地完成"两步读取再抽取"的流水线，
以至于从未通过，所以没有任何 T2 单元格能产生梯度。T3 在 16 个中死了 13 个——这一层本是刻意
设计成承重层（harness 在转义上真正分化之处），在这个模型规模下几乎完全无法测量。

**结论，直说。** 在现有任务集上跑完整消融，得到的主指标会被结构性零值主导：eval 划分中的
T2 与 T3 单元格会在每条臂上都是全零，而所谓"gap"大部分会是任务所属层级造成的假象。必须先让
T2/T3 具备难度梯度，跨 harness gap 才有意义。这是下一步工作，也是本次发布被标注为"基础设施"
而非"结果"的原因。

## 复现

需要 Python 3.12+，Windows 上需要 Git-Bash。核心部分——harnesses、任务、verifier、全部静态
守卫——**不依赖第三方库**，所以下面的检查不需要装 torch。

```bash
git clone https://github.com/possibleme2026-lang/rsi-multi-harness-rl.git
cd rsi-multi-harness-rl

# core checks: no torch, no GPU
./run.sh scripts/guard_tool_surface.py
./run.sh tests/test_path_errors.py
./run.sh tests/test_shell_timeout.py
./run.sh tests/test_scan_tooling.py
./run.sh tests/smoke_env.py
```

`run.sh` 是受支持的入口，不是便利脚本。它会设置 `APPDATA`、HuggingFace 缓存目录并清掉
`PYTHONPATH`，因为在 Windows 上这三者配错时给出的都是**误导性**报错而不是缺依赖报错——
`from_pretrained` 会说没有网络连接，或者在明明装了 torch 的机器上 `import torch` 抛
ModuleNotFoundError。

完整 pipeline 需要 RL 依赖栈（`trl` 走 dev 线；见 `pyproject.toml`）和一块 GPU：

```bash
pip install -e ".[rl,dev]"

bash pipeline.sh                    # scan -> train single -> train multi -> eval
STEPS=20 N_EVAL=4 bash pipeline.sh  # a smaller run
SKIP_SCAN=1 bash pipeline.sh        # reuse an existing difficulty scan
```

产物都在 `outputs/`（可用 `MULTIHARNESS_OUT` 覆盖）。在 RTX 5060 Laptop 上扫描约 7 分钟
（每个单元格 0.8 秒）；训练和评估更长。

| 脚本 | 作用 |
| --- | --- |
| `scripts/probe.py` | 能力探测 + 难度扫描，写出 `scan_all.json` |
| `scripts/train.py` | 消融的一条臂（`--mode single` / `--mode multi`） |
| `scripts/eval.py` | baseline + 两条臂在同一进程内跑，打印消融表 |
| `scripts/diag.py` | 单个 (harness, task) 的完整未截断轨迹 |
| `scripts/errs_report.py` | 归类扫描中的工具错误：harness 缺陷还是模型行为 |
| `scripts/guard_tool_surface.py` | 宣称的工具必须等于实现的工具 |
| `scripts/merge_scan.py` | 把局部重扫合并进旧 dump，带不变量检查 |

## 目录结构

```
src/multiharness/
  harnesses/core.py     BaseHarnessEnv、verifier、shell runner、路径解析器
  harnesses/pool.py     五个 harness
  tasks/suite.py        24 个任务，以及 train/eval 切分
  rollout.py            独立重实现的 TRL 工具调用循环
  _bootstrap.py         仓库根目录 + 产物目录
scripts/                入口脚本（probe、train、eval、守卫）
tests/                  smoke 测试与回归测试
tools/                  README 双语一致性守卫
pipeline.sh            完整运行流程，按依赖顺序
```

## 两个值得记录的 bug

两个都是"实测与预期不符"发现的，都有回归测试。写进 README 是因为它们各自会静默污染结果，
而不是崩溃。

**一个值 84 分钟空转 GPU 的 subprocess 死锁。** shell runner 用了
`subprocess.run(capture_output=True, timeout=...)`。超时触发时它只杀掉直接子进程，然后去
排空管道——但继承了写端的孙进程让管道一直开着，排空永远等不到 EOF，于是永久阻塞。
`TimeoutExpired` 甚至不会被抛出，因为超时已经触发过了；**安全网本身挂死了**。症状是"84 分钟
没有输出"且 GPU 占用 0%，而只修超时是没用的。现在 runner 改为写入临时文件、关闭 stdin、
超时时杀掉整个进程树。

**`Path(workdir) / ""` 是目录本身，不是错误。** 模型调用
`write_file(path="", content=...)` 或 `replace_in_file(path=".")` 时，实际是对工作目录本身
发起写入，而原始 OS 报错被当作工具结果回给了模型：Windows 上是 `Permission denied`，POSIX
上是 `IsADirectoryError`。报错随平台而变才是真正的问题——它污染了实验所测的那条轴，会让某个
harness 因为与模型毫无关系的原因而得分更低。同一次排查还发现 POSIX 绝对路径
（`/config.txt`、`/Users/qwen/...`）在 Windows 上会静默改指向工作区之外。现在所有接受路径的
工具共用一个解析器，以与操作系统无关的报错拒绝空路径、`.`、目录目标和越界路径。

第三个相关的修复：`errs_report.py` 最初会给出**假绿**结论——"没有 harness 缺陷"——而实际上
有 25–27 条原始 OS 报错躺在它的兜底分类里。现在它有明确的 OS 报错类别，并会让该次扫描判为
无效。一个不可能失败的守卫比没有守卫更糟。

## 引用

如果本仓库对你有用，请引用它所基于的工作：

```bibtex
@misc{mimo2026v26pro,
  title={MiMo-V2.6-Pro-RL},
  author={{Xiaomi MiMo Team}},
  year={2026},
  howpublished={\url{https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL}},
}
```

## 许可

Apache-2.0。见 [LICENSE](LICENSE)。
