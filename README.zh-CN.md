# rsi-multi-harness-rl

**面向 agentic RL 的递归自我改进：任务、奖励、harness 三者都由系统自己构建，再由会失败的闸门逐一把关。**

策略是在某个 *agent harness* 内部训练的。本仓库要问的是：当 harness 不再是一个训练之后才
拍板的产品决策——当系统**自己生成任务、自己写奖励函数、自己演化 harness**，并且只保留那些
增益超过实测噪声底线的改动时，会发生什么。

这里有三个轴是自构建的，每一个都是被验证出来的，而不是被断言的：

| 轴 | 由谁构建 | 由谁把关 | 证据 |
| --- | --- | --- | --- |
| **任务** | `rsi/task_gen.py` | 闸门 V1–V4 | `outputs/rsi/validation.json` |
| **奖励** | `rsi/verifier_gen.py` | V1 永不触发 / V2 总是触发 | 同一产物，按闸门分列 |
| **harness** | `rsi/harness_evolve.py` | 噪声底线 + 工具面守卫 | `outputs/rsi/ledger.jsonl` |

[English](README.md)

---

## 为什么这才是难点

一个 agentic 策略是在某个 *harness* 内部训练的：一段 system prompt、一组工具、一套提交协议。
通常人们把 harness 当成训练之后的产品决策——先把模型训好，再套上 Claude Code，或者 Codex，
或者自己写的脚手架。

这个视角掩盖了一种失败模式。如果一个策略只在一个 harness 里训练过，它可以靠记住那个
harness 的接口来满足奖励，而不是靠理解任务本身。奖励上升了，能力没有迁移。然后脚手架一换，
策略就得重训。

小米的 MiMo-V2.6 技术报告直接把这一点讲了出来。§4.2.5 采用多 harness 训练，理由是开源用户会
各自搭建自己的 harness、而不会收敛到同一个；发布说明则指出多 harness 训练提升了
*"the model's generalization ability across different frameworks, including unseen ones"*。
报告的效果：把 harness 换成训练中从未见过的那些（Codex、Claude Code、mini-swe-agent），
平均通过率从约 50% 升到 66%。

要在一块笔记本 GPU 上复现它，得先解决一个前置问题：**固定的任务集和手写的奖励回答不了关于
泛化性的问题**，因为任务集恰好覆盖到什么，就会被测成什么。所以本仓库自己造课程、自己造奖励、
自己造脚手架——然后想尽办法去证伪它们。

## 轴一 —— 任务是生成的，不是维护出来的

`rsi/task_gen.py` 从一个显式参数空间采样，产出一次完整的试验：环境、任务、奖励，以及一份
参考解法。

| 参数 | 取值 | 它控制什么 |
| --- | --- | --- |
| `tier` | T1 / T2 / T3 / T4 | 难度族 |
| `payload_len` | 1、2、4、8、16 | 有多少文本必须完整走完一个来回 |
| `escape_density` | 0.0、0.15、0.35、0.6 | 需要多少 shell 转义 |
| `steps` | 1、2、3 | 写出去之前要经过几步变换 |
| `read_source` | false、true | 内容在 prompt 里，还是在磁盘上 |
| `verify_mode` | `file_equals` / `file_contains` / `python_exit` | 结果怎么判分 |

这里每个函数都是"参数 + 种子"的纯函数，所以同一个种子会产出逐字节一致的任务。这正是 CI 能
重新推导整套任务、并与真正拿来训练的那个产物逐条比对的前提。

**手写的"简单层"试过，被否掉了。** 本工作最早的设计是加一层手写的 T2-lite 简单任务，去填补
扫描发现为空的那些单元格。手写一层等于让人来维护课程，而这恰恰是 RSI 要取代的东西。生成器用
参数就能产出同样的覆盖度，并且可以按需产出更多。

## 轴二 —— 奖励是生成的，而且它会以两种方式出错

`rsi/verifier_gen.py` 把一个期望答案变成一个奖励。生成出来的奖励只有两种失败方式，两种都是
闸门而不是评审意见：

* **它永不触发**——参考解法拿不到 1.0，于是任务无解，每一条 rollout 都被浪费（闸门 **V1**）；
* **它总是触发**——空目录也能拿到 1.0，于是任务什么都没测，策略学会了"什么都不交"（闸门 **V2**）。

第三种情况更微妙，它靠构造而不是靠闸门解决。`check_script` 是一个**文件名**，不是源码：
`core.verify` 会以工作目录为 cwd 执行 `python <check_script>`，而把文件放进那个目录的唯一
机制是 `setup`。这就把检查脚本放进了 agent 伸手可及的范围内，所以生成出来的检查脚本存的是
**SHA-256**，而不是明文答案。这又意味着**子串奖励不能做成摘要**——判断包含关系需要那个 needle
本身——所以它落在 `file_contains` 模式里，比较发生在 harness 进程内，磁盘上什么都不写。
`check_script_source` 遇到这种情况会直接抛错，而不是产出一个根本不可能工作的检查脚本：

| 强度 | 模式 | 原因 |
| --- | --- | --- |
| `exact` | `file_equals` | 比较在进程内完成，沙箱里什么都不进 |
| `normalised` | `python_exit` | 需要断言一个性质，而不是一个字符串 |
| `substring` | `file_contains` | **摘要无法判断包含关系** |

而且，短答案用 `python_exit` 判分会把可暴力破解的摘要送出去，所以生成器会去咨询
`recommend_mode`，而不是相信自己手里的参数向量。在一条 300 任务的批次上，这条规则接上之前
有 96 条（32%）属于这种情况；现在是 0 条，同时长答案的 `python_exit` 依然可达。那个 96 是
seed 0 的结果——比例随 seed 波动但很集中，八个 seed 下是 79–99 条，而 `--seed` 默认是 11，
给出的是 91 条。写上 seed 是因为不带 seed 的比例不可复现。

## 轴三 —— harness 被演化，对照一条实测的噪声底线

`rsi/harness_evolve.py` 在一个由八个具名描述符改动构成的编辑空间里搜索，每一条都是一个
*关于"策略为什么失败"的假设*：

| 改动 | 组件 | 假设 |
| --- | --- | --- |
| `guidance+=submit_echo` | prompt | 写出了文件，但从不核对内容 |
| `guidance+=retry_hint` | prompt | 一次调用失败后就停下 |
| `guidance+=exactness` | prompt | 在需要精确匹配的答案后面加了话 |
| `guidance+=one_line` | prompt | 要的是一个值，它写了一个句子 |
| `prompt-=verbose_preamble` | prompt | 要求推理的引导把轮次预算花在了散文上 |
| `output_plumbing+=explicit_path` | output_plumbing | 写到了错误路径或嵌套目录里 |
| `client_tool-=read_file` | client_tool | 用不上的工具占上下文，还招来乱调用 |
| `context_mgmt+=keep_last_error` | context_mgmt | 重复一次已经失败过的调用 |

四条设计约定让它成为一次搜索，而不是一次随机游走：

**候选必须比在位者高出噪声底线以上。** 底线是 `z · √2 · se`，`z = 2`，跨单元格池化，改写自
RRSI 的 `calibrate.py`。没有它，演化循环会永远在采样误差上爬坡。账本把
`rejected_within_noise` 与 `rejected_worse` 分开记——这个区分很重要，因为"我们分辨不出来"和
"它更差"是两个不同的发现。

**编辑预算做退火。** `b_t = ceil(b_min + (b_max − b_min)·½(1 + cos(πt/T)))` 约束的是
`‖z_t‖₀`，也就是一次提案里独立改动的条数。它不是步长，也不是分数阈值。当
`T=12, b_min=1, b_max=3` 时，这条调度是 `[3,3,3,3,3,3,2,2,2,2,2,2]`。

**新颖度只数结构性组件。** 一个 harness 的引导文字被改写五次，不等于它探索了五个区域；把这算
成新颖度，只会推着循环继续改写散文，而不是去改接口。这里的"结构性"指 `client_tool` 和
`output_plumbing`——改变 agent **能做什么**的组件，而不是改变它**被告知什么**的组件。

**每一次尝试都记录，剪枝只看零产出。** RRSI 每条被接受的改动记一条，其余靠轨迹反推。这里的
编辑空间很小，所以被拒记录承载的信息比例更高："这一整片邻域已经穷尽"只有在失败也进了账本时
才看得见。一个组件只有在被尝试至少 `min_attempts` 次**且从未被接受过一次**时才会被剪掉——
用产出率阈值会误杀偶尔有效的组件，而这恰恰是统计层存在的意义所在。

产物是一份 JSONL 账本，它是对每个假设的检验而不是一个分数：在回放评分下，尝试 30 次改动、
接受 10 次，并且 `context_mgmt` 因为历次尝试零产出而被剪掉。（下面那次实测剪掉的是另一组组件、
保留下来的也是另一个组件，这正是要跑实测的意义——见对比表。）

**这些数字来自 `--score ledger-replay`，产物本身也这么标注。** `scripts/rsi_loop.py` 有两种评分
模式，区别记录在 `outputs/rsi/harness.json` 的 `score_mode` 字段里：

| 模式 | 谁给候选打分 | 这个数字意味着什么 |
| --- | --- | --- |
| `ledger-replay` | 一个确定性的替身函数 | 在**不需要 GPU** 的前提下走遍预算、守卫、噪声地板、账本与剪枝规则 |
| `rollout` | 模型，在任务批次上 rollout | 诚实的 harness 数字 |

replay 模式的存在，是为了让循环的机械结构能在 CI 跑的那台机器上被运行和测试——那台机器没有
GPU。它**不是**模型测量，把轨迹 `[0.471, 0.528, 0.528, …]` 读成"harness 变好了"是错的：那是一条
合成曲线，唯一职责是走到循环的每个分支。replay 产物里的噪声地板被标为 `fixed fallback` 出于同样的
理由——每个状态只有一个分数，支撑不起 bootstrap 地板。

**`rollout` 那条臂现在跑过了，而它的第一批输出就是对上面那张表的更正。** 四个可训练 harness，
由**未经修改的**基座模型打分，同一批 30 个生成任务 × 8 次 rollout：

| harness | 实测基线 | 回放替身 |
| --- | --- | --- |
| `bash_minimal` | **0.008** | 0.25 |
| `react_tools` | **0.054** | 0.25 |
| `json_strict` | **0.029** | 0.25 |
| `longctx_summary` | **0.029** | 0.25 |

每一个真实分数都比替身假设的 `0.25` **低 3–30 倍**，而且替身给四个 harness 的是**同一个**数字，
真实的四个之间却差了近 7 倍。所以那条回放轨迹不只是"对 harness 有没有变好没有信息量"——它建立在
一个错了一个数量级的基率上，这正是那次运行里被接受的编辑看起来像是把某个东西从 0.471"改进"到了
0.528 的原因。那些数字描述的是替身函数，不是 harness。

**噪声地板也变了，而且方向相反。** 从实测 rollout 上 bootstrap 出来是 **0.0176**，而回放模式退回的
是 `0.05`——差 2.8 倍。所以那次合成运行同时在**基率高约 8 倍**、且**接受一条编辑前要求的余量大约
3 倍**这两点上都是错的。这两个数字都不是测出来的，却决定了那次运行里的每一次接受，而且它们的偏
向相反——这正是回放结果看起来"合理"而不像明显坏掉的原因。

实测基线同时也是消融的诚实语境：约 0.03 就是一个未经训练的 0.5B 模型在这套 harness 池、这批生成
任务上得到的分数，它与随仓库发布的套件在 train harness 上给出的 0.1406 是一致的——生成任务更难，
这正是闸门 V4 想要的。

**而实测那次并没有复现回放那次——它把结论翻了过来。** 同样 30 个任务 × 8 次 rollout，
`--score rollout` 尝试了 **24** 次改动、接受 **1** 次（4%），而回放那次是尝试 30 次、接受 10 次：

| | 回放（`outputs/`） | 实测（`outputs_rollout/`） |
| --- | --- | --- |
| 尝试改动 | 30 | 24 |
| 接受改动 | 10 | 1 |
| 接受率 | 33% | **4%** |
| 噪声地板 | `0.05`（固定退路） | **`0.0176`**（bootstrap） |
| 因零产出被剪 | `context_mgmt` | `prompt`、`output_plumbing` |
| 轨迹 | `[0.471, 0.528, 0.528, …]` | `[0.0917, 0.0917, 0.0917]` |

唯一被接受的那条是 `react_tools` 上的 `context_mgmt+=keep_last_error`，分数
`0.0542 → 0.0917`（`d=+0.0375`）——**恰好就是回放那次因零产出而剪掉的组件。** 它的分组件产出率是
`context_mgmt 0.25`、其余全为 `0.0`：在真实 rollout 上，替身函数丢掉的那个组件是唯一越过地板的。
23 次拒绝里有 16 次是 `rejected_worse`、7 次是 `rejected_within_noise`——这个区分回放模式做不出来，
因为它的地板是个常数而不是一次测量。

有两条限定必须和这张表放在一起，而不是塞进脚注。轨迹是**平的**、不是上升的：只有一次被接受的编辑，
之后没有在它之上继续，所以这证明搜索在真实分数上*能跑*，不证明它*有用*。另外实测那次是 3 轮、
回放那次是 6 轮，所以尝试次数并非同口径——可比的是接受*率*，它掉了大约 8 倍。

### held-out 项是一个地板，于是 gap 是一条恒等式

消融的头条数字是 `gap = mean(train harnesses) − mean(held-out harness)`。三个臂池化之后，
唯一的 held-out harness `codex_style` 是 **96 次里 0 次通过**。它的 95% 区间是 `[0, 0.0385]`，
落在 `0.05` 信号地板之下：这是 `DEAD`，不只是「测得不够」，而且它是一个**正面发现**而不是
「再多测测」的请求。held-out 项不是「很小」，是**整体落在策略能学到的带外**。

后果是 `gap ≡ mean(train)` 是一条**恒等式**而非一次测量，训练多少步都不会改变这一点。按 gap 给
各臂排名，就是换个名字按 train 均值排名——而且是**反着排**的，因为提升最少的那个臂 gap 最小。
`eval.py` 早先的版本正是据此打出「multi 那条假设不成立」，而同一次运行里对同一个臂的逐臂判词是
「overfitting the train harnesses」。两者不可能同时是头条结论；现在当 held-out 均值恰为 0 时，
该比较会被拒绝执行，改为打印这条恒等式。

能报的是**单边**结论，而它值得写出来，因为它并非一无所有：既然 held-out 项不可能超过 `0.0385`，
gap 就不可能低于 `mean(train) − 0.0385`。

| 臂 | mean(train) | gap ≥ |
| --- | --- | --- |
| `train-single-s64` | 0.2656 | **0.2271** |
| `train-multi-s64` | 0.3047 | **0.2662** |

所以 gap 是真实且很大的。它同时也**完全等于** train 项，这正是它对泛化没有信息量的原因——
那些下界的排名就是 `mean(train)` 的排名。

**在这个 0 能被读成能力结论之前，有两件事必须先排除。** `tests/smoke_env.py` 早就断言 oracle 在
全部 24 个任务上拿到 1.0——但走的是 `OracleHarness`，它直接写答案、**从不碰 harness 声明的工具**。
那证明的是*判分器*对，不是「拿着该 harness 工具面的 agent 能解这个任务」，而这是两个不同的断言。
`scripts/harness_solvability.py` 把后者补上：让参考解**穿过每个 harness 自己的工具**驱动，
**120/120**（5 个 harness × 24 个任务），`codex_style` 经 `apply_patch` 24/24 通过。
harness 是有能力的。而 transcript（`scripts/diag.py --harness codex_style`）显示模型为什么仍然得 0：
它**一次工具调用都没有发**，预算全花在规划性散文上——这恰恰是这个 harness 自己的
"Plan first, then act" guidance 招来的——或者花在写在 markdown 围栏里的 JSON 调用上，而解析器不认。

这个地板是策略的，不是 harness 的。把这个 gap 当作已测的泛化数字来报，等于在报 `0 == 0`。

## 四道闸门，以及为什么是四道

一个生成出来的任务，只有过了全部四道才被信任。它们被施加在**每一个**任务、每一次运行上，
而不是靠约定来断言。

| 闸门 | 问题 | 它抓什么 |
| --- | --- | --- |
| **V1** oracle | 参考解法能拿到 1.0 吗？ | 任务规格错误 |
| **V2** nop | 空目录能拿到 0.0 吗？ | 总是触发的奖励 |
| **V3** safety | `check_script` 真的随 `setup` 发出去了吗？ | 判分时根本不存在的检查脚本 |
| **V4** cross-harness | 在**每一个**可训练 harness 上都可解，且它们之间**不完全相同**吗？ | 一个在测 harness 完备性的任务，或一个不含 harness 轴信息的任务 |

**V4 是两个参考系统都不需要的闸门**，而它正是本仓库存在的理由。RRSI 是在一个*固定*任务集上
演化 harness；SPADE 是在一个*固定* agent 上生成环境。同时变动两者，就同时逼出两个条件：一个
任务必须在全部四个可训练 harness 上可解，否则某个单元格上的低分测的是那个 harness 的完备性、
而不是策略的能力；同时它**不能**在所有 harness 上得分完全相同，否则它不含被测那条轴的信息。

在一条 30 任务的生成批次上实测：**接受 30，拒绝 0**，其中 T1 13、T2 6、T3 3、T4 8——19 个任务
带源文件，模式分布为 `file_equals` 17 / `file_contains` 12 / `python_exit` 1。

该批次的参数向量**请求**的是 `file_equals` 10 / `file_contains` 12 / `python_exit` 8。两个分布
相差 **7 次覆盖**，而这个落差正是下文那个"可暴力破解的摘要"缺陷的修复：每一个被降级的任务，
都是在短到无法安全摘要的答案上请求了 `python_exit`，而生成器现在会去问 `recommend_mode`，
不再相信自己抽出来的参数向量。产物把两个计数并排记录下来，正是为了让这个落差始终可见——
只给一张"实际使用的模式"表，会把产生它的那次修正一起藏掉。

## 测量上的设计约定

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

**被留出的是两条轴，不是一条。** 只切分 harness 会让模型背下任务却仍显得"泛化了"；只切分任务
则让 harness 这条轴完全没被检验。所以任务集按 16/8 切分，而 `codex_style` 从头到尾不出现在
训练里。

**能力探测把住 GPU 花费这道关。** 训练之前，四道门先确认基座模型能否驱动这套 harness。
如果不能，之后测出的任何 gap 都是噪声，诚实的输出是"没有结论"而不是一个数字。

| 门 | 阈值 | 实测（n=32） | 判定 |
| --- | --- | --- | --- |
| G1 tool-call 率 | ≥ 50% | 2,048 条 rollout 中 **77.3%** | 通过 |
| G2 可达单元格（pass@32） | ≥ 1 | **64 个中 42 个** | 通过 |
| G3 跨 harness 差值 | > 0 | **0.72** | 通过 |
| G4 多轮采纳率 | ≥ 50% | 1,584 条调用工具的 rollout 中 **100%** | 通过 |

**Go/No-Go：GO。** Qwen2.5-0.5B-Instruct 能驱动四个可训练 harness。平均轮数 1.78，平均工具调用
1.04 次。

探测只覆盖可训练池——`scripts/probe.py` 通过 `TRAIN_HARNESSES` 解析名字，因此够不到
`codex_style`。这是有意为之，不是疏漏：held-out harness 是这次测量的因变量，一个拿它来校准自己的
能力闸门会把被测的东西花掉。代价是这个闸门对 `codex_style` 什么都没说，而 eval 后来显示它在上面
得 `0/32`——所以"GO"是关于训练能看见的那四个 harness 的结论，不是关于全部五个。

## 测量实际发现了什么

主消融现在已经跑过了——数字，以及它为什么比协议设计时要解决的问题更少，见下文**结果**一节。
这里测出来的是它的前置条件，而这个前置条件推翻了本文件早先的一个说法。

本仓库早先的一个版本报告过：**"64 个单元格里有 33 个（52%）是死的"**，以及
**"T2 在全部 16 个单元格上都是死的"**。这两个数字都来自一次 n=8 的扫描。在 n=32 上重跑同样的
单元格、rollout 数翻四倍之后，说明它们是欠测量的产物，而不是一个发现。

| | n=8 | n=32 |
| --- | --- | --- |
| 可判为 **live** 的单元格 | 0 | **31** |
| 可证明为 **dead** 的单元格 | **0** | **0** |
| **欠测量** 的单元格 | 64 | 33 |
| 因证据太薄被丢弃的单元格 | 33 | 29 |
| 平均 GRPO 组信号 | 0.419 | 0.491 |
| 分层：frontier / unresolved | 11 / 53 | 23 / 41 |

**在 n=8 时，64 个单元格里没有一个可分类。在 n=32 时，依然没有一个单元格可以被证明是死的。**
旧的"52% 是死的"其实是 33 个被点估计过滤器丢掉的单元格，而被丢掉不等于死。

把算术写下来就不微妙了。在 `G` 次生成、通过概率 `p` 的条件下，一个组**不含**梯度的概率是
`p^G + (1−p)^G`。当 `p = 0.05, G = 8` 时它是 **0.6634**——也就是说仍有 **33.66%** 的组带梯度。
只有 `p = 0` 和 `p = 1` 是可证明死掉的。观测到的 `0/8`，其 95% 上界是 **0.312**（精确形式
rule of three，`1 − (1−c)^(1/n)`；大家更熟的 `3/n` 给出 0.375，那是大样本近似），而一个真实
通过率为 0.30 的单元格，大约有 5.8% 的概率表现为 `0/8`。

**随后训练证实的是这条算术，而不是那个过滤器。** 单 harness 臂——64 步、全部 16 行保留、丢弃
0 行——记录的 `frac_reward_zero_std` 均值为 **0.617**，其中 **64 步里有 26 步**恰好等于 1.0
（完全无梯度）。若每一行都落在信号带下沿，公式预测 0.6634；实测的 0.617 就是这个量在一次真实
运行上的取值。按旧过滤器，预期会得到一次几乎全死的运行——它把 16 行里的 8 行丢掉了——而实测
59% 的步带梯度，正是那个过滤器会扔掉的东西。

所以判定是三值的，而中间那个值才是诚实的那个：

* **`DEAD`**——置信区间完全落在信号带 `[0.05, 0.95]` 之外。零通过时需要 **n ≥ 73**；观测到一次
  通过则需要 n ≥ 110。
* **`LIVE`**——区间被包含在带内，**并且**宽度不到带的一半。只满足包含是不够的：`1/2` 给出
  `[0.09, 0.91]`，它确实被包含在一条 0.90 宽的带里，却什么都没定位到。当 `p = 0.5` 时，第一个
  满足宽度上限的 `n` 是 **16**（`8/16` 过，`7/15` 不过）。
* **`UNDER_MEASURED`**——其余全部；在 n=8 时就是全部。

用 Wilson 区间而不是 Wald，因为 Wald 在零通过时退化成 `[0, 0]`——恰好制造出我们正要消除的
那种虚假确定性。Wilson 在 `n=8` 时围绕收缩估计 `(z²/2)/(n+z²) = 0.1622`（而不是 0）居中，
给出上界 **0.3244**。

n=32 的矩阵在**两个方向**上推翻了 n=8 的个别读数，这是旧数字属于噪声的最清楚证据：

| 单元格 | n=8 | n=32 | 旧读数是什么 |
| --- | --- | --- | --- |
| `bash_minimal` × t1-04 | 1/8 = 0.125 | **1/32 = 0.031** | 大致对了，确实接近死 |
| `bash_minimal` × t1-08 | 2/8 = 0.250 | **22/32 = 0.688** | 被严重低估 |
| `react_tools` × t1-08 | 0/8 = 0.000 | **3/32 = 0.094** | 被读成"死"——其实只是罕见 |
| `longctx_summary` × t1-08 | 0/8 = 0.000 | **6/32 = 0.188** | 被读成"死"——其实只是罕见 |

n=32 下按层：**T1** live 26、dead 0、欠测量 6；**T2** live 0、dead 0、欠测量 16；**T3** live 5、
dead 0、欠测量 11。

**T2 依然是真问题，而现在它是一个被测量过的问题。** 四个 harness 在四个 T2 任务上全部得
0/32——16 个单元格里 15 个恰好为零，第 16 个是 1/32——所以 0.5B 模型无法足够可靠地完成"两步
读取再抽取"的流水线，以至于从未通过。这是一致的**能力**失败，而不是 harness 差异。**不能**
支持的是把它叫做死：在 n=32 时 `0/32` 的区间是 `[0, 0.107]`，而一个真实通过率为 0.10 的单元格
在 57% 的 GRPO 组里仍然带信号。

**结论，直说。** T2 的解法是任务生成器，不是手写一个更简单的层：T2 的失败是难度标定问题，
而参数空间能移动它。消融应当跑在一套重新生成的、其 frontier 单元格确实落在 frontier 上的任务集
上——这正是分层的作用。

**以及一处更正。** 本文件早前的一个版本写着 `rsi/band.py` 里的 `steer`「已经在返回这个覆盖值了」，
言下之意课程回路是闭合的。它并没有闭合：`steer` 只能从它自己的测试里到达，没有任何流水线脚本
调用过它。机制存在、有测试覆盖、但从未被执行过——这正是本仓库反复在自己身上抓到的那个模式。
现在它被接上了，位置是 `rsi/curriculum.py` 与 `pipeline.sh` 的第 3 阶段；而接上它的过程暴露了
两个被「缺失的调用点」掩盖的 bug：

* **难度过滤器把转向功能关掉了。** GenEnv 的 `|p̂ − α| > k_min` 被按**单任务**应用；而 `mastered`
  是 `p > 0.9`、`out_of_reach` 是 `p < 0.1`，所以每一个 `steer` 会去动的单元格距离 `α = 0.5`
  都至少 0.4——远在 0.1 的带之外。过滤器恰好拒绝了那条规则存在就是为了移动的单元格，第一次实跑
  报出的就是 `moves: 0`。该规则现已回到 GenEnv 写下它的作用域：批次级。这个错误的算术被钉在测试里。
* **任务 id 不匹配会产出静默错误的覆盖值。** 随仓库发布的 scan 测的是 24 任务套件（`t1-01`），
  而生成的 batch 带的是哈希 id（`t1-8f87ad9e`），两个集合不相交。`steer` 需要任务的参数，所以
  查不到时它回落到自己的默认值，为一个不存在的任务产出覆盖值——而输出里没有任何东西看起来是错的。
  现在这是一个显式的拒绝，并且 id 重叠数会被报出来。

有一个代价值得在任何人期待「很快就有重新生成的任务集」之前说清楚：**转向要真正触发，每格需要
`n = 64` 次 rollout。** 在 `n = 32` 时，全通过的单元格 Wilson 下界是 0.8928，刚好低于
`MASTERED_ABOVE = 0.9`，于是它判定为 `unresolved`，无论点估计看起来多清晰都不会产生任何移动。
上面那张表所用的 scan 就是 `n = 32`。

**以及第二处更正——因为第一处更正自己也说过头了。** 上面那段写着课程「现在它被接上了」。它是
*可到达*了，但并没有被*到达*。接线确实落在了 `rsi/curriculum.py`，它暴露的两个 bug 也确实修了——
但 `pipeline.sh` 仍然把 `scan_all.json` 交给课程，那份 scan 测的是随仓库发布的 16 个 id，而正在
被转向的 batch 带的是生成出来的 id。于是每次运行的计划都是空的，理由由代码自己报了出来：

```
skipped: the scan measures 16 task ids, none of which are in the batch of 2;
         no task can be steered
```

这条消息是对的，同时它就是失败本身：回路的**顺序**错了。batch 必须先存在才能被测量，必须先被测量
才能被转向，而流水线直接做了第三件事、跳过了前两件。那个手工的两命令绕法被写进了一条注释
（`probe.py --from-batch`，然后用 `CURRICULUM_SCAN` 重跑），并且从未被执行过——这和 `steer`
本身是同一个模式：有文档、可用、没跑过。

修法是顺序，`tests/test_rsi_closed_loop.py` 现在把它钉住了——生成在扫描之前，扫的是**那个**
batch，课程与训练读的是同一份 scan，而 id 不匹配是致命错误而不是一份静默的「0 次移动」计划。

**同一个模式的第三处出现在训练器里。** `train.py` 的行是从硬编码的 `TRAIN_TASK_IDS` 构造的，
所以即使某份 scan 覆盖到了生成任务，也没有任何一个能到达梯度。它现在接受 `--batch`（`rsi/batch.json`
或 `rsi/env_batch.json`，两者是不同形状的任务），并把 batch 注册进 `Agent.run` 解析 id 所用的
进程级任务注册表。`--require-signal` 会在 scan 与行完全不重叠时拒绝，`--dry-run` 在加载模型之前
就停下，于是接线不占 GPU 也能查。

这对训练曲线意味着什么：两个 128 步的臂训的是从十六个冻结 id 里抽出的 64 行，日志里的
`epoch` 字段给出的实际值是 **4.0 个 epoch**（step 1 时 `epoch` 为 0.03125 = 2/64，step 128 时为
4.0）。第 65 步之后 reward 进入平台、`frac_reward_zero_std` 升到 0.66，这与「在一份小的固定集合上
记忆」是相符的——而生成任务本来就是本仓库用来逃出这个局面的机制。这里**没有**声称的东西同样要说
清楚：0.5B 模型跑 4 个 epoch 并不显然足以饱和，所以平台只能说明「这套配置停止改进了」，不能证明
「任务集是唯一原因」。但它确实证明了梯度从未见过任何一个生成任务，而这无论平台意味着什么都是缺陷。

**生成任务的难度是被测过的，但测得很薄。** 这一点必须说准，因为两个方向上都容易说过头。对生成
产物的两份 scan 是：

| 产物 | 测的是什么 | 结果 | 判定 |
| --- | --- | --- | --- |
| `rsi/scan_batch_smoke.json` | 生成的字符串任务，`n=2`，单 harness | **0/16** | `under_measured`（hi 0.1936） |
| `rsi/env_scan.json` | 生成的环境任务，池化 | **0/96** | `dead`（hi 0.0385） |

也就是说：**环境**批次是真实的能力地板——0/96 排除了任何高于 0.0385 的通过率。但**字符串**批次
不是：每格 `n=2` 时它的上界是 0.1936，这与 shipped 套件 T1 的 0.3613 在低端重叠，也完全覆盖了
T3 的 0.0859。这一节的初稿写着生成器「产出的任务策略做不了」，而 0/16 的 scan 并不支持这句话。
shipped 套件展示出来的是一条值得沿着它做转向的难度曲线：

| tier | 池化通过率（shipped 套件，`n=32`） | 判定 |
| --- | --- | --- |
| T1 | 370/1024 = **0.3613** | `live` |
| T2 | 1/512 = **0.0020** | `dead` |
| T3 | 44/512 = **0.0859** | `live` |

而 `sample_params` 在四个 tier 上均匀采样，所以一个生成批次里大约有一半落在 T4——一个 shipped
套件从未触碰过的档位。这是否太难就是那个开放问题，而它只需要在 `n >= 32` 下对生成批次做一次 scan
就能回答。在这份数据存在之前，诚实的说法是「生成器的标定尚未被测量」，而不是「它是错的」。

## 图表

全部十三张都由 `./run.sh tools/plot.py` 从记录下来的产物重新生成，每一张都在 CI 里被校验：
只用标准库去解码提交进仓库的 PNG 字节——一个画出空白画布的绘图 bug 依然会写出尺寸合理、格式
合法的 PNG，所以检查的是像素，而不是文件是否存在。

**扫描结果，以及为什么 n=8 不够。**

![通过率矩阵](docs/figures/fig01_scan_matrix.png)

![每个单元格的 Wilson 区间](docs/figures/fig02_measurement.png)

**判定分布（旧样本量对新样本量），以及说明"死"到底该是什么意思的解析信号曲线。**

![判定计数](docs/figures/fig03_verdicts.png)

![GRPO 组信号概率](docs/figures/fig04_grpo_signal.png)

**每个单元格在难度曲线上的位置，以及那些零背后的机制。**

![难度分层](docs/figures/fig05_bands.png)

![rollout 为什么停下](docs/figures/fig06_stop_reasons.png)

**轮次与工具调用行为，按结果和按层拆分。**

![轮次与工具调用](docs/figures/fig07_turns.png)

![工具调用对奖励](docs/figures/fig08_toolcall_reward.png)

**退火预算（这是算出来的，不是测出来的），以及它约束的那份账本。**

![退火编辑预算](docs/figures/fig09_edit_budget.png)

![harness 演化账本](docs/figures/fig10_ledger.png)

账本这张图从磁盘上**存在的那份** ledger 绘制，标题里写明是哪种评分模式。这个标注不是装饰：
一条回放轨迹和一条实测轨迹都是"一条上升的线"，而只有其中一条意味着 harness 真的变好了。
现在两份 ledger 都在，所以 fig10 画的是**实测**那份——`outputs_rollout/rsi/ledger.jsonl`，
标题为 `score mode: rollout (measured)`；回放那份是实测臂尚未运行时它退回的目标。

**两个自构建轴：生成批次上的闸门结果，以及标出零梯度步的训练曲线。**

![四闸门验证](docs/figures/fig11_validation.png)

![训练曲线](docs/figures/fig12_training.png)

**结果本身——三条臂，以及各自留下的 gap。**

![消融](docs/figures/fig13_ablation.png)

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

**两条训练臂按优化步数对齐，而不是按 epoch 对齐，这个选择是有影响的。** `single` 看到 16 行，
`multi` 看到 64 行——同样 16 个任务叉乘四个 harness——所以按 epoch 对齐会让 `multi` 拿到四倍的
梯度更新，于是 gap 上的任何差异都会和训练预算的差异混在一起。因此两条臂跑相同的 `--steps`，
`multi` 每个 epoch 看一次的行数相当于 `single` 的四分之一。这个比较问的是"相同算力、不同的
环境多样性"，而这正是实验要问的问题；另一种做法回答的是"训练更久是否有帮助"，那不是本实验的
问题。

## 结果

同一个进程、同一个种子（42）、共享 harness 实例，5 个 harness × 8 个留出任务 × 4 次 rollout =
**每条臂 160 次 rollout，合计 480 次**。每个单元格 `n = 4`。

| 臂 | train harness reward | held-out reward | gap |
| --- | --- | --- | --- |
| baseline（不训练） | 18/128 = **0.1406** | 0/32 = 0.0000 | **+0.1406** |
| `train-single-s64` | 34/128 = **0.2656** | 0/32 = 0.0000 | **+0.2656** |
| `train-multi-s64` | 39/128 = **0.3047** | 0/32 = 0.0000 | **+0.3047** |

**训练是有效的，在它见过的那些 harness 上。** 两条臂在 train harness 上都超过 baseline，而且效应
大于抽样噪声：single `d = +0.1250`、`z = +2.49`；multi `d = +0.1641`、`z = +3.15`。single 臂的
平均 reward 在 64 步的四个四分位上是 `0.367 → 0.488 → 0.520 → 0.508`，也就是说它学到了东西然后
进入平台期，而不只是漂移。

**multi 臂的曲线没那么干净，这一点值得写出来。** 它前期同样在涨——`0.227 → 0.313 → 0.441`——然后
在最后一个四分位**回落到了 0.352**。它 64 步的总体均值（`0.333`）**低于** single（`0.471`），而它
eval 的 train harness reward（`0.305`）**高于** single（`0.266`）。这两件事不矛盾——训练均值包含了
早期那些正在吸收四个 harness 方差的步骤，而 eval 只读最终 checkpoint——但"最后一个四分位下滑"不是
平台期，而每个四分位只有 16 步，无法把"训练后期的回归"和噪声分开。下面的消融数字来自最终
checkpoint，所以如果那个 checkpoint 恰好是一个局部低谷，multi 臂的优势就被高估了。这次运行里没有
任何东西能排除这一点。

**主对比不显著，而且算术上解释得很清楚。** `multi` 比 `single` 高 `d = +0.0391`、`z = +0.69`。
由于 held-out 项在**每一条**臂上都等于 0，`d_gap ≡ d_train` 严格成立——gap 的差值就是 train 的
差值，没有任何东西被减掉。所以诚实的表述是：

> 观测到的方向支持多 harness 训练，但在当前样本量下该效应**与零不可区分**。本仓库预先登记的
> 假设**没有得到支持**，但也没有被证伪。

train 均值上的 Wilson 区间把分辨力说得很明白：baseline `[0.0908, 0.2114]`、single
`[0.1968, 0.3482]`、multi `[0.2316, 0.3892]`。single 与 multi 在大部分区间上重叠。两个数字说明
设计差了多少：在每条臂 `n = 128` 次 rollout 下，**最小可检测效应**是 `d ≥ 0.1581`，而观测到的效应
是 `0.0391`——只有它的四分之一。要在观测到的效应量上达到 80% 检验力，需要**每条臂约 2,094 次
rollout，是本次 eval 的 16 倍**，在这块 GPU 上这不是一个笔记本规模的实验。这是一句预算陈述，不是
借口：数字就是数字，而这次运行能诚实报告的只有方向。

**而且 held-out 项是一个 floor effect，所以这套协议围绕的那个指标根本无法被这次运行检验。**
`codex_style` 在 baseline 臂上就是 0/32——未经训练的模型在它上面完全无法得分——这正是 gap 差值
坍缩成 train 差值的原因。`0/32` 的 Wilson 区间是 `[0, 0.1072]`，判定为 `under_measured`：与
"做不到"一致，但也与真实通过率 10% 一致。

原因**不是** harness。它已经修好、有回归测试，而 `codex_style` 依然读作零。重新跑一次 `diag.py`
的完整记录说明了原因：模型发出了格式正确的 `apply_patch` 调用，然后把裸的答案字符串放进了本该是
unified diff 的位置。

```
parsed calls : [{'function': {'name': 'apply_patch', 'arguments': {'patch': 'reward is one'}}}]
tool apply_patch -> patch: **** Only garbage was found in the patch input.
```

其余条件不变，`*** Add File: answer.txt\n+reward is one` 得 **1.0**，一份真正的 unified diff 得
**1.0**；裸字符串得 0.0。所以这条留出轴测的是"一个 0.5B 模型到底能不能产出 unified diff"，这是
关于基座模型的能力问题，而不是关于训练的泛化问题。**正确的修法是换一个基座模型本来就能驱动的
第二个留出 harness**——否则 held-out 项会一直钉在 0，再多的 rollout 也解不开。

**这次运行真正确立的东西，直说：**

* 在它见过的 harness 上训练，能提高这些 harness 上的 reward
  （`z = +2.49` 与 `+3.15`——真实）；
* multi 对 single 的效应方向与预测一致，`z = +0.69`——有提示性，不构成证据；
* 预先登记的跨 harness 泛化检验**没有跑成**，因为它的 held-out 项是 floor effect；
* GRPO 信号算术在真实运行上成立——见上文 `0.617` 对 `0.6634` 的对照。

三条前进路线，按能解决的问题量排序：

1. **替换留出 harness**，换成一个 baseline 能驱动的，让 `gap = train − held_out` 有活的第二项。
   便宜，且能解开协议。
2. **给 train 对比补足检验力**（每臂约 2,094 次 rollout），如果问题就是那 0.0391 本身而不是 gap。
3. **重新生成任务套件**，让 frontier 单元格真的落在 frontier 上——这是任务生成器的职责，见上文
   T2 的发现。

## 复现

需要 Python 3.12+，Windows 上需要 Git-Bash。核心部分——harnesses、任务、verifier、整个 RSI
层、全部静态守卫——**不依赖第三方库**，所以下面的检查不需要装 torch。

```bash
git clone https://github.com/possibleme2026-lang/rsi-multi-harness-rl.git
cd rsi-multi-harness-rl

# core checks: no torch, no GPU
./run.sh scripts/guard_tool_surface.py
./run.sh tests/test_path_errors.py
./run.sh tests/test_shell_timeout.py
./run.sh tests/test_scan_tooling.py
./run.sh tests/test_verifier_modes.py
./run.sh tests/test_rsi_stats.py
./run.sh tests/test_rsi_generator.py
./run.sh tests/test_rsi_loop.py
./run.sh tests/test_rsi_closed_loop.py
./run.sh tests/smoke_env.py
```

RSI 循环本身在默认模式下不需要模型，所以生成器、四道闸门和 harness 演化全都能在 CPU 上跑：

```bash
./run.sh scripts/rsi_loop.py --rounds 6 --task-batch 30
./run.sh tools/check_figures.py
./run.sh tools/check_readme_i18n.py
```

闭合回路也有一条纯 CPU 路径，改动这四个阶段中任何一个之后，这是检查接线最省的办法：

```bash
# 1. 写出 batch（对 --seed 确定性）
./run.sh scripts/rsi_loop.py --batch-only --task-batch 12 --env-batch 4 --seed 11
# 2. 测量它 —— 这一步才是闭合回路
./run.sh scripts/probe.py --from-batch outputs/rsi/batch.json \
    --n 32 --out outputs/rsi/scan_batch.json
# 3. 不加载模型，检查训练将要消费什么
./run.sh scripts/train.py --mode multi --steps 1 \
    --batch outputs/rsi/batch.json --scan outputs/rsi/scan_batch.json \
    --require-signal --dry-run
```

`pipeline.sh` 默认就跑这个顺序。`TRAIN_ON_BATCH=0` 可以退回「在随仓库发布的套件上训练」的旧行为。

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

产物都在 `outputs/`（可用 `MULTIHARNESS_OUT` 覆盖）。n=32 的扫描是 2,048 条 rollout；
训练和评估更长。

| 脚本 | 作用 |
| --- | --- |
| `scripts/probe.py` | 能力探测 + 难度扫描，写出 `scan_all.json` |
| `scripts/rsi_loop.py` | 两条 RSI 轴：生成并验证任务、演化 harness |
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
  rsi/task_gen.py       轴一 —— 生成任务、环境与参考解法
  rsi/envgen.py         轴四 —— 合成**有状态环境**：工具依赖图、初始状态、链路
  rsi/envtask.py        任务即图上的一条路径，按留下的状态评分
  rsi/harbor_export.py  这样一个任务的 Harbor 包
  rsi/env_adapter.py    同一个任务，换成 harness 池能跑的形态
  rsi/verifier_gen.py   轴二 —— 生成奖励，三种模式之一
  rsi/validate.py       闸门 V1-V4
  rsi/harness_evolve.py 轴三 —— 描述符改动、守卫、评判
  rsi/ledger.py         只追加的 JSONL、退火预算、剪枝、停滞
  rsi/stats.py          Wilson 区间、GRPO 信号、噪声底线
  rsi/band.py           后悔分层与转向信号
  rsi/curriculum.py     α 奖励，以及闭合回路的那份计划
  rollout.py            独立重实现的 TRL 工具调用循环
  _bootstrap.py         仓库根目录 + 产物目录
scripts/                入口脚本（probe、rsi_loop、train、eval、守卫、
                        harbor_local_run、env_batch_make、env_harness_smoke、
                        env_reward_ceiling、env_scan_report、guidance_shape_check）
tests/                  smoke 测试与回归测试
tools/                  绘图、图表校验、README 双语一致性守卫
docs/refs/              从 RRSI、Dream-RSI 与环境合成文献借了什么、为什么
pipeline.sh            完整运行流程，按依赖顺序
```

## 值得记录的 bug

每一个都是"实测与预期不符"发现的，每一个都有回归测试。写进 README 是因为它们全都会静默污染
结果，而不是崩溃。

**每个 harness 都叫 agent 去写 `answer.txt`，于是 96 个 rollout 全是 0.00。** 这是本文件里
最严重的 bug：不是因为它大，而是因为它产出了一个**完整的、可信的、完全虚假的**能力结论。每个
harness 会追加一段静态 `GUIDANCE`——为字符串任务写的，对字符串任务是对的——里面说要靠写
`answer.txt` 提交。而有状态环境任务评的是 `state.json`，所以哪怕解法完全正确也只会拿 0.0；而且
那段话从不提 `envtool`，于是 agent 通向环境的唯一路径就是猜工具名。transcript 里正是如此：
`update_inventory_item_by_id: command not found`、`cat /var/log/syslog`、
`echo "set balance to closed"`。

Harbor 导出路径没有这个 bug，因为 `_instruction_md` 会渲染一张工具表。但那张表是在
`harbor_export` 里生成的，永远到不了 harness 池——池读的是任务 **dict**，不是包——于是两条路径
分岔了，而且**只有其中一条被执行过**。修法是给任务加 `guidance` 覆盖，由
`core._instruction` 优先于静态文本。

**第一次修没修好，而且它被验证所依据的那个数字是往上涨的。** 那个覆盖把环境的命令渲染成了一张
裸列表——`envtool list_tickets`、`envtool set_state <id> <value>`。这读起来就是一张**工具列表**，
而 prompt 里本来就有一张：chat template 会在指令上方把 harness 自己的工具渲染成 schema 块。面对
两份同形状的列表，模型把它们合并了，把 `envtool list_tickets` **当成工具名**去调：

```
envtool list_tickets({'query': 'state=open'})
-> Tool envtool list_tickets not found. Available: ['bash']
```

实测：全部 96 个 rollout 里 **70%** 是这一种形状，而且模型在根本没有该命令的环境上凭空造出了
`list_accounts`、`list_items`——它在回答 prompt 的**形状**，不是它的内容。与此同时工具调用率从
35.4% 升到 **81.2%**，通过率仍旧恰好 `0.00`，因为**调用一个不存在的工具也算工具调用**。那个本该
检测「模型能否起码进入循环」的门，被它本该抓住的失败模式本身满足了。**一个错误调用就能通过的门
不是门。**

现在的 guidance 把命令介绍为**shell 命令行**，点名了通向它们的那个 shell 工具，并把错误形状明确
标为错误。`scripts/env_scan_report.py` 会对每次调用分类形状（`no_call` / `bad_args` /
`unknown_tool` / `query_only` / `wrong_target` / `mutated_ok`），于是调用率再也不能单独被读成进展
——这正是当初发现它的方式。一矩阵的零，在「模型从没调用工具」「调用了一个不存在的工具」「调对了
工具然后停下」三种情况下长得一模一样。

**一个审计全绿却跑不起来的包。** `audit_package` 检查了 `environment/assets/tools.py` 存在、
格式正确、列出了正确的工具——全都为真。它没检查工具是否**可被派发**，而有两个不可：`restore`
和 `pin` 在进程内评分器（`envtask.apply_trace`）里实现了，导出程序里却缺失，于是导出的环境与
驱动它的解法已经漂移。实测：**30 个包 0 个能执行**，每一个都报 `unknown tool: restore`。而测试
套件全程是绿的，因为结构审计和分数比较在它们描述的东西已经坏掉时都照样成功。是本地执行器抓到的。
教训是这个仓库反复学到的那条：**没被执行过的包只是一个主张，不是 benchmark。**

**难度旋钮说了三次谎，三种不同的方式。** `n_steps` 是环境轴唯一的难度旋钮，每修一层就露出下一
层：（1）链路被**查询**填充，查询不贡献参考步骤也不贡献检查点，于是 4 步链路评的是 1 步解法；
（2）链路被参考**拒绝使用**的工具填充（`reset_log`、`bulk_update`），43% 的任务有 4 步链路而
参考只有 1 步；（3）上限本身随一次未记录的抽样而变，`n_steps=5` 在 60 个种子里有 27 个被以
"3 个可用变异器"拒绝、33 个以"4 个"拒绝——旋钮的**合法性**取决于种子。现在阶梯在每个设定下
60/60 确定，超过上限 60/60 拒绝。

**任务 dict 里的一个 callable 让批次写不出来。** `env_adapter` 在 `env` 键里放了个 `lambda`
来把每个 rollout 指向自己的状态文件——意图正确，但它让 `json.dumps` 抛错，于是环境批次根本写不
出来，跨 harness 差距就只能永远在字符串任务上测。修法是 `reset` 时解析的声明式模板。而**危险**
的修法是把那个键丢掉：没有 `ENVTOOL_STATE`，每个 rollout 都会读同一个默认路径，所有 rollout
静默共享一个状态文件——一个看起来能跑、实则让所有 rollout 互相耦合的环境。

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

**一个生成出来的、会把摘要送出去的奖励。** `verify_mode` 与 payload 是独立抽取的，于是生成器
产出了"三字符答案 + `python_exit`"——正是 `verifier_gen` 自己的文档点名说错的那种组合，因为
`check_script` 会落在 agent 的工作目录里。在 300 个任务上实测（seed 0）：**96 个（32%）受影响**。
八个 seed 下这个数字在 300 里占 79–99，所以比例是随 seed 变的；这里写明 seed，是因为默认的
`--seed 11` 给出的是 91 而不是 96。现在
生成器去问 `recommend_mode`，而不是相信自己手里的参数向量；受影响数为 0，同时显式指定的长答案
`python_exit` 依然被尊重。修好它又暴露出同一个提交里的第二个缺陷：`summarise_batch` 统计的是
`params["verify_mode"]`，于是修复之后覆盖率报告声称有 8 个 `python_exit` 任务、而实际只有 1 个
——现在它报告任务**实际使用**的模式，并把请求的模式并列在旁边。

**一个靠猜的 `DEAD` 阈值。** `stats.py` 的文档说零通过时排除信号带"大约需要 `n >= 128`"。
真实边界是 **73**（`wilson_interval(0, 73)[1] = 0.04999`；`n = 72` 给出 0.0507）。这个差距不是
装饰性的：73 条 rollout 是一块笔记本就能做的扫描，128 条则是"停下来重新设计"的量级。文档现在
同时写明两个边界，并有测试钉住它们。

**`rule_of_three_upper` 用错了底数。** 它算的是 `1 − confidence^(1/n)`，在 `n=8` 时返回
0.0064——小了 49 倍——而它自己的文档写的是 0.312。这是拿文档做算术对出来的。正确的底数是
`1 − confidence`；之所以有测试，是因为这个笔误产出的浮点数看上去很合理。

**`classify_cell` 会把 `0/32` 判成 live。** 一个 `(hi − lo) <= 0.25` 的捷径接受了任何窄区间，
包括 `[0, 0.107]`——它确实窄，却完全在信号带之外。已改为"包含 **且** 宽度不超过带的一半"。

**第三个相关修复：** `errs_report.py` 最初会给出**假绿**结论——"没有 harness 缺陷"——而实际上
有 25–27 条原始 OS 报错躺在它的兜底分类里。现在它有明确的 OS 报错类别，并会让该次扫描判为
无效。一个不可能失败的守卫比没有守卫更糟。

**一个读起来像模型失败、实际是 harness 缺陷的问题——是真的、已修，但并不是它被归咎的那个零分的原因。**
`codex_style` 的 `apply_patch` 回退路径，把
`|| echo '[error] patch tool unavailable or patch failed'` 拼在了 heredoc 终止符**之后**：

```bash
command -v patch >/dev/null 2>&1 && patch -p0 -f <<'__PATCH__'
<patch body>
__PATCH__
|| echo '[error] patch tool unavailable or patch failed'
```

heredoc 在终止符处结束，于是那个 `||` 变成了独立的一行，整条命令是一个 bash **语法错误**。
任何真正到达 shell 的 unified diff 都返回 `syntax error near unexpected token '||'`，从未被应用。
harness 的 `*** Add File` 捷径在日常测试里把它掩盖了——那条路径在到达 shell 之前就返回了——
于是这个工具被声明、被注册、可达，却从来不工作。这正是 `guard_tool_surface.py` 想防住的失败，
而且落在它看不见的地方。

修法是把喂给 heredoc 的命令用 `{ ... }` 包起来，好让一个 `||` 合法地跟在后面，并从
`_run_shell_rc` 读退出状态而不是解析文本——`python_exit` 犯过同一个错误。回归测试断言：合法 diff
被应用、文件出现、且打过补丁的答案得 1.0。

**这个缺陷是真的。但它并不是 `codex_style` 拿零分的原因，而本节初稿是这么写的。** 这个区分很重要，
所以它被记录下来，而不是悄悄改掉。修复**之后**重跑 eval，`codex_style` 依然是**三条臂全部 0/8，
包括 baseline**——所以语法错误不可能是原因。重新跑一次 `diag.py` 显示出真正的原因：

```
decoded      : '<tool_call>{"name": "apply_patch", "arguments": {"patch": "reward is one"}}</tool_call>'
parsed calls : [{'function': {'name': 'apply_patch', 'arguments': {'patch': 'reward is one'}}}]
tool apply_patch -> patch: **** Only garbage was found in the patch input.
```

模型选对了工具、也选对了参数名，然后把裸的答案字符串放进了本该是 unified diff 的位置。`patch`
拒绝它是正确的。其余条件不变，三种载荷并排对比：

| 传给 `patch` 的东西 | 结果 |
| --- | --- |
| `*** Add File: answer.txt\n+reward is one` | **reward 1.0** |
| `reward is one`——模型实际发出的 | 0.0，`Only garbage was found` |
| 一份真正的 unified diff | **reward 1.0** |

提示词里确实写了 `` `apply_patch` takes a unified-diff body``。0.5B 模型照样不产出它。所以 harness
的管线修好了，harness 也**不是**瓶颈——但留出 harness 依然读作 0/8，而这个原因与跨 harness 泛化
毫无关系，这才是让测量作废的部分。见"结果"一节。

**一个让主指标变得不可检验的 floor effect。** `codex_style` 在 **baseline** 臂上就是 0.0000，
也就是说未经训练的模型在它上面完全无法得分。留出项被钉在 0 时，`gap = train − held_out` 只可能
增大："多 harness 训练缩小 gap"这个说法无法被这次测量证伪，而第一次完整消融就是据此报告
"单 harness 训练得到更小的 gap——假设不成立"的。那个结论依然不成立。这次运行真正确立了的东西在
"结果"一节。

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
