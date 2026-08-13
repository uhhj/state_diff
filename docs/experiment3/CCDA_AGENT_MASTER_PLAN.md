# CCDA Agent Research Master Plan
## 科学主线、执行边界与工程守则

> **PB3 current status (2026-08-13)**: PB3-R2 is `PB3R2_REPLAY_FLOOR_CALIBRATED` and PB3-R3 is frozen at `PB3R3_ALIGNMENT_RULE_PREREGISTERED`. Formal PB3 Resume stopped with `PB3_R3_LIVE_BRANCH_ALIGNMENT_FAILED` at `failure_component=targeted_replay_alignment`: all 20 formal branch states were evaluated, 17 passed and rollouts 39/55/97 at t20 failed; rollout 55 also failed exact winding-index equality. The global pre-future barrier worked: live-pair revalidation, branch snapshot creation, future suffix, Gate 4, PB4, and StateDiff training were not executed. Preserve the committed 61.945756736 um coordinate-RMSE rule, 50 um snapshot max-abs rule, and frozen 10-pair shortlist; do not drop/replace pairs or resume automatically.

> **用途**：本文件用于约束 Agent / Codex / 自动化研究执行，防止项目在工程细节、审计、防御性编程、无效仿真调参或局部诊断上偏离主研究目标。  
> **适用项目**：CCDA — Contact-Conditioned Deformation Branch Ambiguity  
> **当前主仓库**：`uhhj/state_diff`，主研究分支 `Experiment3`  
> **当前已发布仿真基座**：DLO-Lab，固定 revision `c5026a9416b03c6bc5186eba13cd4ffd4c0e7796`  
> **当前科学状态**：Wiring-post 主路线已停止。DLO-Lab Wrapping 的 PB2-A/B 已正式通过：官方任务复现、task-native winding 定义和语义 winding 转换均成立。PB2-C 已正式通过并得到 5,534 个 frozen discovery candidates。PB3 已在 future suffix 前因 targeted replay alignment 失败而阻断，未产生 Gate 4 结论。PB3-R1 未确认 acquisition-history dependence。PB3-R2 在第 8 个 logical worker 无 artifact 的 returncode 120 后 blocked；当前立即任务为 PB3-R2E execution-harness diagnosis 与 resume freeze。PB3 future suffix / Gate 4 仍未运行；在 PB3 恢复并通过以及 PB4 完成前禁止启动 StateDiff/CFPM 大规模训练。
> **原则**：科学结论必须严谨；工程执行必须简洁；一旦机制证据充分，立即推进端到端系统，不继续无限审计。

---

# 1. 项目最终目标

CCDA 研究的核心问题是：

> 柔性物体在机器人可观察历史、机器人自身状态和过去动作几乎相同的情况下，因为内部或遮挡区域的接触状态不同，在当前时刻看起来几乎一样，但执行相同未来动作后进入不同未来，并最终导致不同的控制决策。

目标不是单纯证明“存在不可观测接触”，而是建立完整因果链：

```text
hidden contact / routing state
        ↓
deployable sensing
        ↓
different future deformable dynamics
        ↓
different control consequence
        ↓
sensor-conditioned prediction / control improves closed-loop result
```

最终系统必须能够形成：

```text
observable history H
+
deployable sensor history Z
        ↓
StateDiff / StateDiff-FT
        ↓
CFPM contact-conditioned future guidance
        ↓
future deformable state
        ↓
IDM / action selection
        ↓
closed-loop deformable manipulation
```

项目不能长期停留在：

```text
环境搭建
→ debug
→ audit
→ threshold tuning
→ 再 audit
```

而无法进入模型和闭环实验。

---

# 2. CCDA 的科学判定条件

一个正式 CCDA pair 至少需要满足以下五个条件。

## Condition 1 — Observable-history equivalence

两个状态的可部署可观察历史应相同或足够接近：

\[
H_A^{obs} \approx H_B^{obs}
\]

包括：

- 柔性物体可见部分；
- 机器人状态；
- 必要的历史窗口；
- 不得依赖部署时不可获得的信息。

“近似”必须通过预先定义、可解释的 observable metric 判断。

## Condition 2 — Past-action equivalence

最近动作历史应相同或足够接近：

\[
A_{A,t-h:t} \approx A_{B,t-h:t}
\]

不能把机器人过去已经执行了明显不同控制的两个状态称为同一 CCDA branch point。

## Condition 3 — Hidden-state difference

必须存在可信的隐藏物理状态差异：

\[
z_A \neq z_B
\]

例如：

- 接触 / 非接触；
- stick / slip；
- jam / release；
- routing / winding topology；
- prestress；
- hidden local contact mode。

优先级：

```text
task-native privileged state
>
simulator-native interaction state
>
连续且物理可解释的 oracle descriptor
>
人工几何 proxy
```

**禁止把未经 native / task-native 验证的 heuristic proxy 当成物理 ground truth。**

## Condition 4 — Same-action future bifurcation

从两个 branch state 出发施加相同未来动作：

\[
a_{t:t+H}^{A}=a_{t:t+H}^{B}
\]

并得到明显不同未来：

\[
D_{branch}(h)
\]

必须显著超过 deterministic repeat / simulator uncertainty floor：

\[
D_{branch}(h) \gg D_{repeat}(h)
\]

不能再把简单的：

```text
future/current >= 2
```

作为正式 Gate 4。

多 horizon 应各自使用自己的有效时间范围，不能统一截到最长 horizon。

如果只在离散 horizons 上取最大值，必须使用：

```text
horizon_of_maximum_sampled_future_distance
```

而不是误导性的：

```text
peak horizon
```

因为前者只表示“所采样 horizons 中最大”，并不声称真实连续时间峰值在那里。

## Condition 5 — Control relevance

hidden state 和 future bifurcation 最终必须影响最优动作或控制损失：

\[
a_A^* \neq a_B^*
\]

或至少两个状态下的错误动作 regret 明显不同。

如果 hidden state 对未来预测有影响，但对控制没有影响，则它不是最终需要解决的 CCDA 控制问题。

---

# 3. 模型研究主线

在任务机制通过前，禁止启动大规模模型训练。

当 CCDA task structure 已经可信后，进入以下模型链。

## B0 — State-only StateDiff

参考 StateDiff 思路，用低维 deformable state：

\[
S_t=[D_t,R_t]
\]

其中：

- \(D_t\)：柔性物体关键点 / task DOF；
- \(R_t\)：最小机器人 / EE 状态。

建立 state-only future predictor：

\[
p_\theta(S_{future}|H_t)
\]

用于测量 branch ambiguity。

## B1 — Direct sensor-conditioned StateDiff

部署可获得的因果传感历史：

\[
Z_t
\]

例如：

- wrist F/T；
- robot reaction；
- tactile；
- tracking residual；
- 其他真正部署可获取的信息。

编码：

\[
c_t^{sensor}=E(Z_{t-h:t})
\]

并建模：

\[
p_{\theta}(S_{future}|H_t,c_t^{sensor})
\]

传感器是 conditioning，不要求作为 future state 本身被预测。

## B2 — StateDiff + CFPM

建立：

\[
G_\phi(H,Z,Y_k,k)
\]

在 reverse diffusion 过程中利用 contact/sensor information 引导 future-state denoising。

## B3 — StateDiff-FT + CFPM

结合直接 sensor conditioning 和 reverse-process guidance。

## B4 — Oracle

只作为 upper bound：

```text
oracle hidden interaction state
```

不能进入部署模型。

---

# 4. 已完成任务路线与正式结论

## 4.1 OHJ-Cable

Occluded Hidden-Jam Cable。

最终结论：

```text
OHJ_CONTROL_STRUCTURE_INSUFFICIENT
```

虽然能够产生一定 future separation，但最优控制动作没有形成预期的 hidden-state-conditioned switch。

因此停止继续调 OHJ geometry。

## 4.2 DHR-Cable

Directional Hidden-Release Cable。

正式 smoke 结果：

```text
PHASE0F0_DHR_ACTION_SWITCH_FAIL
```

关键问题：

- 目标 hidden contact mechanism 实际没有 engage；
- intended pocket contact fraction 为 0；
- 最优动作不符合设计的 FREE/JAM action switch；
- common load / controller effect 主导结果。

因此停止 DHR。

---

# 5. Published benchmark 路线

为了避免继续人工设计任务，研究转到已发表论文使用的仿真任务。

固定使用：

```text
DLO-Lab
revision:
c5026a9416b03c6bc5186eba13cd4ffd4c0e7796
```

原则：

> 优先使用 published task 的原始 geometry、physics、reward 和 task semantics，只增加 observation mask、logging、oracle diagnostics、snapshot branching 和模型 wrapper。

禁止为了制造 CCDA：

- 修改物理参数；
- 修改 contact solver；
- 添加 hidden obstacle；
- 改 reward；
- 改目标；
- 人为设计一个只服务于论文结论的新机制。

---

# 6. Wiring-post 路线最终结论

## PB0 — Natural pair mining

正式冻结协议：

```text
same timestep
observable Chamfer <= 10 mm
EE distance <= 10 mm
hidden descriptor different
future t+10 / current >= 2
```

512×101 rollout 后：

```text
formal candidate = 0
```

正式 verdict：

```text
PB0_WIRING_POST_NO_NATURAL_CCDA_PAIRS
```

正确解释仅为：

> preregistered PB0 protocol 找到 0 candidate。

不能解释为 Wiring-post 不存在 CCDA。

## PB0-S REV3 — Screening assumption audit

发现：

```text
original hidden-descriptor pairs = 27,569
formal t+10 ratio pass = 0
```

并明确发现：

- future separation 大部分在 `t+1/t+2` 更明显；
- `t+10` 明显 horizon-sensitive；
- cross-time / Chamfer approximate audit 的 0 candidate 不能作为不存在证据；
- multi-horizon 必须各自使用可用时间范围；
- ordered RMSE 只能做诊断，不能通过 visibility intersection 人为把状态拉近。

因此不能再把原 `t+10` ratio 当作 Gate 4。

## PB0-S REV4 — Hidden descriptor robustness

27,569 个 original-hidden pairs 最终全部是：

```text
contact_proxy_only = 27,569
original_wrap_only = 0
contact_and_original_wrap = 0
global-angle-supported = 0
```

旧 hidden descriptor 实际依赖：

```text
surface clearance <= 3 mm
```

的 binary geometric proxy。

虽然有些 pair 远离 3 mm threshold，不只是微米级翻转，但：

> 离阈值远仍不等于真实 simulator-native contact difference。

所以这些 pair 不得直接进入 Gate 4。

## PB0-T REV2 — Simulator-native contact confirmation

对 REV4 shortlist：

```text
12 unique targeted rollouts
20 pairs
```

执行 exact targeted replay。

结果：

```text
global_replay_alignment_valid = true
native mismatch confirmed = 0
stable-core support = 0
confirmed shortlist = 0
```

正式 verdict：

```text
PB0T_NATIVE_POST_CONTACT_MISMATCH_NOT_CONFIRMED
```

因此：

> REV4 的 3 mm proxy-only candidate generation mechanism 没有得到 simulator-native contact state 支持。

正式结论不是：

```text
Wiring-post 不存在 CCDA
```

而是：

```text
当前 Wiring-post candidate-generation route
不能提供可进入 causal Gate 4 的可信 hidden-state pair
```

---

# 7. Wiring-post 停止规则

从现在开始禁止继续：

- 调 3 mm threshold；
- 再做 deadband；
- 再发明 clearance proxy；
- 再修改 Chamfer threshold；
- 再扩大 proxy-only pair search；
- 再增加复杂 hidden descriptor 试图 rescue Wiring-post；
- 重新跑 512×101，只为了寻找同类 proxy pair；
- 启动 PB1 snapshot；
- 启动 StateDiff / CFPM 训练。

Wiring-post 主路线正式停止。

只有在未来出现**新的、独立、published-task-native interaction variable**时才允许重新考虑。

---

# 8. 下一主任务：DLO-Lab Wrapping

下一阶段优先：

```text
DLO-Lab Wrapping
```

原因：Wrapping 的 task semantics 本身包含 winding / wrapping structure，而不是依赖人工 clearance proxy。

优先 privileged variable：

\[
W_t=[w_1,w_2,w_3]
\]

即三个 post 的 signed winding / angular accumulation。

## 推荐阶段

### PB2-A — Official Wrapping reproduction

只确认：

- 官方环境正常；
- 官方 trajectory optimizer / policy replay 正常；
- reward / state 有限；
- 不改 physics；
- 不改 reward；
- 不改 geometry。

一旦 reproduction 成功立即停止 reproduction 工程。

### PB2-B — Task-native privileged-state audit

重点检查：

```text
winding quantity 是否直接来自 task semantics
是否连续
是否稳定
是否物理可解释
是否能够区分不同 routing state
```

优先复用 published code 中已有 winding computation。

不要先自己创造新的 binary proxy。

### PB2-C — Natural pair discovery（已完成）

正式 verdict：

```text
PB2C_NATURAL_WINDING_PAIRS_FOUND
```

冻结结果：

```text
128 total rollouts
123 official-valid full rollouts
5 invalid, all stretch failure
16,012 same-time winding-index-different pairs
5,534 final discovery candidates
```

PB2-C candidate selection 未使用 future divergence、force 或 sensor。rope 表示是 artificial partial-state discovery surrogate，不是 deployable sensing 结论。

停止 PB2-C collection / threshold / ranking 工程，进入 PB3。

### PB3 — Snapshot causal bifurcation（已阻断，冻结不变）

正式 blocked verdict：

```text
PB3_TARGETED_REPLAY_ALIGNMENT_FAILED
```

阻断发生在 future suffix 之前，因此：

```text
Gate 4 未运行
PAIR_METRICS 为空
PB3_TRAJECTORIES.npz 未生成
PB4 未启动
```

失败 branch：

```text
rollout 46
batch 1
env 14
seed 124
t = 13
```

冻结 alignment：

```text
rope max abs <= 5e-5 m
EE max abs <= 5e-5 m
motor qpos max abs <= 5e-5 rad
winding index exact
```

实际失败仅在 rope：

```text
rope max abs = 5.1826239e-5 m
EE / qpos pass
winding index exact
```

禁止事后放宽 `5e-5`。

已完成的 root-cause checks：

```text
seed / RNG state matched
position randomization matched
t0 rope exact
global translation 不是主要来源
build_state restore 只贡献较小局部误差
ordinary isolated collector replay 也出现约 5.17e-5 m 局部误差
```

误差集中在局部 rope vertices，而不是统一平移。

### PB3-R1 — Sequential acquisition-history replay（已完成）

正式 verdict：

```text
PB3R1_ACQUISITION_HISTORY_NOT_CONFIRMED_REPLAY_FLOOR_CALIBRATION_REQUIRED
```

结果：

```text
M0 sequential-history:
  3/3 pass original 50 um
  median coordinate RMSE ≈ 7.886e-6 m
  median rope max abs ≈ 4.667e-5 m

M1 isolated collector:
  3/3 pass original 50 um
  median coordinate RMSE ≈ 8.120e-6 m
  median rope max abs ≈ 4.724e-5 m

M2 build-state restore:
  1/1 pass
```

冻结 diagnostic criterion：

```text
M0/M1 coordinate-RMSE ratio <= 0.5
AND
M0/M1 rope-max-abs ratio <= 0.5
```

实际：

```text
coordinate-RMSE ratio = 0.971148
rope-max-abs ratio    = 0.988013
```

因此 acquisition-history dependence 未得到确认。

PB3-R1 没有修改：

```text
PB3 50 um alignment
formal shortlist
Gate 4
future suffix
```

停止继续研究 batch0 history。

### PB3-R2 — Independent replay-floor calibration（blocked; do not infer a floor）

PB3-R2 只测量 independent replay reconstruction floor。

Calibration states：

```text
20 official-valid PB2-C states
20 unique rollouts
0 overlap with formal PB3 20 rollouts

5 states per original batch
10 states at t13
10 states at t20
```

selection 只允许使用：

```text
official validity
batch/env/rollout metadata
```

禁止使用：

```text
state geometry
winding
candidate rank
future
reward
force
sensor
replay error
```

selection rule：

```text
per batch:
  remove formal PB3 rollouts
  sort eligible by env_index
  choose evenly-spaced order positions
  floor(j*(n-1)/4), j=0..4
```

calibration set 必须在 GPU floor measurement 前生成并 commit。

每个 state 3 个 fresh-process isolated replays。

通过 batching：

```text
4 batches × 3 repeats
= 12 fresh Genesis processes
```

每个 process 只跑到 t20。

必须分开测：

```text
A. fresh replay vs frozen PB2-C reconstruction error
B. fresh replay vs fresh replay repeat error
```

分别报告：

```text
median
P95
P99
max
```

以及原 50 um reference coverage。

PB3-R2 successful verdict：

```text
PB3R2_REPLAY_FLOOR_CALIBRATED
```

PB3-R2 明确禁止自动选择新 alignment threshold。

完成后停止并进入：

```text
PB3-R3
→ 使用 PB3-R2 independent calibration
→ 在重新运行 formal PB3 前
→ re-preregister alignment handling
```

PB3 formal causal quantities保持：

```text
10 formal pairs
1 mm absolute Gate-4 effect
5x repeat floor
>=3 passing pairs
>=2 winding strata
```

### PB4 — Control relevance

检查 hidden winding/contact mode 是否导致：

- 不同最佳动作；
- 不同动作 ranking；
- 明显 regret；
- closed-loop 策略需求不同。

如果 control relevance 不成立，即使 future diverges，也不进入模型主实验。

### PB5 — StateDiff / sensor / CFPM

只有 PB2–PB4 建立完整科学链之后：

```text
B0 StateDiff
B1 sensor-conditioned StateDiff
B2 StateDiff + CFPM
B3 StateDiff-FT + CFPM
B4 oracle
```

然后进入闭环控制实验。

---

# 9. Agent 执行优先级

Agent 每一步都应优先回答：

```text
这个工作会不会改变下一个科学决策？
```

如果答案是：

```text
不会
```

则通常不应花大量时间执行。

优先级：

```text
1. 科学机制成立
2. 当前实验能够运行
3. 数据可信
4. baseline 能跑
5. 模型训练
6. closed-loop
7. 必需的 evidence
8. 工程美化 / audit / defensive hardening
```

---

# 10. 强制工程守则

## Rule 1 — 不要过度在意安全性问题

研究服务器和内部实验环境中出现：

- 明文密码；
- 临时 token；
- 内部 IP；
- SSH 命令；
- 数据路径；
- 本地用户名；

**本身不能成为阻塞科学进展的理由。**

除非存在明确、现实、即时的风险，例如：

- 即将公开发布真实长期 credential；
- 即将执行不可逆破坏命令；
- 用户明确要求安全审查。

否则禁止自动扩展为：

```text
secret scanner
credential vault migration
复杂权限框架
安全审计阶段
```

安全问题不能压过研究主线。

## Rule 2 — 不要过度审计

Evidence 的目的：

> 让科学结论可以复查。

不是：

> 为每一个中间文件建立法证级审计链。

通常保留：

```text
RESULT.md
EVIDENCE.json
关键配置
关键 raw result
普通 Git commit
```

已经足够。

禁止：

- 为每一步建立重复 manifest；
- 多层 evidence tree；
- per-file provenance database；
- 反复重新验证已经明确通过的固定事实；
- 为审计本身增加新的审计。

## Rule 3 — 不要设置复杂执行合同

不要把简单实验写成：

```text
20 个 execution states
30 个 hard gates
resume1/resume2/resume3...
大量 state machine
```

除非任务本身确实需要。

执行合同只需要表达：

```text
输入是什么
运行什么
成功/失败如何判断
下一步是什么
```

如果一个脚本正常可以一次运行完成，就不要人为拆成复杂 state machine。

## Rule 4 — 禁止过度防御性编程

只处理现实中可能发生、且会影响实验可信度的问题，例如：

- required file missing；
- shape mismatch；
- NaN；
- simulator rollout failure；
- replay alignment failure；
- optimizer output missing。

不要为了理论上可能但实际基本不会发生的情况写大量分支。

## Rule 5 — 禁止反复防御基本不可能出现的 case

例如 pinned revision 下已经固定：

```text
数组 shape
字段名称
任务结构
文件布局
solver 类型
```

就不要持续写：

```text
if upstream future version changes...
if unknown alternative schema...
if impossible dtype appears...
if a field has 6 different hypothetical layouts...
```

研究代码不是通用 SDK。

**对固定 revision 编程。**

真正发生异常时再修。

## Rule 6 — 不要过度使用 SHA / SHA256

正常 Git provenance 通常只需要：

```text
main commit SHA
submodule gitlink SHA
```

足够。

禁止默认：

- 给全部数据逐文件 SHA256；
- 给全部 JSON / NPZ / checkpoint 建 hash database；
- 每一步生成 whole-tree hash；
- 为每个中间 tensor 建 checksum；
- 因 hash 不存在阻塞研究。

只有当：

- 数据不可变性本身就是实验核心；
- 明确怀疑文件被篡改；
- byte-exact reproducibility 是当前科学问题；

才增加额外 hash。

## Rule 7 — 科学可信前提下，最大化推进速度

Agent 应不断判断：

> 当前证据是否已经足够支持“继续”或“停止”？

一旦足够，就立刻推进。

不要为了：

```text
再确认一次
再做一个额外图
再多跑 5 个 diagnostic
再提高 audit coverage
```

拖延主流程。

必须避免：

```text
diagnostic perfection
>
research progress
```

## Rule 8 — 结论可信后，重心转向端到端可执行性

一旦：

```text
observable equivalence
hidden state
future bifurcation
control relevance
```

都得到可信证据，立即停止任务机制调试。

研究重心转为：

```text
data generation
→ baseline
→ StateDiff
→ sensor conditioning
→ CFPM
→ IDM
→ closed-loop
```

最终论文贡献必须落在完整系统表现，而不是几十个 task-audit scripts。

## Rule 9 — 定期清理不需要的文件

每完成一个阶段，执行一次轻量 cleanup。

删除：

- temporary clones；
- `.pytest_cache`；
- scratch JSON；
- one-off debug scripts；
- obsolete duplicated configs；
- 无用 video / plot；
- 中间测试数据；
- abandoned implementation；
- 已被新版替代且不再需要的 active instruction 文件。

保留：

```text
最终代码
最终 config
RESULT.md
EVIDENCE.json
关键 raw outputs
论文需要的图表/数据
```

不要让 repo 逐渐变成无法判断哪个文件仍然有效的 archive dump。

---

# 11. 禁止行为清单

Agent 在没有明确必要性的情况下不得主动做以下事情：

```text
❌ 新建安全扫描系统
❌ secret manager 工程
❌ 大规模 credential remediation
❌ 巨型 audit framework
❌ per-file SHA256 manifest
❌ 多层 provenance tree
❌ 复杂 execution state machine
❌ 为 pinned revision 写未来兼容层
❌ 为不可复现的 hypothetical case 写大量 fallback
❌ 无止境调 threshold
❌ 看到 FAIL 就降低 gate
❌ 看到 0 candidate 就直接宣告任务不存在 CCDA
❌ 用 future divergence 反过来定义 hidden state
❌ 用 deployment 不可用的 oracle 当正式传感输入
❌ task mechanism 未通过就开始大规模模型训练
❌ scientific gate 已通过后继续 task engineering
❌ 为“代码更完整”增加与下一个科学决策无关的功能
```

---

# 12. 阈值原则

所有科学 threshold 必须遵守：

## 可以

- 在实验前冻结；
- 来源于 deterministic repeat floor；
- 来源于 measurement resolution；
- 来源于 task semantics；
- sensitivity analysis；
- held-out validation。

## 不可以

```text
实验结果 FAIL
→ 把 threshold 调低
→ 宣布 PASS
```

也不可以：

```text
在同一批数据上找最有利 threshold
→ 再把它称作 pre-registered criterion
```

Diagnostic sensitivity analysis 可以做，但必须明确：

> diagnostic only。

---

# 13. Positive evidence 与 negative evidence

对于 approximate search：

```text
candidate > 0
```

可以作为 positive evidence。

但：

```text
candidate = 0
```

通常不能证明不存在。

尤其：

- kNN search；
- heuristic pair mining；
- approximate cross-time search；
- sampled horizon；
- limited rollout policy distribution。

必须区分：

```text
no candidate found
```

和：

```text
candidate does not exist
```

这是整个 CCDA 项目长期需要保持的推理纪律。

---

# 14. Pair mining 与正式 causal experiment 的边界

Pair miner 的任务只是：

> 找到值得做 causal branching 的候选。

它不能直接完成 Gate 4。

正式流程：

```text
pair discovery
        ↓
snapshot A / B
        ↓
same future action
        ↓
repeat baseline
        ↓
multi-horizon branch divergence
        ↓
control relevance
```

不能拿自然 rollout 中本来不同的 future action 当 same-action causal evidence。

---

# 15. Deployable sensor 原则

如果后续研究使用传感器：

优先：

- robot joint reaction；
- wrist F/T；
- tactile；
- deployable tracking residual；
- 现实机器人可部署的 proprioceptive / contact signal。

Oracle 信息只用于：

```text
hidden-state labeling
mechanism validation
upper bound
```

不能泄漏到部署模型输入。

并且：

> 没有物理 signal path，就不要盲目增加 sensor model capacity。

---

# 16. GPU 使用原则

GPU 有成本。

默认：

```text
CPU first
```

CPU 能完成的：

- 代码实现；
- unit test；
- pair analysis；
- config；
- offline statistics；
- report generation；
- repository work；

全部先做。

只有在：

- simulator 必须 GPU；
- model training；
- GPU-only reproduction；

确实需要时才启动 GPU。

启动后：

```text
一次解决明确科学问题
→ 输出结果
→ 立即停止
```

不要长时间让 GPU 做无决策价值的 diagnostic sweep。

---

# 17. 每阶段标准输出

每个正式 research phase 尽量只保持：

```text
config
main execution script
minimal tests
RESULT.md
EVIDENCE.json
necessary raw result
```

RESULT.md 应回答：

```text
做了什么？
关键数字是什么？
科学 verdict 是什么？
为什么？
下一步是什么？
```

EVIDENCE.json 保存机器可读数据。

不要把同一信息复制到 5 个不同报告。

---

# 18. Stop / Go 规则

## 任务机制尚未成立

```text
STOP MODEL TRAINING
```

继续修 task / benchmark selection。

## observable + hidden + future divergence + control relevance 都成立

```text
STOP TASK ENGINEERING
GO MODELING
```

Agent 不得继续花数周优化任务。

## 某条 task rescue 连续失败且 failure mechanism 已明确

```text
STOP RESCUE
SWITCH TASK
```

不要 sunk-cost。

Wiring-post 当前已经属于这一状态。

---

# 19. 当前立即下一步

当前不得继续 Wiring-post。

下一步：

```text
PB2-A
DLO-Lab Wrapping official reproduction
        ↓
PB2-B
task-native winding privileged-state audit
        ↓
PB2-C
natural observation-near / winding-different pair mining
        ↓
PB3
snapshot same-action multi-horizon vs repeat floor
        ↓
PB4
control relevance
        ↓
B0/B1/B2/B3/B4
StateDiff / sensor / CFPM
        ↓
closed-loop
```

Agent 的首要任务不是再讨论 Wiring-post，而是尽快确认：

> Wrapping 的 task-native winding state 是否可以成为 CCDA 所需的可信 privileged interaction variable。

---

# 20. Agent 自检问题

每开始一个新工作前，只问以下问题：

1. **它解决当前哪个科学问题？**
2. **结果会改变下一步决策吗？**
3. **有没有更简单的方法得到同样结论？**
4. **是否在重复已经完成的 audit？**
5. **是否在为基本不会发生的 case 写代码？**
6. **是否正在因为工程完美主义阻塞研究？**
7. **如果当前结果通过，能否立刻推进下一阶段？**
8. **如果当前结果失败，是否已经足够决定换路线？**
9. **这个文件/脚本完成阶段后是否还需要保留？**

如果第 2 条答案是“不改变决策”，通常应删除或取消该工作。

---

# 21. 最终原则

本项目的研究执行应始终保持：

```text
科学性
    >
结论可信
    >
端到端可执行
    >
工程简洁
    >
审计完整度
    >
防御性完美
```

目标不是建设一个“最安全、最复杂、最可审计”的研究仓库。

目标是：

> **用最少但足够可信的工程和证据，尽快证明或否定 CCDA 机制，然后完成 StateDiff + sensing + CFPM + IDM + closed-loop 的端到端研究。**


### PB3-R2 REV1 implementation invariants

PB3-R2 execution additionally requires:

```text
t0 precondition inside each worker
before first future command

all 60 target samples explicitly finite

50 um coverage = rope-only max-abs coverage
```

`t0` uses the historical full PB3 alignment identity predicate, but the reported 50 um coverage must use only:

```text
rope_max_abs_coordinate_m <= 5e-5 m
```

These are implementation correctness conditions, not new scientific gates.
