---
title: "CCDA Research Roadmap"
subtitle: "柔性物体操作中的接触条件形变分支歧义：长期研究路线与 Agent 防偏航合同"
version: "1.0"
status: "Active"
primary_environment: "DeformableRavens"
primary_claim: "在视觉与机器人历史近似相同、未来动作干预相同的条件下，视觉潜在但可由接触信息推断的接触状态会导致不同柔性形变未来；接触信息应帮助扩散模型选择正确未来分支，并且该未来必须可转化为可执行动作。"
---

# 0. 文件用途

本文件是 CCDA 项目的长期路线、实验合同和 Agent 执行约束。

Agent 的目标不是“尽快训练一个模型”，而是按顺序建立以下证据链：

1. **CCDA 现象真实存在；**
2. **该现象不是视觉泄漏、动作差异、初始化漂移或模拟器伪影；**
3. **接触信息确实能降低未来分支不确定性；**
4. **扩散去噪阶段的接触引导优于简单拼接或动作空间引导；**
5. **预测未来能够转化为可执行动作；**
6. **方法在闭环任务中改善成功率并具有跨条件泛化能力。**

任何阶段的证据失败，都必须停止后续阶段，先修复根因。

---

# 1. 核心科学命题

## 1.1 CCDA 的研究对象

CCDA：**Contact-Conditioned Deformation Branch Ambiguity**，即“接触条件导致的柔性形变未来分支歧义”。

研究场景满足：

- 当前视觉形状相同或近似；
- 机器人本体状态相同或近似；
- 过去动作历史相同或近似；
- 隐藏接触条件不同；
- 接触条件对视觉是潜在的，但可通过触觉、力、力矩、张力、滑移或接触冲量推断；
- 在相同未来动作干预下，柔性物体进入不同未来形变分支；
- 分支差异进一步影响任务结果、安全性或成功率。

## 1.2 必须使用的因果形式

未来预测模型的研究对象必须显式包含候选未来动作：

\[
p_\theta\left(
Y_{t+1:t+H}
\mid
h_t^{obs},
h_t^{robot},
h_t^{action},
u_{t:t+H-1},
h_t^{contact}
\right)
\]

其中：

- \(h_t^{obs}\)：视觉或物体状态历史；
- \(h_t^{robot}\)：机器人本体状态历史；
- \(h_t^{action}\)：过去动作历史；
- \(u_{t:t+H-1}\)：固定的未来动作干预；
- \(h_t^{contact}\)：接触相关观测历史；
- \(Y_{t+1:t+H}\)：未来柔性物体形变轨迹。

如果未来动作没有固定，则只能证明“相似历史对应多模态未来”，不能证明“隐藏接触在相同动作下导致未来分叉”。

## 1.3 可解性边界

必须区分两类隐藏变量：

### A. 视觉潜在但接触可观测

视觉无法区分，但触觉或力学信号可以区分。

这是 CCDA 的正式研究对象。

### B. 对所有输入均不可观测

如果视觉、机器人状态、动作历史和接触信号都完全相同，则模型不可能提前确定唯一真实未来，只能输出多模态后验。

不得要求模型解决信息论上不可解的问题。

---

# 2. 论文叙事

## 2.1 正式叙事

现有方法分别研究：

- 未来状态扩散与逆动力学；
- 触觉约束下的动作空间去噪引导；
- 接触丰富操作中的多模态策略；
- 柔性物体的动力学预测。

本项目不以“把两个已有模块拼接起来”为主要贡献，而以以下命题为核心：

> 在视觉历史和动作干预近似相同的条件下，不同的视觉潜在接触状态会使柔性物体进入不同的未来形变分支。CCDA 对这一现象进行操作性定义，构造配对反事实基准，并研究接触信息能否在未来状态扩散去噪过程中恢复正确分支，同时通过独立的可执行性验证排除物理看似合理但机器人无法实现的未来。

## 2.2 预期贡献

论文贡献应至少包含：

1. **CCDA 的操作性定义；**
2. **严格配对的反事实基准；**
3. **面向未来状态分支的接触条件去噪引导；**
4. **Hard branch negatives，而不是普通跨时间负样本；**
5. **未来轨迹到动作的可执行性验证；**
6. **从1D绳索到2D布料的泛化；**
7. **无CCDA刚体任务上的负对照。**

## 2.3 不得声称的内容

除非有充分文献检索和实验证据，不得声称：

- 首次将触觉用于扩散模型；
- 首次将接触信息用于未来状态预测；
- 首次结合状态扩散和逆动力学；
- 首次进行柔性物体接触条件控制；
- 单纯增加接触输入即可证明 CCDA；
- 预测轨迹视觉合理即代表机器人可执行。

---

# 3. 现有论文中可继承与不可直接继承的部分

## 3.1 状态扩散与逆动力学论文

可继承：

- 将未来状态预测与动作生成解耦；
- 使用扩散模型生成未来状态；
- 使用逆动力学模型将状态轨迹转换为动作；
- 使用 Push-L 作为刚体、多阶段、接触丰富负对照；
- 比较带/不带 IDM 的性能。

不可直接继承：

- 原始未来状态模型未必显式条件化于固定未来动作；
- 原始 IDM 通过动作 MSE 训练，不能独立证明预测未来可达；
- 原始仿真任务以刚体为主，不能直接验证 CCDA。

## 3.2 TouchGuide

可继承：

- 在去噪后期使用接触条件进行引导；
- 使用接触物理兼容性评分；
- 扫描 guidance scale 与 guidance steps；
- 使用 noise-aware 训练使评分器适配扩散中间态；
- 比较触觉图像与力场等不同接触模态；
- 保留 task-specific guidance 的局限性分析。

不可直接继承：

- TouchGuide 的评分对象是候选动作，而本项目的核心评分对象应是候选未来形变轨迹；
- 普通跨时间负样本不足以验证分支识别；
- TouchGuide 的五个任务主要是实体机器人任务，不能直接作为本项目主仿真环境；
- 动作空间引导只能作为基线，不能作为 CCDA 的核心贡献。

---

# 4. 仿真环境路线

## 4.1 主环境：DeformableRavens

**决策：继续以 DeformableRavens 为主，不在核心证据建立前迁移主环境。**

理由：

- 已有 cable 与 fabric 柔性物体任务；
- 绳索有有序粒子/珠子表示；
- 适合构造反事实状态快照；
- 已有示范生成和任务执行基础；
- 当前项目已在该环境建立数据、状态、动作和隐藏接触基础设施；
- 迁移成本高，且不会自动提高科学有效性。

## 4.2 扩展环境

### SoftGym

用途：

- 跨模拟器 rope/cloth 泛化；
- 排除结论依赖 DeformableRavens 特定实现。

进入条件：

- 主环境至少一个 Cable 任务完成闭环显著增益；
- Fabric 任务通过数据门禁；
- 核心模型和评价协议冻结。

### 高保真触觉/FEM环境

用途：

- 接触力场与形变更真实；
- 验证简化接触信号是否仍有效；
- 为未来实体机器人实验做准备。

进入条件：

- 主论文主张已经在轻量环境中成立；
- 资源、接口和复现成本可控。

## 4.3 负对照环境

Push-L 保留为无柔性隐藏接触负对照。

预期：

- State Diffusion + IDM 可有效；
- CCDA 接触模块不应产生稳定显著增益；
- 错误或随机接触信号不应系统性改变结果。

---

# 5. 任务路线

## Task 0：Push-L Negative Control

### 目的

验证 CCDA 模块不会在不存在隐藏柔性接触分支时凭借额外参数或梯度引导获得虚假增益。

### 成功标准

- 基础方法能复现合理性能；
- CCDA 模块增益不显著或接近零；
- shuffled contact 不影响结果；
- 不出现接触模块导致的系统性退化。

---

## Task 1：Hidden-Friction Cable Pull

### 地位

最小、主要、第一优先级 CCDA 任务。

### 条件

- `free / low_friction`
- `hidden_high_friction`

### 设计

- 绳索初始可见几何相同；
- 局部段处于不可见摩擦区；
- 使用相同的微小预加载动作；
- 预加载尽量不产生明显可见形变；
- 接触信号出现可识别差异；
- 随后回放完全相同的主拉动动作。

### 预期分支

- 低摩擦：整体滑动、接近目标线；
- 高摩擦：局部停滞、弯折、拱起或旋转；
- 两分支需要不同后续策略。

### 必须避免

- 通过颜色、粒子编号、隐藏几何渲染泄漏条件；
- 不执行动作时就发生明显漂移；
- 接触差异只有未来主动作执行后才出现；
- 摩擦参数同时改变其他可见动力学变量。

---

## Task 2：Hidden-Hook Cable Routing

### 目的

验证方法能处理接触拓扑变化，而不只是连续摩擦变化。

### 条件

- 自由状态；
- 绳索局部穿过隐藏导环、挂钩或狭槽。

### 设计

- 当前俯视图近似一致；
- 相同预加载动作产生不同张力方向或接触冲量；
- 相同拉动动作产生不同拓扑未来。

### 预期分支

- 自由：整体平移；
- 隐藏卡接：绕钩旋转、环收紧、端点反向移动或局部拉伸。

### 禁止实现

不得使用不符合真实接触的强制世界坐标绑定作为正式任务。

如需使用简化约束，必须先证明：

- 无动作稳定；
- 条件不泄漏；
- 力学响应合理；
- 结果对约束参数连续变化；
- 不依赖单一脚本特例。

### 当前 Stage M0 状态（Phase 0H，2026-08-05）

- **Hidden-Friction：BLOCKED。** 现有证据不足以建立有效 CCDA 环境，不得进入模型训练。
- **Hidden-Hook bootstrap：BLOCKED。** Phase 0G 已证明精确运动和 fixed-step 确定性工程链路可用，但隐藏挂钩未通过完整可观测性与结果分支门禁。
- Phase 0H 仅审计单一不可见 recessed U-hook，固定种子 `71001–71003`，不进行几何搜索或训练。
- Phase 0H 的精确 probe 阶段全部达到未放宽的 Cartesian 容差；关节 timeout 作为可恢复事件保留在 `motion_debug.json`，不得静默删除。
- 原始正式传感信号包括 joint motor torque、joint reaction force/torque、EE constraint reaction、grasp state 与 probe phase index；Oracle hook contact 仅用于时间对齐审计。
- 当前新目标是在任何模型开发之前，先建立一个通过全部 Stage M0 数据门禁的有效 CCDA 环境。
- Phase 0G 失败证据保留在 `reports/experiment2/phase0_hidden_hook/phase0g/`；Phase 0H 证据保留在 `reports/experiment2/phase0_hidden_hook/phase0h/`，不得用新报告覆盖旧失败结论。

### Phase 0I — Formal Sensor Reanalysis and Hidden Z-Latch Smoke

- Phase 0H 的 joint reaction force/torque 已接通并保留在 raw trace，但原 formal classifier 复用了未包含该通道的 legacy preload feature。
- Phase 0I 使用固定的 49 维 robot-event-aligned formal feature 离线重算 Phase 0H：legacy、corrected 和 reaction-only accuracy 均为 `0.333333`。该 post-hoc diagnostic 不改变 Phase 0H 的 BLOCKED verdict。
- Phase 0I 只运行单一 `delayed_z_latch_v1` topology 和 seeds `71001–71003`，未进行参数网格或模型训练。
- Result: `HIDDEN_Z_LATCH_SMOKE_BLOCKED`。
- Main metrics：RGB raw `0.5`、RGB delta `0.666667`、formal sensor `0.833333`、Oracle `1.0`；preload visible difference `0.003603 m`、ADE `0.022073 m`、FDE `0.039645 m`、branch amplification `5.001043`、progress gap `0.002585 m`、engagement `0.743494`。
- 现有门禁中仅 `outcome_progress_gap` 未通过（目标 `>=0.01 m`）；因此这仍只是固定 3-seed smoke 的阶段性证据，不是 Scientific PASS。
- Model training remains blocked unless a later expanded Stage M0 audit passes.

### Phase 0J — Outcome Localization and Fixed Wide-Stop Z-Latch Smoke

- Phase 0I established low-leakage, sensor-observable future deformation branches but failed the unchanged task-level mean progress-gap gate.
- Phase 0J preserved the Phase 0I probe, main action, 49D formal sensor feature, thresholds and seeds. The only physical change was stop-wall tangent width: `0.032 m -> 0.080 m`.
- No geometry grid search, action search or model training was run. Outcome decomposition was diagnostic only and did not replace the official `mean_cable_progress_gap >= 0.01 m` gate.
- Phase 0I offline outcome medians were: all `0.002585 m`, blocked `0.001884 m`, pulled-side `-0.000832 m`, trailing-side `0.004282 m`; motion-difference RMS normal/tangent/Z was `0.050417 / 0.008770 / 0.004611 m`.
- Verdict: `HIDDEN_WIDE_STOP_SMOKE_BLOCKED`.
- Phase 0J metrics: RGB raw `0.5`, RGB delta `0.833333`, formal sensor `0.833333`, Oracle `1.0`; preload `0.004502 m`, ADE `0.004668 m`, FDE `0.007658 m`, branch amplification `1.700896`, official progress gap `0.003728 m`, engagement `0.740909`.
- Failed checks: `preload_visibility`, `main_ade`, `main_fde`, `branch_amplification`, `vision_screen`, `sensor_over_vision_margin`, `outcome_progress_gap`, `fde_seed_fraction`, and `amplification_seed_fraction`.
- Wide-stop tangent projection covered all beads for seeds `71001/71002` and the pulled endpoint for seed `71003`; blocked-segment median was `0.003728 m`, pulled-side was unavailable, and the only non-empty trailing-side diagnostic was `0.001976 m`. Missing empty-segment statistics were not invented and were not treated as gates.
- Training remains blocked; this failed fixed-topology smoke does not authorize expanded validation, new geometry, new actions or training.

### Phase 0K — Fixed Same-End Tension Extension

- Restored the Phase 0I `delayed_z_latch_v1` geometry with `wall_width=0.032 m`; the Phase 0I probe, observation, 49D formal sensor feature, seeds, execution settings and scientific thresholds were unchanged.
- Replaced the single `0.080 m` main pull with one continuous-grasp, collinear `0.080 m + 0.040 m` intervention. No re-grasp, geometry search, action search, extension search or model training was performed.
- Stage-1 and stage-2 tension metrics were diagnostic only. The official gate remained final median `mean_cable_progress_gap >= 0.01 m`.
- Verdict: `HIDDEN_TENSION_EXTENSION_SMOKE_BLOCKED`.
- Main metrics: RGB raw `0.5`, RGB delta `0.666667`, formal sensor `0.833333`, Oracle `1.0`, sensor-over-vision margin `0.166667`; preload `0.003816 m`, ADE `0.014014 m`, FDE `0.022885 m`, branch amplification `5.997040`, engagement `0.691617`.
- Tension diagnostics: stage-1 gap `0.001557 m`, stage-2 gap `-0.009300 m`, final official gap `-0.009326 m`, median stage-2-minus-stage-1 change `-0.000805 m`; stage-1/stage-2 branch distance `0.021347 / 0.022846 m`.
- All Cartesian stages and free/hidden grasp-retention checks passed for all three seeds. The only failed formal check was `outcome_progress_gap`.
- The fixed continuous-grasp tension extension did not convert the Phase 0I local deformation branch into a sufficient and stable final task-level mean progress gap. The hidden Z-latch environment family remains blocked for training.

---

## Task 3：Hidden-Clamp Fabric Fold

### 目的

证明 CCDA 不限于1D珠链，可推广到2D柔性表面。

### 条件

- 自由角点；
- 隐藏弱夹持；
- 隐藏高摩擦；
- 可释放黏附。

### 设计

- 初始布料表面视觉近似一致；
- 使用相同双臂预加载；
- 使用相同双臂抬起或折叠动作；
- 接触条件导致不同折叠方向、局部拉伸或掉落结果。

### 评价

- 折叠方向；
- 角点对应关系；
- 覆盖率；
- 褶皱结构；
- 峰值拉伸；
- 是否掉落；
- 最终任务成功率。

---

# 6. 数据集合同

必须建立两个互相独立的数据集。

## 6.1 分支识别反事实数据集

用途：

- 证明 CCDA 现象；
- 训练未来分支模型；
- 评价接触信息是否帮助分支恢复。

每个基础快照组执行：

1. 保存完全相同的模拟器状态；
2. 复制到多个隐藏接触条件；
3. 回放相同过去动作；
4. 回放相同预加载动作；
5. 回放相同未来干预动作；
6. 记录不同未来轨迹；
7. 记录接触观测与特权真值；
8. 记录任务结果。

样本结构：

\[
(
group\_id,
h^{obs},
h^{robot},
h^{action},
h^{contact},
c^{priv},
u,
Y,
outcome
)
\]

约束：

- `group_id` 相同的条件必须进入同一数据划分；
- `c_priv` 只能用于审计和 Oracle 上界；
- 正式模型不得读取隐藏条件标签；
- 未来动作必须逐元素一致；
- 快照恢复必须可重复。

## 6.2 策略示范数据集

用途：

- 训练不同接触分支下的自适应动作；
- 评价闭环任务成功率。

要求：

- 每个隐藏条件有对应专家动作；
- 与固定动作反事实数据分开；
- 不得用策略差异反向证明物理分支存在；
- 必须记录失败示范或自动生成失败候选用于可执行性判别。

---

# 7. 数据有效性门禁

模型训练前，所有任务必须通过以下门禁。

## Gate A：Observable Match

两条件之间：

- 视觉历史距离低于阈值；
- 无接触本体状态距离低于阈值；
- 过去动作历史距离低于阈值。

未通过：

- 检查视觉泄漏；
- 检查初始化不一致；
- 检查渲染、粒子排序、速度状态或接触几何外露。

## Gate B：Future Action Match

未来干预动作必须相同。

未通过：

- 数据无效；
- 禁止进入 CCDA 统计；
- 禁止解释为接触导致分支。

## Gate C：No-Action Stability

不执行动作时：

- 物体不能系统性漂移；
- 不同条件的可见状态不能自行分离；
- 接触条件不能通过静态形变被轻易识别。

未通过：

- 首先修复模拟器任务；
- 禁止训练模型掩盖任务缺陷。

## Gate D：Contact Informativeness

在预加载后：

- 接触历史应能显著区分条件；
- 视觉或普通本体状态仍不应轻易区分条件。

预期：

\[
Acc(contact) \gg Acc(vision)
\]

如果视觉分类器显著高于随机：

- 首先按泄漏处理；
- 不能直接解释为视觉发现了真实隐变量。

## Gate E：Future Divergence

相同未来动作下：

- 未来轨迹距离达到预设效应量；
- 分支在多个随机种子和快照上稳定存在；
- 不是单步噪声或粒子排序错误。

## Gate F：Outcome Impact

至少部分 CCDA 样本应导致：

- 成功/失败差异；
- 目标误差差异；
- 安全风险差异；
- 掉落、滑移、过度拉伸或接触力差异。

说明：

“Outcome-critical CCDA”可作为强子集，不必强制所有分支都导致不同最终成功率。

## Gate G：Leakage Audit

必须检查：

- 颜色；
- 粒子编号；
- 接触几何是否被渲染；
- 相机参数；
- 仿真内部状态；
- 速度与加速度；
- 时间戳；
- 文件顺序；
- 条件专属随机种子；
- 数据归一化统计；
- 训练/测试快照重复。

## Gate H：Determinism

相同快照、条件、动作和随机种子必须：

- 状态恢复一致；
- 接触信号一致；
- 未来轨迹一致或误差在容许范围内；
- CPU/GPU和多worker差异被记录。

---

# 8. 模型路线

## Stage M0：无模型物理审计

目标：

- 证明任务本身成立。

交付物：

- 反事实配对可视化；
- 接触信号曲线；
- 未来分支距离；
- 无动作稳定性；
- 条件泄漏分类器；
- outcome impact；
- 数据合同哈希。

停止条件：

任一数据门禁失败。

当前状态：**BLOCKED（Phase 0H）**。Hidden-Friction 与 Hidden-Hook 均未建立可供后续模型训练使用的有效 CCDA 环境；Stage M1 及以后阶段保持禁止。

---

## Stage M1：基础未来状态扩散

模型：

\[
p_\theta(Y \mid h^{obs}, h^{robot}, h^{action}, u)
\]

目标：

- 学习不带接触信息的多分支未来；
- 建立 CCDA 下的错误分支基线；
- 验证未来动作条件的必要性。

必须消融：

- 有/无未来动作条件；
- 直接单步动力学；
- 确定性回归；
- diffusion；
- 不同预测时域。

停止条件：

- 无接触基础模型无法覆盖真实分支；
- 未来状态表示本身不稳定；
- 分支指标无法可靠计算。

---

## Stage M2：Contact Concatenation

模型：

\[
p_\theta(Y \mid h^{obs}, h^{robot}, h^{action}, u, h^{contact})
\]

目标：

- 建立最直接的接触条件基线；
- 判断训练期特征融合是否已足够。

该方法是基线，不是默认主要贡献。

---

## Stage M3：CCDA Contact-Physical Steering

核心：

- 基础扩散模型先生成视觉和任务合理的粗未来；
- 去噪后期使用接触条件未来兼容性评分引导；
- 评分对象是候选未来轨迹，而不是候选动作。

评分器：

\[
g_\phi(
Y^k,
h^{obs},
h^{robot},
h^{action},
u,
h^{contact}
)
\]

训练样本：

- 正样本：同组、同条件、真实未来；
- Hard negative：同组、相同视觉历史、相同动作、不同接触条件的真实未来；
- 额外负样本：错误拓扑、过度拉伸、不可达形变、时间错配未来；
- 噪声增强：使用与扩散中间态一致的噪声调度。

必须扫描：

- guidance scale；
- guidance start/end step；
- 引导步数；
- 不同接触模态；
- 不同负样本构造；
- classifier guidance 与 energy guidance。

目标：

- 提高正确分支概率质量；
- 降低错误分支；
- 不应只是降低总体多样性。

---

## Stage M4：逆动力学模型

输入：

\[
a_t = f^{-1}(
h_t^{robot},
h_t^{action},
Y_{t+1:t+H}
)
\]

目标：

- 将未来状态转换为候选动作；
- 与第一篇论文框架保持可比。

限制：

- 动作 MSE 低不代表未来可达；
- 多个动作可能对应同一状态；
- MSE 可能输出平均动作；
- IDM 不应读取相机参数、条件标签或其他非动作因果变量。

必须审计：

- 动作维度和语义；
- 近零方差维度；
- 相机、时间或状态泄漏；
- 控制模式；
- 动作限幅；
- 训练与部署归一化一致性。

---

## Stage M5：可执行性门

必须建立独立的候选未来可执行性验证。

最低形式：

\[
f_{forward}(s_t,\hat a_t,h^{contact})
\approx
\hat s_{t+1}
\]

可选实现：

- 已知模拟器一步回放；
- 学习前向动力学；
- IDM循环一致性；
- reachability classifier；
- trajectory-level action feasibility discriminator。

输出：

- 接受候选；
- 拒绝候选；
- 重新采样；
- 调整 guidance；
- 选择下一分支。

目标：

- 拒绝视觉合理但动作不可实现的未来；
- 降低不可达未来执行失败；
- 分离未来预测误差与动作解码误差。

---

## Stage M6：闭环策略

闭环流程：

1. 获取视觉、机器人状态和接触历史；
2. 生成多个未来候选；
3. 通过接触引导提高真实分支权重；
4. 通过可执行性门过滤；
5. 使用 IDM 解码动作；
6. 执行短动作块；
7. 重新观测并滚动规划。

禁止：

- 一次生成整段轨迹后完全开环执行；
- 只报告离线预测，不报告闭环结果；
- 用 Oracle 接触标签替代真实接触观测。

---

# 9. 基线与消融矩阵

必须至少包含：

1. State Diffusion + IDM；
2. State Diffusion，无 IDM；
3. Deterministic Future Regression + IDM；
4. Contact Concatenation；
5. Contact-Conditioned Diffusion；
6. CCDA Contact-Physical Steering；
7. TouchGuide-style Action-Space Guidance；
8. CCDA Steering + IDM；
9. CCDA Steering + IDM + Feasibility Gate；
10. Oracle Contact Label；
11. Shuffled Contact；
12. Zero Contact；
13. Delayed Contact；
14. No Future-Action Conditioning；
15. Oracle Future + IDM；
16. Ground-Truth Action + Forward Rollout；
17. Random or Wrong Contact Condition；
18. Early-only / Late-only / Full denoising guidance；
19. Ordinary temporal negatives / Hard branch negatives；
20. Single sample / Best-of-K / guided Best-of-K。

---

# 10. 评价指标

## 10.1 数据与现象指标

- 视觉历史匹配距离；
- 机器人状态匹配距离；
- 过去动作匹配误差；
- 未来动作逐元素匹配；
- 无动作漂移；
- 接触条件可分性；
- 视觉条件可分性；
- 固定动作未来分支距离；
- outcome impact；
- 配对效应量；
- 95%置信区间。

## 10.2 未来预测指标

不得只报告平均 MSE。

必须包括：

- Correct Branch Probability Mass；
- Branch Top-1 Accuracy；
- Branch NLL；
- Best-of-K ADE；
- Best-of-K FDE；
- 分支覆盖率；
- 分支校准误差；
- 错误分支率；
- 多样性；
- 条件交换后的性能下降；
- shuffled-contact sensitivity。

几何指标：

- 有序点 ADE/FDE；
- Chamfer Distance；
- 曲率误差；
- 弧长误差；
- 端点误差；
- 折叠拓扑或角点对应；
- 覆盖率；
- 峰值应变；
- 接触区域误差。

## 10.3 动作与可执行性指标

- IDM动作误差；
- 前向回放一致性；
- 可执行候选比例；
- 不可达未来拒绝率；
- 错误拒绝率；
- 动作限幅违规；
- 接触力违规；
- 预测正确但执行失败比例；
- 预测错误但偶然成功比例。

## 10.4 闭环指标

- 总成功率；
- 各接触条件成功率；
- 最坏条件成功率；
- 到达目标时间；
- 重规划次数；
- 重新抓取次数；
- 掉落率；
- 滑移率；
- 峰值拉伸；
- 峰值接触力；
- 恢复失败率；
- 任务级置信区间。

## 10.5 泛化指标

至少设置：

- ID：新初始化；
- Contact-OOD：未见摩擦或夹持强度；
- Location-OOD：未见接触位置；
- Geometry-OOD：新绳长、材料、初始形状；
- Action-OOD：未见动作幅度或方向；
- Sensor-OOD：不同接触噪声与延迟；
- Task-OOD：Cable-Line → Cable-Routing → Fabric；
- Simulator-OOD：DeformableRavens → 第二模拟器。

---

# 11. 数据划分与统计合同

## 11.1 数据划分

必须按 `group_id` 划分。

同一基础快照的：

- free；
- hidden friction；
- hidden hook；
- 其他接触条件；

不得分散到训练、验证和测试中。

## 11.2 随机种子

- 至少3个训练种子；
- 评估快照固定；
- 所有方法使用同一评估集合；
- 所有随机数来源记录；
- 模拟器种子、模型种子和数据加载种子分开记录。

## 11.3 统计报告

必须报告：

- 均值；
- 标准差或标准误；
- 95%置信区间；
- 配对效应量；
- 条件级结果；
- 失败案例数量；
- 不是只报告最佳 checkpoint。

## 11.4 最低规模建议

核心任务：

- 每个任务约500–1000个基础配对快照组；
- 每组至少2个隐藏条件；
- 每个方法、每个条件、每个种子至少50个闭环评估快照；
- 超参数扫描必须使用独立验证集；
- 最终测试集只使用一次或严格受控使用。

---

# 12. Agent 防偏航合同

## 12.1 执行原则

Agent 每次开始工作必须：

1. 读取本文件；
2. 明确当前 Stage；
3. 明确当前 Gate；
4. 读取最近一次证据报告；
5. 检查仓库、数据、配置与依赖哈希；
6. 只执行当前 Gate 所允许的工作；
7. 失败时停止，不得跳过；
8. 输出可复现证据；
9. 更新阶段状态；
10. 不得自行改变核心科学命题。

## 12.2 不可违反的约束

### 科学约束

- 不得把普通多模态未来称为 CCDA；
- 不得在未来动作不一致时声称接触导致分支；
- 不得使用特权隐藏标签作为正式模型输入；
- 不得用视觉泄漏代替接触推断；
- 不得把模拟器伪影解释为物理规律；
- 不得把低MSE等同于正确分支；
- 不得把IDM输出等同于可执行性；
- 不得只看成功率而忽略分支预测；
- 不得只看离线预测而跳过闭环；
- 不得只报告单一任务或单一随机种子。

### 工程约束

- 不得在 Gate 失败时继续正式训练；
- 不得修改冻结数据后继续复用旧报告；
- 不得在未记录的情况下改变动作语义；
- 不得混用位置、速度或增量控制；
- 不得混用不同归一化统计；
- 不得在训练/测试之间共享快照组；
- 不得用测试集调超参数；
- 不得删除失败证据；
- 不得静默修复后覆盖原始失败报告；
- 不得仅提交代码而不提交运行证据。

### 路线约束

- 核心 Cable 任务未成立前，不迁移主模拟器；
- Cable 任务未通过闭环前，不优先扩展到复杂布料；
- 数据门禁未通过前，不优化网络规模；
- 基线未复现前，不宣称新方法有效；
- 接触拼接基线未完成前，不宣称去噪引导必要；
- 可执行性门未完成前，不宣称“预测未来可用于控制”；
- 负对照未完成前，不排除额外模块带来的伪增益。

## 12.3 必须停止的情况

出现以下任一情况，Agent 必须停止并报告：

- 数据快照无法确定性恢复；
- 未来动作不一致；
- 无动作条件出现系统性漂移；
- 视觉分类器能够轻易识别隐藏条件；
- 接触信号无法区分条件；
- 未来分支效应不稳定；
- 指标实现与物理直觉冲突；
- 训练输入包含条件标签泄漏；
- IDM动作维度或控制语义不明确；
- 正向回放无法复现预测状态；
- 配置、数据或模型哈希不一致；
- 测试集被用于调参；
- 结果无法由独立脚本复算。

## 12.4 失败输出格式

每次失败必须输出：

```yaml
stage:
gate:
verdict: BLOCKED
root_cause:
evidence:
affected_artifacts:
invalidated_results:
required_fix:
next_allowed_action:
forbidden_actions:
```

## 12.5 通过输出格式

每次通过必须输出：

```yaml
stage:
gate:
verdict: PASS
implementation_commit:
evidence_commit:
dataset_hash:
config_hash:
model_hash:
test_summary:
scientific_summary:
remaining_risks:
next_stage:
```

---

# 13. 证据链与仓库合同

每个阶段必须至少保存：

- `contract.yaml`
- `config.yaml`
- `dataset_manifest.json`
- `snapshot_manifest.json`
- `metrics.json`
- `summary.md`
- `failure_cases/`
- `plots/`
- `checkpoints/`
- `reproduce.sh`
- `environment.txt`
- `git_state.txt`
- `hashes.json`

## 13.1 冻结规则

以下任一项变化，旧结果自动失效：

- 数据快照；
- 任务物理参数；
- 动作定义；
- 状态定义；
- 接触观测定义；
- 归一化统计；
- 数据划分；
- 模型结构；
- 训练损失；
- 评价指标；
- 模拟器版本；
- 子模块提交；
- 随机种子策略。

## 13.2 报告分层

必须区分：

- **Implementation PASS**：代码和测试正确；
- **Audit PASS**：数据、哈希、复现和指标正确；
- **Scientific PASS**：核心科学门槛达到；
- **Closed-loop PASS**：真实策略性能改善；
- **Generalization PASS**：跨条件、跨任务或跨模拟器成立。

工程测试通过不能替代科学结论。

---

# 14. 推荐阶段顺序

## Phase 0：任务与数据审计

- Hidden-Friction Cable Pull；
- 配对快照；
- 无动作稳定；
- 视觉泄漏；
- 接触可分；
- 固定动作未来分支；
- outcome impact。

**Exit：所有数据门禁 PASS。**

## Phase 1：基础未来分布

- 动作条件未来扩散；
- 无接触多分支建模；
- Branch metrics；
- Best-of-K；
- No future-action conditioning 消融。

**Exit：基础模型能覆盖真实分支，但在 CCDA 条件下分支选择不可靠。**

## Phase 2：接触条件建模

- Contact concatenation；
- Contact-conditioned diffusion；
- Shuffled contact；
- Oracle contact；
- 接触噪声与延迟。

**Exit：接触信息提供稳定增益，且不是泄漏。**

## Phase 3：CCDA 去噪引导

- Future compatibility scorer；
- Hard branch negatives；
- Noise-aware training；
- Guidance scale/steps；
- Late vs early guidance；
- 动作空间引导基线。

**Exit：CCDA 引导在正确分支指标上显著优于直接拼接和动作引导。**

## Phase 4：IDM 与可执行性

- IDM；
- Oracle future + IDM；
- Forward rollout；
- Feasibility gate；
- 不可达未来拒绝。

**Exit：预测未来可稳定转化为动作，且可执行性门有效。**

## Phase 5：闭环控制

- 短动作块执行；
- 滚动规划；
- 成功率；
- 失败恢复；
- 安全指标。

**Exit：核心 Cable 任务闭环显著改善。**

## Phase 6：任务扩展

- Hidden-Hook Cable Routing；
- Hidden-Clamp Fabric Fold；
- Task-OOD；
- Geometry-OOD；
- Contact-OOD。

**Exit：方法不依赖单一摩擦任务。**

## Phase 7：负对照与跨模拟器

- Push-L 负对照；
- 第二模拟器验证；
- 接触模态变化；
- 传感噪声与延迟。

**Exit：排除模块伪增益，并证明一定外部有效性。**

## Phase 8：论文冻结

- 主张与证据一一映射；
- 完整消融；
- 失败案例；
- 可复现实验；
- 代码和数据合同；
- 图表与统计复核。

---

# 15. 关键决策表

| 问题 | 默认决策 |
|---|---|
| 是否更换主模拟器 | 否，核心证据成立后再考虑 |
| 首个主任务 | Hidden-Friction Cable Pull |
| 第二任务 | Hidden-Hook Cable Routing |
| 第三任务 | Hidden-Clamp Fabric Fold |
| 刚体任务用途 | Push-L 负对照 |
| 是否固定未来动作 | 必须 |
| 是否允许特权接触标签输入 | 不允许，仅 Oracle |
| 接触信息何时可用 | 预测前的预加载/历史阶段 |
| 评分对象 | 候选未来形变轨迹 |
| 普通跨时间负样本是否足够 | 不足，必须有 hard branch negatives |
| 是否只用 MSE | 不允许 |
| IDM 是否证明可执行 | 否 |
| 是否需要前向验证 | 需要 |
| 是否需要闭环 | 需要 |
| 是否允许只报告最佳 checkpoint | 不允许 |
| 是否允许 Gate 失败后继续训练 | 不允许 |

---

# 16. 最小可发表版本

最低可发表证据集合：

1. 一个严格成立的 Hidden-Friction Cable CCDA 基准；
2. 一个接触拓扑变化任务；
3. 一个2D Fabric任务；
4. Push-L负对照；
5. 动作条件未来扩散基线；
6. 接触拼接与接触条件扩散基线；
7. CCDA未来去噪引导；
8. Hard branch negatives；
9. IDM；
10. 独立可执行性门；
11. 完整离线分支指标；
12. 完整闭环任务指标；
13. Contact-OOD、Location-OOD 和 Geometry-OOD；
14. 三个随机种子；
15. 配对统计与95%置信区间；
16. 失败案例与局限性。

---

# 17. 当前优先级

Agent 当前只应执行以下主线：

1. 冻结 Hidden-Friction Cable 任务定义；
2. 通过全部数据门禁；
3. 建立动作条件未来扩散基线；
4. 证明无接触模型存在错误分支问题；
5. 完成 Contact Concatenation；
6. 完成 CCDA Future Steering；
7. 完成 IDM 与可执行性门；
8. 完成闭环；
9. 再扩展到 Hidden-Hook 与 Fabric；
10. 最后进行 Push-L负对照和跨模拟器验证。

任何不直接服务于以上顺序的工作，默认视为路线偏移。

---

# 18. Agent 每次运行前检查表

```text
[ ] 我是否读过本路线文件？
[ ] 当前处于哪个 Phase？
[ ] 当前唯一允许推进的 Gate 是什么？
[ ] 上一阶段是否有 Scientific PASS，而不只是测试通过？
[ ] 数据、配置、仓库和子模块哈希是否一致？
[ ] 当前实验是否固定未来动作？
[ ] 当前输入是否含特权变量或潜在泄漏？
[ ] 当前指标是否能区分正确分支与平均MSE？
[ ] 当前结果是否包含配对统计与置信区间？
[ ] 当前失败是否应该停止后续训练？
[ ] 当前修改是否会使旧证据失效？
[ ] 当前输出是否包含可复现命令和证据文件？
```

---

# 19. 最终判定标准

本项目只有在以下命题都被证实后，才能宣称完成：

1. 在严格配对条件下，隐藏接触状态导致不同柔性未来；
2. 接触信息在预测前具有可观测信息；
3. 无接触模型无法稳定选择真实分支；
4. 接触条件未来引导提高正确分支概率；
5. 增益优于简单接触拼接和动作空间引导；
6. shuffled contact 与错误接触条件会破坏性能；
7. 预测未来通过可执行性验证后可转化为动作；
8. 闭环策略成功率显著提高；
9. 方法对未见接触强度、位置和几何具有泛化；
10. 在无CCDA负对照中不产生虚假增益；
11. 所有结论可复现、可审计、可追溯。

在此之前，只能声称“阶段性工程或科学证据”，不能提前宣称完整方法成立。

---

# 20. 参考基础

本路线以以下两类方法为起点：

- **Learning Coordinated Bimanual Manipulation Policies using State Diffusion and Inverse Dynamics Models**  
  提供“未来状态扩散 + 逆动力学”的框架基础。

- **TouchGuide: Inference-Time Steering of Visuomotor Policies via Touch Guidance**  
  提供“接触物理评分 + 去噪后期引导 + guidance超参数消融”的方法启发。

本项目的创新重点不在模块组合，而在：

- CCDA 的严格定义；
- 固定未来动作下的反事实分支；
- 面向未来形变轨迹的接触条件引导；
- Hard branch negatives；
- 可执行性闭环；
- 可复现的任务与证据合同。

### Phase 0L Resume1 — Routing-Gate Geometry Provenance

- Preserved the original Phase 0L engineering BLOCKED evidence.
- Added complete rejection provenance for all four fixed endpoint/normal
  geometry candidates.
- Pre-registered one fixed engineering correction:
  `barrier_width: 0.340 m -> 0.350 m`.
- The one-shot seed-71001 capture contradicted the nominal straight-cable
  diagnosis: the settled cable required at most `0.098931884 m`, so the
  original width already had positive coverage margin.
- All four original and repaired candidates failed only the unchanged
  workspace predicate; the repaired geometry had zero accepted candidates.
- The required preflight therefore failed and the three-seed smoke was not run.
- No workspace, target, action, outcome, threshold or classifier field changed.
- No geometry grid search or automatic retry was performed.
- Verdict: `HIDDEN_ROUTING_GATE_RESUME1_SMOKE_BLOCKED`.
- Scientific status remained `UNTESTED`; training remained disabled.

### Phase 0L — Hidden Routing-Gate Outcome-Aligned Smoke

- Closed the prior `delayed_z_latch + whole-cable mean progress`
  task combination without changing Phase 0I/0J/0K verdicts.
- Introduced one fixed `hidden_routing_gate_v1` topology.
- The public task outcome was pre-registered as whether the pulled
  endpoint crossed a fixed target plane inside a fixed corridor.
- The hidden fixture contained an invisible preload probe roof and
  an invisible transverse cable-only barrier.
- Reused the existing precise continuous-grasp routing primitive.
- No geometry, action, outcome or classifier search was performed.
- Official gate: `median routing_success_gap >= 1.0`.
- Verdict: `HIDDEN_ROUTING_GATE_SMOKE_BLOCKED`.
- The fixed layout had no legal workspace geometry at the first
  environment reset, so no pair metrics or physical conclusion were produced.
- Training remained disabled; no retry followed the completed engineering
  result and no geometry change was performed. One earlier invocation was
  client-timeout-aborted before producing any pair, raw file, or result.

### Phase 0L Resume2 — Endpoint-Corridor Barrier Constructor

- Preserved the original Phase 0L and Resume1 engineering BLOCKED
  evidence and their `UNTESTED` scientific status.
- Replaced the whole-settled-cable barrier coverage predicate with a
  constructor aligned to the pre-registered pulled-endpoint routing
  corridor.
- The barrier tangent center was fixed to the pulled endpoint routing
  centerline.
- Barrier width was derived rather than tuned:
  `2 * (corridor_half_width + sqrt(3) * bead_radius + 0.005 m)`,
  giving `0.097320508 m`.
- Workspace, barrier normal offset, target plane, corridor, action,
  sensor feature, scientific thresholds, seeds and official outcome
  were unchanged.
- The Resume1 seed-71001 settled geometry was used for one offline
  engineering preflight; no scientific pair was produced by that
  preflight.
- The preflight passed with two legal endpoint-corridor candidates, but
  the fixed smoke was engineering BLOCKED during seed 71001 reset because
  the post-settle hidden-fixture clearance was `-0.005694588 m`, below the
  fixed `0.002 m` minimum. No action or pair metric was produced.
- Seeds 71002 and 71003 were not run, and no retry or geometry change
  followed the physical failure.
- No geometry grid, width scan, action search, outcome search or model
  training was performed.
- Verdict: `HIDDEN_ROUTING_GATE_RESUME2_SMOKE_BLOCKED`.
- Scientific status: `UNTESTED`.
- Training remained disabled.

### Phase 0L Resume3 — All-Bead-Clearance Probe Selector

- Preserved the Phase 0L, Resume1 and Resume2 engineering BLOCKED
  evidence and their `UNTESTED` scientific status.
- Identified the Resume2 offline/runtime mismatch: the pure geometry
  evaluator checked the probe roof only against its anchor bead, while
  the runtime safety gate checked every hidden fixture against all cable
  beads.
- On the frozen seed-71001 settled cable, fixed probe index 10 had an
  all-bead roof clearance of approximately `-0.005154 m`, with bead 15
  as the nearest non-anchor bead.
- Replaced the fixed center-ratio probe index with a deterministic
  all-bead-clearance selector. It keeps the existing roof geometry and
  `0.002 m` clearance requirement, then chooses the legal interior bead
  nearest to the original center-ratio target.
- The selector chose index 7 on the frozen seed-71001 fixture with
  `0.002050 m` all-bead clearance, and the runtime exact clearance gate
  was passed.
- The fixed smoke was engineering BLOCKED after the seed-71001 free and
  hidden rollouts returned because the action and task public routing
  layouts did not match. No pair was accepted and no scientific metric
  was persisted; seeds 71002 and 71003 were not run.
- Barrier geometry, workspace, target, action, sensor features,
  scientific thresholds, seeds and official outcome were unchanged.
- No probe-index list, roof search, clearance relaxation, geometry grid,
  action search, outcome search, automatic retry or model training was
  performed.
- Verdict: `HIDDEN_ROUTING_GATE_RESUME3_SMOKE_BLOCKED`.
- Scientific status: `UNTESTED`.
- Training remained disabled.

### Phase 0L Resume4 — Frozen Public Routing Layout

- Preserved the Phase 0L through Resume3 evidence and their prior
  scientific status.
- Resume3 had already passed the all-bead probe selection and runtime
  initial-clearance gates, and both seed-71001 rollouts returned.
- The remaining blocker was a public-task lifecycle mismatch: the task
  and hidden fixture used the branch-arm layout, while the action
  generator reconstructed the layout after the no-action interval.
- Resume4 froze one public routing layout immediately after successful
  branch arming.
- The hidden fixture, visible target, action targets, task metadata and
  official outcome reused the frozen public layout. Action grasp poses
  still used the current no-action-end bead positions.
- The exact action/task public-layout equality gate was retained without
  tolerance relaxation.
- Probe selector, fixture geometry, clearance threshold, workspace,
  target offsets, action distances, sensor features, scientific
  thresholds, seeds and official outcome were unchanged.
- The one-shot run was engineering blocked on seed 71001 in the free
  branch before main routing motion. Tangential no-action endpoint drift
  made the current pose0 and frozen stage-1/final targets fail the
  existing primitive's strict collinearity precondition.
- No pair was accepted; seeds 71002 and 71003 were not run, and no
  scientific metric was calculated.
- No preflight, layout search, geometry search, action search, outcome
  search, automatic retry or model training was performed.
- Verdict: `HIDDEN_ROUTING_GATE_RESUME4_SMOKE_BLOCKED`.
- Scientific status: `UNTESTED`.
- Training remained disabled.

### Phase 0L Resume5 — Frozen Task Layout with Relative Pull Action

- Preserved the Phase 0L through Resume4 evidence and their prior
  scientific status.
- Resume4 had already retained one frozen public task layout and passed
  the seed-71001 probe-selection and runtime initial-clearance gates.
- The remaining blocker was an action-coordinate mismatch: the main
  grasp pose used the no-action-end endpoint, while stage-1 and final
  poses used branch-arm absolute targets. Tangential no-action drift
  therefore violated the existing primitive's strict collinearity
  precondition.
- Resume5 retained the frozen target plane, corridor, normal, visible
  target, fixture and official outcome.
- The main pull used the current no-action-end endpoint as pose0 and
  commanded fixed relative displacements of `0.080 m` and `0.120 m`
  along the frozen public normal.
- The existing primitive and its exact collinearity check were unchanged.
- The seed-71001 free and hidden rollouts returned, and the exact
  action/free-task/hidden-task public-layout match passed. The Resume4
  collinearity exception did not recur.
- The run was then engineering blocked before pair acceptance because
  the free metadata had no `tension_pull_lift` event required by the
  unchanged motion-validity validator. No scientific metric was
  persisted, and seeds 71002 and 71003 were not run.
- The hidden-task topology, probe selector, fixture geometry, clearance
  threshold, workspace, pull distances, sensor features, scientific
  thresholds, seeds and official outcome were unchanged.
- The deformable-ravens submodule was unchanged.
- No preflight, distance search, direction search, geometry search,
  outcome search, automatic retry or model training was performed.
- Verdict: `HIDDEN_ROUTING_GATE_RESUME5_SMOKE_BLOCKED`.
- Scientific status: `UNTESTED`.
- Training remained disabled.

### Phase 0L Resume6 — Precise Main-Pull Acquisition

- Preserved the Phase 0L through Resume5 evidence and their prior
  scientific status.
- Resume5 had already passed the fixed endpoint-corridor geometry,
  all-bead probe selection, runtime initial-clearance, frozen public-layout,
  relative-pull distance and exact collinearity gates.
- The remaining blocker was before the first formal tension motion event.
  The tension primitive still used legacy joint-return motion for approach
  and contact lowering, while the exact counterfactual runner discarded the
  primitive result returned by `Environment.step()`.
- Resume6 retained the same pick target, lowering step, contact detector,
  suction grasp predicate, lift height, pull direction, pull distances,
  target plane, corridor, fixture and official outcome.
- Tension approach and lowering reused the existing Cartesian endpoint
  recovery path. Per-lowering-step event rows were suppressed, and one
  `tension_pull_acquisition` result recorded approach, contact and grasp
  success or a single failure reason.
- Exact counterfactual metadata persisted each action result, and the routing
  runner stopped before scientific metrics when a primitive failed.
- The one-shot seed-71001 run was engineering blocked by that new gate when
  the free `routing_contact_probe` preload returned `done=True` with
  `task.done=False`. Main-pull acquisition was therefore not attempted,
  no pair was accepted, and seeds 71002 and 71003 were not run.
- The action/free-task/hidden-task public-layout exact match still passed.
- The existing formal tension events and motion-validity thresholds were
  unchanged.
- No pick search, lowering search, grasp-threshold change, tolerance change,
  geometry search, action-distance search, outcome search, automatic retry or
  model training was performed.
- Verdict: `HIDDEN_ROUTING_GATE_RESUME6_SMOKE_BLOCKED`.
- Scientific status: `UNTESTED`.
- Training remained disabled.

### Phase 0L Resume7 — Completed-Pair Recovery

- Preserved all Phase 0L through Resume6 evidence and their prior
  engineering and scientific status.
- Resume6 had already passed geometry, probe selection, runtime clearance,
  frozen public-layout, relative main-pull distance and command-collinearity
  checks, but its explicit action-result propagation exposed an earlier
  failure in the free `routing_contact_probe` preload.
- Resume7 added no scientific feature, sensor feature, classifier, outcome,
  geometry, action distance or peripheral audit gate.
- The existing Cartesian endpoint-recovery motion was reused for routing
  probe approach and contact lowering. Probe target, lowering step, suction
  activation, grasp predicate, lift height and all formal probe events were
  unchanged.
- One `routing_probe_acquisition` event recorded only the acquisition result;
  per-lowering-step events were not persisted.
- Environment action completion, primitive success, task success and episode
  termination were represented by independent fields. Experiment2 no longer
  inferred primitive or workflow state from the legacy `done` return value.
- Workflow termination was determined separately from task success.
- Engineering status and scientific status were reported separately.
- A seed counted as a completed pair only after both branch actions, existing
  formal probe/tension motion validity, public-layout equality, traces,
  metrics and pair persistence completed.
- Previously completed seeds were retained in the count if a later seed was
  engineering blocked.
- The one-shot seed-71001 run was engineering blocked in the free
  `routing_contact_probe` preload: precise acquisition reached contact, but
  the unchanged suction grasp predicate returned false (`grasp_failed`).
- Verdict: `HIDDEN_ROUTING_GATE_RESUME7_SMOKE_BLOCKED`.
- Engineering status: `BLOCKED`.
- Scientific status: `UNTESTED`.
- Completed pair count: `0`.
- No automatic retry or model training was performed.
