# 附录 A：V4 受控修订案

# CCDA 研究执行主计划 V4：Soft BlockPush 克制修订案

> **状态**：本文件是《CCDA 研究执行主计划（第三版：State Diff / CFPM / TouchGuide 对齐）》的受控修订案。  
> **优先级**：V3 中未被本文件明确覆盖的内容继续有效；发生冲突时，以本修订案为准。  
> **修订原则**：只修改已经由当前工程结果和讨论证明必须修改的任务、环境、变量定义和 Phase 0 路线；State Diff、CFPM、TouchGuide 对齐、反向扩散内部引导、IDM 和闭环成功率主线保持不变。

---

## 1. 保持不变的核心科学问题

项目仍研究：

> **Contact-Conditioned Deformation Branch Ambiguity（CCDA）**  
> 接触条件导致的未来形变分支歧义。

核心因果链保持为：

\[
\text{近似相同的可观察历史}
+
\text{相同动作}
+
\text{不同隐藏接触条件}
\rightarrow
\text{不同未来状态/形变分支}
\rightarrow
\text{不同后续最优动作}
\rightarrow
\text{不同闭环成功率}
\]

核心方法保持为：

\[
p_{\mathrm{CCDA}}(Y\mid H,Z)
\propto
p_\theta(Y\mid H)
\exp(\lambda G_\phi(H,Z,Y))
\]

CFPM 仍必须：

- 输入 `history + contact + noisy future + timestep`；
- 输出可微标量一致性；
- 与 State Diff 使用相同状态布局、normalizer、前向噪声、scheduler 和 timestep 语义；
- 在反向扩散内部提供梯度；
- 不退化为 `free/high_friction` 分类器、候选排序器或事后过滤器。

---

## 2. 主任务受控替换

V3 的 OCCP 不再作为当前主可行性任务。新的主任务是：

> **Hidden Local-Friction Soft Block Pushing（HLF-SBP）**  
> 隐藏局部摩擦条件下的软体 BlockPush。

OCCP 的新定位：

- 保留全部代码、失败结果和复现实验；
- 不再阻塞当前主线；
- 后续作为跨几何、跨接触机制的泛化验证任务；
- 在 HLF-SBP 主线建立前，不继续扩大 OCCP 几何调参。

---

## 3. 环境角色修订

### 3.1 `coord_bimanual/state_diff`

当前 State Diff 仓库中的 3D XArm BlockPush/PyBullet 环境同时承担：

- Phase 0 严格反事实 Pair 生成；
- `ccda_audit`；
- `physics_pairs`；
- `expert_policy`；
- State Diff 训练；
- CFPM 训练和推理；
- IDM；
- 闭环 rollout；
- 最终成功率评估。

因此，HLF-SBP 不再需要先经 DeformableRavens 转换到 `coord_bimanual`。数据和执行环境统一，减少跨模拟器语义偏差。

### 3.2 DeformableRavens

DeformableRavens 保留为：

- OCCP 的数据生成器；
- 后续电缆任务的物理真值环境；
- 跨任务泛化验证环境。

当前 HLF-SBP Phase 0 不修改 DeformableRavens 子模块。

---

## 4. HLF-SBP 第一版任务

第一版主动简化原 Multimodal BlockPush：

```text
一个 XArm
一个圆柱形 pusher
一个三维可变形软块
一个目标区域
一个视觉不可见的局部摩擦 patch
```

暂时移除：

- 第二个 block；
- 第二个目标；
- 原任务由推物顺序产生的多模态；
- 随机软体拓扑；
- 多 patch；
- 多材料；
- 真实触觉图像；
- 多视角视觉。

目的：把分支来源限定为隐藏局部摩擦，而不是原任务顺序多模态。

---

## 5. 两个反事实条件

\[
C\in\{
\texttt{uniform\_low},
\texttt{right\_local\_high}
\}
\]

### `uniform_low`

局部 patch 与其余桌面使用相同低摩擦参数。

### `right_local_high`

patch 几何、外观和位置完全不变，只提高其接触摩擦参数。

Pair 中保持完全一致：

- 软体节点位置和速度；
- 软体拓扑和材料参数；
- 机器人关节位置和速度；
- 末端实际位姿与目标位姿；
- pusher；
- 目标；
- 相机；
- patch 几何；
- 动作数组；
- 物理步数组；
- 随机种子。

唯一正式干预变量是 patch 的局部摩擦接触定律。

---

## 6. “接触条件分离”而非“接触产生”

在 HLF-SBP 中，软块底面从 no-action 阶段就同时与桌面和 patch 法向接触。因此：

- 基线存在法向接触不是污染；
- 不要求 High 条件在 probe 后才首次接触 patch；
- 需要审计的是切向载荷、stick-slip 模式和机器人反馈何时开始分离。

关键时间关系修订为：

\[
t_{\text{contact-condition-separable}}
<
t_{\text{visible-state-divergence}}
\]

其中 `contact-condition-separable` 由正式传感定义，例如：

- joint motor torque；
- joint reaction wrench；
- end-effector tracking error；
- wrist/末端接触 wrench 代理。

Oracle 只用于解释：

- patch 法向力；
- patch 切向摩擦力；
- 接触 patch 的底部节点；
- 节点切向速度；
- stick/slip 比例；
- 真实摩擦系数。

---

## 7. 未来分支的双层定义

设正式可见关键点为：

\[
X_t=[p_t^1,\ldots,p_t^K]\in\mathbb R^{K\times 3}
\]

### 7.1 全局未来状态分支

\[
D_{\mathrm{state}}(t)
=
\operatorname{RMSE}
\left(
X_t^{low},
X_t^{high}
\right)
\]

它包含：

- 整体平移差；
- 整体旋转差；
- 非刚性形变差；
- 任务进度差。

关键点最终位置不同属于正式未来状态分支，并且可以影响闭环成功率。

### 7.2 非刚性未来形变分支

为了避免把普通摩擦导致的滑动距离差夸大为形变创新，必须计算刚体对齐残差：

\[
D_{\mathrm{def}}(t)
=
\min_{R\in SO(3),\,q\in\mathbb R^3}
\operatorname{RMSE}
\left(
X_t^{low},
R X_t^{high}+q
\right)
\]

并补充：

- 结构边应变差；
- 剪切差；
- 弯曲或局部法向差；
- patch 邻域局部形变；
- 质心和目标进度差。

正式强主张要求同时存在：

```text
全局关键点未来分支
+
刚体对齐后仍存在的非刚性形变分支
```

若只有质心位置或刚体姿态不同，应报告为 `future-state branch`，不得单独作为强 `deformation branch` 证据。

---

## 8. 软体表示

Phase 0 第一版使用固定拓扑的 3D 结点—约束软块：

```text
6 × 4 × 3 = 72 nodes
```

包括：

- structural links；
- shear links；
- bending links；
- 固定节点顺序；
- 固定 top/bottom/visible mask；
- 全部节点状态可由 PyBullet 直接读取；
- 不使用原生 FEM 作为第一版阻塞项。

正式可见柔性状态第一版使用 top-layer 24 个关键点：

\[
D_t^{formal}
=
[p_t^i]_{i\in\mathcal V_{\mathrm{top}}}
\]

全部 72 节点用于：

- simulator supervision；
- oracle audit；
- full-state branch metrics；
- 结构应变和稳定性审计。

后续可以比较 top-only、surface-keypoints 和 RGB/深度观测，但 Phase 0 不扩展。

---

## 9. 正式联合状态与接触状态

\[
S_t=[D_t,R_t]
\]

第一版：

- \(D_t\)：top-layer 关键点位置；
- \(R_t\)：joint position、joint velocity、EE pose/velocity、EE target、tracking error。

接触历史：

\[
Z_t=
[
\tau_t^{motor},
W_t^{joint},
W_t^{ee},
e_t^{track}
]
\]

不得输入正式模型：

- condition；
- patch 位置；
- patch 摩擦参数；
- bottom-node patch membership；
- oracle stick/slip；
- simulator-only friction force。

---

## 10. 数据角色保持不变

仍分成：

- `ccda_audit`
- `physics_pairs`
- `expert_policy`

### `ccda_audit`

相同 snapshot、相同固定 probe/test joint-target 数组，用于证明：

- no-action 稳定；
- 正式传感先分离；
- 相同动作产生状态和非刚性形变分支；
- 任务进度出现差异。

### `physics_pairs`

覆盖：

- patch 位置；
- 摩擦强度；
- 推动方向、距离、速度；
- 成功和失败 rollout；
- 真实分支和反事实错配。

### `expert_policy`

允许使用 oracle 条件生成：

```text
Low-friction expert:
probe → continue straight/diagonal push → goal

High-friction expert:
probe → unload/reposition → side push or route around patch → goal
```

正式测试不得读取 condition 或 patch oracle。

---

## 11. Phase 0 路线修订

### Phase 0B0：原 BlockPush 基线确认

- 原 BlockPush 环境可 reset/step/render；
- 不修改原始行为；
- 清理交互式 `pdb` 测试；
- 记录起始 SHA。

### Phase 0B1：软体与局部摩擦表面

- 创建 72 节点软块；
- 创建同高度、无视觉泄漏的分区碰撞桌面；
- Free/High 只改变 patch 摩擦；
- no-action 单分支稳定。

### Phase 0B2：正式传感与精确配对

- joint torque/reaction、EE tracking、EE wrench 非零；
- `saveState/restoreState`；
- 固定 joint-target 命令；
- 两分支 physics-step/phase/action 完全一致。

### Phase 0B3：probe-only 单 Pair

必须先验证：

- no-action 可见和机器人状态近似；
- probe 激活摩擦条件差异；
- 正式传感分离先于可见分叉；
- High patch 的 oracle 切向载荷高于 Free；
- 不要求两分支首次法向接触时间不同。

失败则停止，不执行 full test。

### Phase 0B4：full 单 Pair

同一 test push 后验证：

- visible keypoint branch；
- full-node branch；
- COM/target-progress branch；
- rigid-aligned deformation branch；
- 结构稳定；
- 不只是整体移动速度差。

单 Pair 通过仍只是 smoke，不构成 held-out 结论，也不允许直接训练模型。

---

## 12. 后续阶段

### Phase 0B5：小规模 development seeds

只在单 Pair 机制成立后：

- 5–10 development seeds；
- 检查现象稳定性；
- 冻结第一版几何和动作；
- 再生成独立 held-out seeds。

### Phase 1：原始 State Diff/IDM 恢复

- 建立 HLF-SBP dataset；
- top-keypoint 联合状态；
- 原始 State Diff；
- IDM；
- 最小闭环。

### Phase 2：CCDA 现象与闭环相关性

- 错误分支导致错误后续动作；
- Free/High 最优恢复动作不同；
- unguided 闭环成功率受损。

### Phase 3–6

CFPM noisy-space 训练、反向扩散内部引导、IDM 可执行性和闭环成功率路线继续沿用 V3。

---

## 13. 评价指标修订

### 数据质量

- initial/top-keypoint RMSE；
- full-node state max difference；
- robot-state RMSE；
- action/joint-target equality；
- physics-step equality；
- formal sensor separation step；
- visible divergence step；
- oracle friction-load separation step。

### 全局未来状态

- visible keypoint RMSE；
- full-node RMSE；
- COM gap；
- orientation gap；
- target progress gap；
- final goal distance gap。

### 非刚性形变

- rigid-aligned keypoint RMSE；
- structural edge strain gap；
- shear-link strain gap；
- local patch-neighborhood deformation；
- bending metric；
- edge-length stability range。

### 闭环

- Low/High 分支成功率；
- High-friction recovery success；
- wrong-branch action rate；
- replanning steps；
- peak force；
- IDM executability。

---

## 14. 必须保留的科学边界

1. “关键点绝对位置不同”可以构成未来状态分支。
2. 强形变主张必须额外通过刚体对齐残差或内部应变指标。
3. patch 法向接触在 no-action 存在不是污染；no-action 污染指两分支在未加载时已经出现不可接受的可见/机器人状态分离。
4. 正式传感必须先于宏观可见分叉。
5. Pair 必须使用相同 snapshot 和相同固定命令。
6. 第一个 Phase 0 只验证机制，不训练 State Diff、IDM 或 CFPM。
7. 不通过降低传感阈值、扩大视觉阈值或只报告 COM 差来制造通过。
8. 不修改或删除 OCCP 失败证据。
9. 不新建不必要的模拟器子模块；当前仓库已有 BlockPush 实现。
10. 不把项目退化成摩擦系数分类。

---

## 15. V4 当前固定决策

1. 主可行性任务改为 HLF-SBP。
2. `coord_bimanual/state_diff` 的 PyBullet BlockPush 同时承担数据生成、训练和闭环。
3. OCCP 保留为后续跨机制验证。
4. 第一版一个软块、一个目标、一个隐藏 patch。
5. 第一版 72 节点固定拓扑，正式状态使用 top-layer 24 keypoints。
6. 两分支 patch 几何完全相同，只改变摩擦参数。
7. 接触时间关系使用“摩擦条件可分时间”，不是首次法向接触时间。
8. 未来分支同时报告绝对关键点分支和 rigid-aligned deformation branch。
9. Phase 0 继续使用固定 low-level joint targets 和绝对 physics-step 分析。
10. 单 Pair 通过后才允许 development seeds；development seeds 通过后才讨论训练。

---
