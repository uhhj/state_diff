# CCDA 研究执行主计划 V4.1：顺应性 Soft BlockPush 修订案

> **状态**：本文件是 V4《Soft BlockPush 克制修订案》的受控增量修订。  
> **依据**：Phase 0B 已建立严格 Pair、隐藏摩擦负载响应、physics-rate 传感领先和全局未来状态分支，但刚体对齐后的形变仅为微米级，任务进度分支未成立。  
> **优先级**：V3 与 V4 中未被本文件覆盖的内容继续有效；冲突处以 V4.1 为准。  
> **修订原则**：只修复已经被实验定位的软体本构、局部锚定和时序评价问题；State Diff、CFPM、TouchGuide 对齐、IDM 和闭环成功率主线保持不变。

---

## 1. Phase 0B 已确认的事实

当前结果确认：

1. Free/High 初始显式状态完全一致；
2. fixed joint-target 数组完全一致；
3. physics-step 和 phase 数组完全一致；
4. 唯一干预变量是局部 patch lateral friction；
5. no-action 可见差异约为 \(10^{-7}\,\mathrm m\)，配对稳定；
6. High patch 切向负载显著高于 Free；
7. 正式传感在 physics trace 上早于视觉分离；
8. 绝对关键点、full-node 和 COM 存在全局未来状态分支；
9. 刚体对齐后形变仅约 \(2\,\mu\mathrm m\)；
10. target-progress gap 未达门槛。

因此：

\[
\text{Strict Pair}
+
\text{hidden-friction load response}
+
\text{global state branch}
\]

已经成立，但：

\[
\text{non-rigid deformation branch}
+
\text{task-progress branch}
\]

尚未成立。

---

## 2. 根因修订

V4 第一版使用大量 midpoint `JOINT_POINT2POINT` 连接 structural、shear 和 bending edges。该结构在高 solver iteration 下形成过约束准刚性桁架；`maxForce` 是约束纠正力上限，不是材料弹簧刚度。

V4.1 将第一版正式软体模型修订为：

> **显式 Kelvin–Voigt 质量—弹簧软体**

每条边的轴向力：

\[
F_{ij}
=
\left[
k_{ij}(l_{ij}-l^0_{ij})
+
c_{ij}(v_{ij}\cdot n_{ij})
\right]n_{ij}
\]

其中：

- \(l^0_{ij}\)：rest length；
- \(l_{ij}\)：当前边长；
- \(n_{ij}\)：从节点 \(i\) 指向节点 \(j\) 的单位方向；
- \(v_{ij}=v_j-v_i\)；
- \(k_{ij}\)：stiffness，单位 N/m；
- \(c_{ij}\)：damping，单位 N·s/m。

对两个端点施加等大反向力：

\[
F_i=+F_{ij},\qquad F_j=-F_{ij}
\]

正式模型不再为 structural、shear、bending 全部创建 P2P 硬约束。

---

## 3. 材料 coupon 先于 CCDA Pair

在重新执行 Free/High Pair 前，必须先运行与隐藏摩擦结果无关的独立材料审计。

固定右侧面节点，向左侧面施加平滑横向载荷，验证：

- 毫米级刚体对齐形变；
- loaded face 与 anchor face 的相对位移；
- 结构和 shear 应变；
- 载荷卸除后的弹性恢复；
- 节点、边长和数值稳定；
- spring force cap 基本不激活。

材料 coupon 未通过时：

- 不调整 patch；
- 不调整 pusher；
- 不运行 CCDA Pair；
- 不训练模型。

材料参数只能由 coupon 选择，不能根据 Free/High Pair 的分支结果回调。

---

## 4. 第一版材料结构保持克制

继续保留：

```text
6 × 4 × 3 = 72 nodes
top-layer 24 formal keypoints
structural / shear / bending edge topology
PyBullet dynamic sphere nodes
固定节点顺序
```

只替换 edge 力学实现。

第一版不加入：

- PyBullet FEM；
- tetrahedral volume constraint；
- 学习材料模型；
- 非线性超弹性；
- 塑性；
- 断裂；
- 多材料；
- 网格自适应。

---

## 5. 局部锚定机制

高摩擦条件不仅要表现为切向力更大，还必须表现为局部运动受抑制。

定义公共 snapshot 时位于 patch 内的底层节点集合：

\[
\mathcal I_{\mathrm{patch}}^0
\]

以及其余底层节点：

\[
\mathcal I_{\mathrm{outside}}^0
\]

对每个分支计算：

\[
\Delta_{\mathrm{inside}}(t)
=
\operatorname{mean}_{i\in\mathcal I_{\mathrm{patch}}^0}
[p_i(t)-p_i(t_0)]
\]

\[
\Delta_{\mathrm{outside}}(t)
=
\operatorname{mean}_{i\in\mathcal I_{\mathrm{outside}}^0}
[p_i(t)-p_i(t_0)]
\]

局部位移梯度：

\[
D_{\mathrm{local}}(t)
=
\|
\Delta_{\mathrm{outside}}(t)
-
\Delta_{\mathrm{inside}}(t)
\|
\]

patch slip reduction：

\[
D_{\mathrm{slip-reduction}}(t)
=
\|
\Delta_{\mathrm{inside}}^{free}(t)
\|
-
\|
\Delta_{\mathrm{inside}}^{high}(t)
\|
\]

新的摩擦机制证据必须同时覆盖：

- High patch 切向载荷增加；
- High patch 节点滑移减少；
- High 条件 patch 内外节点出现更大的位移梯度。

只增加整体阻力不再足以通过局部机制门。

---

## 6. 时序审计修订

physics-rate trace 继续保留，但正式科学时间关系改为 policy-rate：

\[
t_{\mathrm{formal}}^{10\mathrm{Hz}}
<
t_{\mathrm{visual}}^{10\mathrm{Hz}}
\]

物理频率：

```text
240 Hz
```

策略频率：

```text
10 Hz
24 physics steps / policy sample
```

正式门要求：

```text
formal sensor onset 至少早于
macro visible onset 一个完整 policy sample
```

宏观视觉分离必须使用绝对 effect-size floor，而不能因严格 Pair 的 baseline 方差接近零而把阈值降到微米级。

同理，正式传感使用通道物理尺度 floor，避免 baseline gap 为零时 threshold 自动成为零。

---

## 7. 未来分支评价时刻修订

弹性软体在 test 停止后可能恢复，因此不能只用整个 `post_test` 之后的最终状态判定是否出现形变分支。

正式报告同时使用：

1. `test_end`；
2. test 最后三个 policy samples 的 sustained median；
3. `post_test_end`；
4. test 期间 peak。

硬门优先使用：

- `test_end visible/full RMSE`；
- `test_end rigid-aligned RMSE`；
- sustained rigid-aligned RMSE；
- `test_end target-progress gap`。

`post_test_end` 用于报告恢复，不得替代 test 期间真实分支。

---

## 8. Phase 0B-R1 路线

### R1-A：Kelvin–Voigt 实现与单元测试

- 新增显式 spring 力；
- 保留 legacy P2P 实现用于回归；
- edge force 等大反向；
- rest state 零力；
- damping 符号正确；
- force cap 可审计；
- snapshot/restore 确定性保持。

### R1-B：材料 coupon

- no-action 稳定；
- 载荷阶段产生毫米级非刚性形变；
- 卸载阶段部分恢复；
- 无崩塌、穿透或 edge runaway；
- 冻结通过的材料 profile。

### R1-C：局部锚定 probe

- 使用偏心 pusher；
- patch 覆盖软块右前侧少量底层节点；
- High patch 节点滑移减少；
- patch 内外位移梯度建立；
- 10 Hz 正式传感至少领先宏观视觉一个 sample。

### R1-D：full 单 Pair

- test-end 全局状态分支；
- test-end 非刚性形变分支；
- sustained 非刚性分支；
- target-progress branch；
- 结构稳定；
- 同一 snapshot 和 fixed joint targets。

### R1-E：development seeds

只有 R1-D 单 Pair 通过后才允许提出 5–10 个 development seeds。R1 本身不运行多 seed，不训练模型。

---

## 9. V4.1 硬边界

1. 保留旧 Phase 0B 代码、配置、结果和 verdict。
2. 不删除 legacy P2P 软块；将其标为 `p2p_legacy`。
3. Kelvin–Voigt 材料参数只由 coupon 冻结。
4. Pair 阶段禁止再次调整 spring stiffness/damping。
5. 不以提高 solver iterations 代替材料模型。
6. 不以扩大 test push 代替非刚性形变。
7. 不降低 rigid-aligned threshold。
8. 不把 policy-rate 同帧分离称为可利用提前信息。
9. 不把切向力增加单独称为局部 stick–slip。
10. 不训练 State Diff、IDM 或 CFPM。
11. V3 中 State Diff/CFPM/TouchGuide 对齐路线保持不变。
12. V4 中 HLF-SBP 主任务、严格 Pair、正式/Oracle 通道边界保持不变。

---

## 10. V4.1 当前固定决策

1. 当前主任务仍为 HLF-SBP。
2. 72 节点和 top-layer 24 keypoints 保持不变。
3. 正式软体模型改为显式 Kelvin–Voigt。
4. legacy P2P 仅作失败机制回归。
5. 材料 coupon 是重新运行 Pair 的前置门。
6. 正式时序门使用 10 Hz policy-rate trace。
7. physics-rate trace只作诊断。
8. 形变 gate 使用 test-end 和 sustained test-window。
9. 局部机制 gate 增加 patch slip reduction 与 inside/outside displacement gradient。
10. 单 Pair 完成前不进入 development seeds 或任何模型训练。
