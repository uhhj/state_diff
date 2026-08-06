# CCDA 研究执行主计划 V4.2：真正子步积分与分离式材料标定修订案

> **状态**：本文件是 V4 与 V4.1 的受控增量修订。  
> **依据**：Phase 0B-R1 中，C0 已产生毫米级非刚性形变与部分弹性恢复，但面间位移和边长变化过大；C2 在更高刚度下出现大规模 force-cap 激活、穿透与数值 runaway。  
> **优先级**：V3、V4、V4.1 中未被本文件覆盖的内容继续有效；冲突处以 V4.2 为准。  
> **当前范围**：只处理显式软体的数值积分、材料分量标定和桌面静置稳定性，不运行 Free/High Pair，不调整 patch/pusher，不训练任何模型。

---

## 1. 已确认事实

Phase 0B-R1 已确认：

1. 72 节点 Kelvin–Voigt soft block 已实现；
2. structural、shear、bending 内部 P2P constraint 数为零；
3. 单边 Kelvin–Voigt 力方向、等大反向施力和 spring telemetry 已实现；
4. legacy P2P 环境和旧结果保持完整；
5. C0 的 peak rigid-aligned RMSE 为毫米级，说明显式材料能够产生真实非刚性形变；
6. C0 的恢复比例表明材料具有部分弹性恢复；
7. C0 的 face-relative displacement 和 edge ratio 超出可接受范围；
8. C2 的高刚度结果不是“更硬材料的物理形变”，而是显式积分 runaway；
9. 当前 Pair 未运行，符合 coupon 前置门。

因此，Kelvin–Voigt 路线未被否定，但当前材料模拟尚未达到可冻结状态。

---

## 2. 当前数值问题

现实现每个 240 Hz 外层步只计算一次内部力，然后让 Bullet 使用固定外力推进内部 substeps：

\[
F_t = F(x_t,v_t)
\]

内部 substep 之间没有重新计算：

\[
F(x_{t+\delta t},v_{t+\delta t})
\]

因此，当前 `numSubSteps=2` 只能增加 Bullet 碰撞/约束内部更新次数，不能构成真正的 spring-force microstep integration。

V4.2 要求每个 mechanics microstep 都重新：

```text
读取节点位置和速度
→ 计算全部 Kelvin–Voigt forces
→ 施加节点净内部力
→ 施加当前外载
→ stepSimulation
```

---

## 3. 真正的 manual microstep integration

保持外层 trace 和控制频率：

\[
f_{\text{outer}} = 240\ \mathrm{Hz}
\]

设每个外层步有 \(M\) 个 mechanics microsteps：

\[
\delta t = \frac{1}{240M}
\]

每个 microstep 都重新计算内部力。Bullet 配置为：

```text
fixedTimeStep = outer_dt / M
numSubSteps = 1
```

不再使用“外层 240 Hz timestep + Bullet numSubSteps > 1”模拟材料子步。

microstep 数不凭经验直接冻结，而通过：

- 两节点阻尼振荡；
- \(2\times2\times2\) 单元载荷；
- \(M\) 与 \(2M\) 的收敛误差；

选择。候选仅限：

\[
M\in\{4,8,16,32\}
\]

---

## 4. 材料参数降维

不再同时任意搜索六个 stiffness/damping 参数。

保留 C0 的材料比例：

\[
k_s^0:k_h^0:k_b^0 = 8:3:0.8
\]

使用一个 stiffness multiplier：

\[
k_\ell = \alpha k_\ell^0
\]

候选仅限：

\[
\alpha\in\{4,6,8\}
\]

阻尼由统一阻尼比计算。对两个等质量节点的约化质量：

\[
m_r=\frac{m_{\text{node}}}{2}
\]

每类 edge：

\[
c_\ell=2\zeta\sqrt{k_\ell m_r}
\]

第一版固定：

\[
\zeta=0.25
\]

因此材料选择变量只剩一个 \(\alpha\)，而不是六个独立系数。

---

## 5. 分离式材料 coupon

R1 coupon 同时混合了 structural extension、shear、bending、gravity、floor contact、friction 和 P2P fixture。

V4.2 将材料校准分成三个实验。

### 5.1 Axial coupon

- gravity = 0；
- 无 floor；
- `x_max` 面静态固定；
- `x_min` 面沿 `-x` 平滑加载；
- 主要标定 axial/structural compliance；
- 记录 lateral leakage 和恢复。

### 5.2 Shear coupon

- gravity = 0；
- 无 floor；
- 同样固定 `x_max` 面；
- `x_min` 面沿 `+y` 平滑加载；
- 主要验证 shear/bending response；
- 必须产生稳定毫米级 rigid-aligned deformation。

### 5.3 Table-settle audit

- 全部节点恢复为动态节点；
- gravity = \([0,0,-9.8]\)；
- 使用 uniform-low floor；
- 不加载机器人、patch 条件或外载；
- 不在 settle 后 rigid recenter；
- 验证实际桌面环境中的静置稳定性。

执行顺序：

```text
microstep validation
→ axial coupon
→ shear coupon
→ table-settle
→ freeze material
```

任何一步失败都禁止 Pair。

---

## 6. Coupon 边界条件修订

R2 axial/shear coupon 不再使用 world P2P fixture 作为锚点。

固定面的节点在 coupon 专用实例中创建为：

```text
baseMass = 0
```

其余节点仍保持任务中的单节点质量，不重新分配固定节点的质量。

这是一种 coupon 边界条件，不是正式 soft block 内部本构。正式任务中的 72 个节点仍全部动态。

---

## 7. Force cap 的定位

force cap 只允许作为 catastrophic numerical guard，不得成为材料力学的一部分。

正常 coupon 要求：

\[
\text{force-cap fraction} < 10^{-3}
\]

优先要求为零。

若 force cap 大量激活，该 rollout 直接归类为数值 runaway，不允许通过提高 force cap 修复。

---

## 8. 收敛与稳定性证据

### 两节点振荡

验证：

- 有限；
- 无 force cap；
- 边长不崩溃；
- 阻尼后总能量总体下降；
- \(M\) 与 \(2M\) 的主要指标收敛。

### 单元载荷

验证：

- \(2\times2\times2\) 单元有界形变；
- 无 edge runaway；
- 无 force cap；
- face displacement 与 rigid-aligned RMSE 在 \(M\) 与 \(2M\) 间相对差不超过 5%。

冻结最小通过 microstep 数；若所有候选都不收敛，工程阻塞。

---

## 9. Profile 选择

先运行：

\[
\alpha=6
\]

若 axial 或 shear 结果整体过软，只允许尝试：

\[
\alpha=8
\]

若整体过硬，只允许尝试：

\[
\alpha=4
\]

最多运行两个 profile。第二个仍未同时通过 axial、shear 和 table-settle，则停止。

第一个完整通过三个 coupon 的 profile立即冻结。

---

## 10. 与 CCDA 主线的关系

V4.2 不修改：

- HLF-SBP 任务定义；
- strict counterfactual Pair；
- Free/High 唯一干预；
- formal/oracle 边界；
- 全局状态分支和 rigid-aligned deformation branch 的区分；
- policy-rate sensor lead；
- State Diff；
- CFPM；
- noisy-future guidance；
- IDM；
- 闭环成功率评价。

V3 的分布级未来状态引导主线继续有效。  
V4 的 Soft BlockPush 主任务和数据角色继续有效。

---

## 11. 当前硬边界

1. 不运行 probe-only 或 full Pair。
2. 不调整 patch、pusher、goal 或 Pair action。
3. 不训练 State Diff、IDM 或 CFPM。
4. 不运行 development seeds。
5. 不删除 R1 C0/C2 失败证据。
6. 不用更高 force cap 掩盖 numerical runaway。
7. 不用 Bullet internal `numSubSteps` 代替 force-recomputed microsteps。
8. 不在 coupon 通过前修改材料以外的任务参数。
9. 不把 free-space coupon 结果直接等同于桌面稳定性；必须通过 table-settle。
10. 只有 material calibration complete 后，下一阶段才允许将冻结材料与 manual microsteps 接入 R1 Pair。

---

## 12. V4.2 完成标准

只有以下全部成立，才能输出：

```text
PHASE0B_R2_MATERIAL_CALIBRATION_COMPLETE
```

1. manual microstep force recomputation 实现正确；
2. 两节点和单元验证通过；
3. microstep 收敛通过；
4. 一个 profile 通过 axial coupon；
5. 同一 profile 通过 shear coupon；
6. 同一 profile 通过 table-settle；
7. force cap 基本不激活；
8. profile 和 microstep 数被冻结；
9. legacy Phase 0B/R1 未破坏；
10. 未运行 Pair 或模型训练。
