# CCDA 研究执行主计划 V4.3：受限低刚度材料族重新标定修订案

> **状态**：本文件是 V4、V4.1 和 V4.2 的受控增量修订。  
> **依据**：Phase 0B-R2 已完成真正 manual microstep integration、向量化 Kelvin–Voigt 力、时间一致性和数值收敛验证；A6 与 A4 axial coupon 均稳定、zero-cap、无锚点漂移，但轴向位移低于 2 mm 下限。  
> **优先级**：V3、V4、V4.1、V4.2 中未被本文件覆盖的内容继续有效；冲突处以 V4.3 为准。  
> **当前范围**：只修订受限材料 profile family，并按 axial → shear → table-settle 顺序重新标定；不运行 Pair，不调整 patch/pusher，不训练任何模型。

---

## 1. Phase 0B-R2 已确认的结论

Phase 0B-R2 已建立：

1. 外层步长固定为 \(1/240\ \mathrm s\)；
2. 每个 manual microstep 都重新计算 Kelvin–Voigt 内部力；
3. 每个 manual microstep 都重新施加 coupon 外载；
4. Bullet `numSubSteps=1`；
5. 向量化 512-edge 计算与净内部力平衡成立；
6. `M=8` 与 `M=16` 收敛；
7. 选择 \(M=8\)，微步长为：
   \[
   \delta t=\frac{1}{240\times8}
   =0.0005208333333333333\ \mathrm s
   \]
8. A6 与 A4 axial 运行均 finite、zero-cap、无锚点漂移、无 no-action 漂移；
9. A6 位移为 \(0.588406\ \mathrm{mm}\)；
10. A4 位移为 \(0.881438\ \mathrm{mm}\)；
11. shear、table-settle、Pair 和模型训练均未运行。

因此，当前阻塞项不是数值积分，而是 profile family 未覆盖目标顺应性。

---

## 2. 数据驱动的低刚度区间

A6 与 A4 的 axial 结果近似满足：

\[
\alpha d(\alpha)\approx3.528\ \mathrm{mm}
\]

因此：

\[
d(\alpha)\approx\frac{3.528}{\alpha}\ \mathrm{mm}
\]

为了满足 axial gate：

\[
d(\alpha)\ge2\ \mathrm{mm}
\]

需要：

\[
\alpha\lesssim1.764
\]

V4.3 将受限候选修订为：

\[
\alpha\in\{1.0,\ 1.25,\ 1.5\}
\]

预测值仅用于选择搜索区间，不替代真实 coupon：

| Profile | \(\alpha\) | 预测 axial 位移 |
|---|---:|---:|
| A1.0 | 1.00 | 3.53 mm |
| A1.25 | 1.25 | 2.82 mm |
| A1.5 | 1.50 | 2.35 mm |

---

## 3. 保持不变的材料定义

继续保持基础 stiffness 比例：

\[
k_s^0:k_h^0:k_b^0=8:3:0.8
\]

对于 multiplier \(\alpha\)：

\[
k_\ell=\alpha k_\ell^0
\]

阻尼比保持：

\[
\zeta=0.25
\]

节点约化质量：

\[
m_r=\frac{1}{2}\frac{0.03}{72}
\]

每类阻尼：

\[
c_\ell=2\zeta\sqrt{k_\ell m_r}
\]

不改变：

- soft-block topology；
- 节点质量；
- force cap；
- coupon load；
- coupon phase lengths；
- axial/shear gate；
- table-settle gate；
- outer timestep；
- manual microstep integrator。

---

## 4. Microstep 决策

`M=8` 已由比当前 family 更高刚度的 A8 profile完成 `8 vs 16` 收敛验证，因此 V4.3 不重新开放 `4/8/16/32` 搜索。

只进行一次 family-specific confirmation：

```text
profile: A1.5
microsteps: 8 and 16
```

验证：

- two-node finite、zero-cap、bounded；
- cube finite、zero-cap、bounded；
- `M=8` 与 `M=16` 的主要指标相对差不超过 5%。

若通过：

```text
继续冻结 M=8
```

若失败：

```text
工程阻塞
```

不得自动选择 M=16，也不得修改材料系数制造通过。

---

## 5. Profile 选择顺序

第一 profile 固定为：

```text
A1.25
```

选择规则：

### Axial

```text
A1.25 COMPLETE
→ 同 profile 运行 shear

A1.25 TOO_STIFF
→ 第二且最后一个 profile 为 A1.0
→ 从 axial 重新开始

A1.25 TOO_SOFT
→ 第二且最后一个 profile 为 A1.5
→ 从 axial 重新开始

A1.25 UNSTABLE / ENGINEERING_BLOCKED
→ 停止，不切换 profile
```

### Shear

```text
A1.25 shear COMPLETE
→ table-settle

A1.25 shear TOO_STIFF
→ 第二且最后一个 profile 为 A1.0
→ 重新运行 axial，再运行 shear

A1.25 shear TOO_SOFT
→ 第二且最后一个 profile 为 A1.5
→ 重新运行 axial，再运行 shear

A1.25 shear UNSTABLE / ENGINEERING_BLOCKED
→ 停止
```

最多运行两个 profile。第二 profile 未同时通过 axial 与 shear 时停止。

---

## 6. 标定顺序

正式顺序：

```text
A1.5 family-specific M8/M16 confirmation
→ retain M=8
→ A1.25 axial
→ same-profile shear
→ 必要时一个相邻 profile
→ table-settle
→ freeze material
```

任何前置步骤失败均停止。

---

## 7. 冻结要求

只有同一个 profile 同时满足：

```text
family confirmation complete
+
axial complete
+
shear complete
+
table-settle complete
```

才允许生成：

```text
FROZEN_MATERIAL.json
FROZEN_MICROSTEPS.json
calibration_manifest.json
```

冻结内容必须记录：

- profile；
- explicit coefficients；
- \(\alpha\)；
- \(\zeta\)；
- outer timestep；
- \(M=8\)；
- micro timestep；
- family confirmation；
- axial evidence；
- shear evidence；
- table-settle evidence；
- Pair 未运行；
- training 未运行。

---

## 8. 与 CCDA 主线的关系

V4.3 不修改 V3 中的：

- State Diff；
- CFPM；
- TouchGuide 对齐；
- noisy-future training；
- 反向扩散内部梯度引导；
- IDM；
- 闭环成功率证据链。

V4.3 不修改 V4 中的：

- HLF-SBP 主任务；
- strict counterfactual Pair；
- Free/High 唯一摩擦干预；
- formal/oracle 通道边界；
- global future-state branch；
- rigid-aligned deformation branch；
- policy-rate sensor lead。

本阶段仍只是材料前置标定。

---

## 9. 硬边界

1. 不运行 probe-only Pair。
2. 不运行 full Pair。
3. 不调整 patch、pusher、goal 或 Pair motion。
4. 不训练 State Diff、IDM 或 CFPM。
5. 不运行 development seeds。
6. 不重新搜索 microsteps。
7. 不改变 coupon load 以制造位移。
8. 不修改 axial/shear 位移 gate。
9. 不提高 force cap。
10. 不增加 Bullet linear damping。
11. 不删除 Phase 0B、R1、R2 的失败证据。
12. 不根据预期 Pair 效果选择材料。
13. 第一个通过完整标定的 profile 立即冻结。
14. 两个 profile 上限继续有效。

---

## 10. V4.3 完成标准

只有以下全部成立，才能输出：

```text
PHASE0B_R2R1_MATERIAL_CALIBRATION_COMPLETE
```

1. A1.5 的 M8/M16 family confirmation 通过；
2. M=8 保持冻结；
3. 一个 profile 通过 axial；
4. 同一 profile 通过 shear；
5. 同一 profile 通过 table-settle；
6. force cap 基本不激活；
7. 材料 profile 被冻结；
8. legacy Phase 0B、R1、R2 未破坏；
9. 未运行 Pair；
10. 未运行模型训练。

完成后的下一阶段才允许制定：

> 冻结材料和 M=8 接入 HLF-SBP strict Pair 的独立执行计划。
