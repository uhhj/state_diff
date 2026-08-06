# CCDA 研究执行主计划 V4.5：解耦剪切刚度标定修订案

> **状态**：本文件是 V4、V4.1、V4.2、V4.3、V4.4 的最小增量修订。  
> **依据**：Phase 0B-R2-R1 已完成 aligned-energy confirmation 并保留 \(M=8\)。A1.25 axial coupon 通过，但 shear coupon 的主位移达到 33.817 mm，显著高于 12 mm 上限；运行 finite、zero-cap、无 no-action/anchor 漂移且可恢复，因此当前主要问题是 shear/structural stiffness ratio 过低，而不是数值积分失稳。  
> **范围**：固定 structural 与 bending 参数，只对 shear stiffness 进行受限标定；不重新验证 microsteps，不调整 coupon gate，不运行 Pair，不训练模型。  
> **执行风格**：主路径优先、少量日志、普通 Git 版本管理，不建设复杂审计、哈希链或恢复合同。

---

## 1. 已确认状态

当前已经成立：

- Kelvin–Voigt 显式材料实现；
- manual force-recomputed microsteps；
- post-step aligned energy；
- \(M=8\)；
- A1.5 的 M8/M16 confirmation；
- A1.25 axial compliance；
- axial no-action/anchor/force-cap/recovery gates。

当前尚未成立：

- shear compliance；
- table-settle；
- frozen material；
- strict HLF-SBP Pair；
- State Diff / IDM / CFPM 训练。

---

## 2. 对当前 shear 结果的解释

A1.25 参数为：

\[
k_s=10.0,\qquad
k_h=3.75,\qquad
k_b=1.0\quad \mathrm{N/m}
\]

其 shear coupon：

\[
d_{\mathrm{shear}}=33.8166\ \mathrm{mm}
\]

同时：

- force-cap fraction = 0；
- no-action drift = 0；
- anchor drift = 0；
- recovery ratio \(\approx0.18\)；
- rigid-aligned RMSE = 2.337 mm；
- edge ratio = [0.6775, 1.3250]。

因此当前失败解释为：

> **有限、可恢复但横向剪切过软，过大剪切位移进一步推高内部边形变。**

不将其解释为积分器 runaway。

---

## 3. 单一 multiplier family 结束

之前的材料族保持固定比例：

\[
k_s:k_h:k_b=8:3:0.8
\]

并仅改变统一 multiplier。A1.25 已证明：

- structural/axial 顺应性合适；
- shear 顺应性过软。

因此本修订不再改变统一 multiplier，而是解耦：

\[
k_s=10.0\ \mathrm{N/m}
\]

\[
k_b=1.0\ \mathrm{N/m}
\]

只调整：

\[
k_h
\]

---

## 4. 新受限材料族

第一候选：

\[
k_h=12.5\ \mathrm{N/m}
\]

相邻候选：

\[
k_h\in\{10.5,\ 15.0\}\ \mathrm{N/m}
\]

profile：

| Profile | Structural | Shear | Bending |
|---|---:|---:|---:|
| S10.5 | 10.0 | 10.5 | 1.0 |
| S12.5 | 10.0 | 12.5 | 1.0 |
| S15.0 | 10.0 | 15.0 | 1.0 |

阻尼比继续为：

\[
\zeta=0.25
\]

每类 damping 仍按：

\[
c=2\zeta\sqrt{k\,m_r}
\]

计算。

基于 A1.25 的有限近线性响应，预计 shear 位移约为：

| Profile | 预计 shear 位移 |
|---|---:|
| S10.5 | 12.08 mm |
| S12.5 | 10.14 mm |
| S15.0 | 8.45 mm |

这些预测只用于选择候选，不替代真实 coupon。

---

## 5. 执行顺序

```text
复用已通过的 R2-R1 M=8 selection
→ S12.5 axial
→ S12.5 shear
→ 必要时一个相邻 profile
→ table-settle
→ freeze material
```

选择规则：

```text
S12.5 TOO_SOFT
→ S15.0

S12.5 TOO_STIFF
→ S10.5

UNSTABLE / ENGINEERING_BLOCKED
→ 停止
```

最多运行两个 profile。第二 profile 必须重新运行 axial 和 shear。

---

## 6. Shear verdict 语义修订

旧 analyzer 在 common edge-ratio gate 失败时优先返回 `UNSTABLE`。本阶段使用更直接的顺序：

1. engineering malformed / nonfinite / collapse → `ENGINEERING_BLOCKED`；
2. no-action、anchor、force-cap、recovery 失败 → `UNSTABLE`；
3. primary 或 rigid response 高于上限 → `TOO_SOFT`；
4. primary 或 rigid response 低于下限 → `TOO_STIFF`；
5. 位移在目标区间但 edge ratio 仍越界 → `UNSTABLE`；
6. 其余 → `COMPLETE`。

不放宽任何数值 gate，只改进失败归因，使有限、zero-cap 的材料过软能够进入相邻 profile 选择。

---

## 7. 不重新运行 microstep confirmation

不重新执行 M4/M8/M16/M32 或 A1.5 confirmation。

原因：

- R2 已用更高刚度 A8 验证数值收敛；
- R2-R1 已用 aligned energy 再次确认 M8；
- 新 profile 的最大 stiffness 不超过已验证范围；
- 重复验证不会增加当前材料比例结论。

当前继续复用：

\[
M=8
\]

---

## 8. 保持不变

不修改：

- manual microstep integrator；
- Kelvin–Voigt force；
- aligned-energy implementation；
- node topology、mass、spacing；
- coupon load、方向、阶段长度；
- axial/shear/table gates；
- force cap；
- patch、pusher、goal；
- Pair；
- State Diff、IDM、CFPM；
- DeformableRavens gitlink。

---

## 9. 完成标准

阶段完成需要：

1. 一个 decoupled profile 通过 axial；
2. 同一 profile 通过 shear；
3. 同一 profile 通过 table-settle；
4. profile 和 \(M=8\) 写入 frozen outputs；
5. 未运行 Pair；
6. 未训练模型。

完成 verdict：

```text
PHASE0B_R2R2_MATERIAL_CALIBRATION_COMPLETE
```

完成后下一任务直接进入：

> 冻结材料接入 strict HLF-SBP Pair。

不继续扩展 coupon 工具。
