# CCDA 研究执行主计划 V4.4：同步能量修复并继续材料标定

> **状态**：本文件是 V4、V4.1、V4.2、V4.3 的最小增量修订。  
> **依据**：Phase 0B-R2-R1 中，A1.5 的 M8/M16 位移、过零时间、cube 响应、edge ratio 和 zero-cap 均通过；唯一阻塞项是 M8 two-node 的 `energy_increase_fraction=0.0208768267` 略高于 `0.02`。代码检查表明，当前总能量把 outer-step 结束后的动能与最后 microstep 推进前的弹簧势能相加，存在一个 microstep 的时间错位。  
> **范围**：只修复 two-node 能量时间对齐，重新运行 A1.5 M8/M16 family confirmation；若通过，立即沿既有 R2-R1 主路径继续 A1.25 axial → shear → table-settle。  
> **原则**：不新建复杂执行合同，不重新审计已通过的 mechanics core，不修改材料、microstep、coupon、Pair 或模型代码。

---

## 1. 已确认的数值结论

当前证据支持：

- manual force-recomputed microsteps 已实现；
- outer timestep 为 \(1/240\,\mathrm s\)；
- M8 与 M16 的 two-node peak extension 相对差约 1.90%；
- M8 与 M16 的 first zero crossing 完全一致；
- cube primary 和 rigid-aligned response 相对差约 \(5\times10^{-6}\)；
- M8/M16 均 finite、zero-cap、edge bounded；
- final energy ratio 均接近零；
- 只有 M8 的正能量增量计数比例为 \(10/479=0.0208768\)。

因此，本阶段不把结果解释为材料或 M8 失稳，而解释为：

> **two-node energy measurement alignment defect**

---

## 2. 同步能量定义

当前错误组合为：

\[
E_n^{old}
=
K(x_{n+1},v_{n+1})
+
U(x_{n+1-\delta t})
\]

修订后，outer step 完成后使用同一时刻的位置和速度：

\[
U_n^{aligned}
=
\frac12 k(l_n-l_0)^2
\]

\[
K_n^{aligned}
=
\frac12 m\|v_n\|^2
\]

\[
E_n^{aligned}
=
K_n^{aligned}+U_n^{aligned}
\]

`OuterStepMechanicsStats.final_energy_by_kind_j` 继续作为最后一个 microstep **施力前**的 telemetry 保存，但不得再与 post-step kinetic energy 相加作为稳定性 gate。

---

## 3. Gate 处理

本修订：

- 不放宽 `energy_increase_fraction_max=0.02`；
- 不新增复杂稳定性 gate；
- 不选择 M16；
- 不重新搜索 M4/M32；
- 不改变 A1.5；
- 不改变两节点初始伸长、时长或 damping；
- 仅使既有 gate 使用时间同步的能量序列。

额外报告：

- positive increment count；
- positive increment fraction；
- cumulative positive injection / initial energy；
- maximum positive increment / initial energy；
- maximum total energy / initial energy。

这些额外指标只用于解释，不作为本阶段新增硬门。

---

## 4. 推进路径

```text
修复 aligned energy
→ focused tests
→ 重跑 A1.5 M8/M16 family confirmation
→ 若通过，保留 M=8
→ 立即运行既有 A1.25 axial-first calibration
→ shear
→ table-settle
→ freeze material
```

不再单独建立一个长期“能量审计阶段”。修复确认后立即回到材料标定主路径。

---

## 5. 保持不变

不修改：

- Kelvin–Voigt force formula；
- manual microstep integrator；
- A1.0/A1.25/A1.5 profile；
- R2-R1 config；
- coupon load 和 phase lengths；
- axial/shear/table gates；
- patch、pusher、goal、Pair action；
- State Diff、IDM、CFPM；
- DeformableRavens gitlink。

---

## 6. 工程约束

遵守以下执行风格：

1. 只改与能量错位直接相关的函数和测试。
2. 不建设安全审计、secret scanner 或权限系统。
3. 不新增 hash manifest、tensor SHA 或 provenance graph。
4. 不新增 resume contract、frozen contract 或自动恢复框架。
5. 不重复运行与此次改动无关的全部历史审计。
6. 使用正常 Git commit 管理版本。
7. 日志只保留 summary、必要指标和既有结果文件。
8. confirmation 通过后继续现有 calibration，不长期停留在工具代码。

---

## 7. 完成标准

本修订完成有两种有效结果。

### A. Confirmation 仍失败

- aligned energy 已正确实现；
- focused tests 通过；
- A1.5 M8/M16 已重新运行；
- 输出准确 failure cause；
- 停止，不运行 coupon。

### B. Confirmation 通过

- `PHASE0B_R2R1_FAMILY_CONFIRMATION_COMPLETE`；
- 保留 M=8；
- 立即继续既有 axial → shear → table-settle；
- 最终由既有 R2-R1 verdict 表达材料是否完成标定。

本阶段仍禁止 Pair 和模型训练。
