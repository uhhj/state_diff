# CCDA 研究执行主计划 V4.6：独立角度剪切弹性修订案

> **状态**：本文件是 V4–V4.5 的最小增量修订。  
> **依据**：Phase 0B-R2-R2 证明，现有“结构边 + 对角 shear 边 + bending 边”的标量边弹簧族无法同时满足 axial 与 shear coupon。S12.5 的 axial 低于 2 mm；S10.5 的 axial 刚好通过，但 shear 仍为 20.55 mm。  
> **结论**：对角 shear springs 同时参与轴向伸长和剪切，继续调三个标量 edge-family stiffness 很难独立控制两个模态。下一步不再扩大标量 profile 搜索，而是在现有通过 axial 的 A1.25 edge material 上增加独立的面角度弹性。  
> **范围**：实现正交面角度能量、做一个最小 M8/M16 单元确认、完成 axial/shear/table-settle；不运行 Pair，不训练模型。

---

## 1. 固定的 edge material

继续使用已经通过 axial 的 A1.25 edge coefficients：

\[
k_{\rm structural}=10.0\ {\rm N/m}
\]

\[
k_{\rm diagonal}=3.75\ {\rm N/m}
\]

\[
k_{\rm bending}=1.0\ {\rm N/m}
\]

阻尼比：

\[
\zeta=0.25
\]

该材料的 axial peak primary 为 2.797 mm，已在目标范围内。

---

## 2. 为什么需要独立角度项

当前 lattice 的 shear edges 是各网格面的对角弹簧。它们既阻止面内剪切，也在轴向拉伸时改变长度，因此提高 diagonal stiffness 会同时：

- 降低 shear displacement；
- 降低 axial displacement。

R2-R2 已观察到这种耦合：

- \(k_h=10.5\) 时 axial 约 2.008 mm，但 shear 约 20.55 mm；
- \(k_h=12.5\) 时 axial 已降至 1.862 mm。

因此下一步需要一个在纯轴向伸长下近似不激活、但在网格面失去正交性时产生恢复力的独立能量。

---

## 3. 角度能量

对每个网格面角 triplet：

\[
p_0,\ p_1,\ p_2
\]

定义：

\[
u=p_1-p_0,\qquad v=p_2-p_0
\]

\[
c=\frac{u^\top v}{\|u\|\|v\|}
\]

初始正交网格：

\[
c_0=0
\]

角度能量：

\[
E_{\rm angle}
=
\frac12 k_{\rm angle}(c-c_0)^2
\]

阻尼使用：

\[
q
=
k_{\rm angle}(c-c_0)
+
d_{\rm angle}\dot c
\]

并通过 \(c\) 对三个节点位置的解析梯度施力。

该项在保持正交的纯轴向伸长下接近零，在 xy/xz/yz 面发生剪切时激活。

---

## 4. Angle triplets

每个网格 cell 只在其最小坐标角建立一个正交 triplet，避免四角重复计数：

- xy：45；
- xz：40；
- yz：36；
- 总计：121。

节点顺序和 triplet 顺序必须确定性固定。

---

## 5. 受限 angle family

固定 edge material，只调整：

\[
k_{\rm angle}\in
\{0.00040,\ 0.00055,\ 0.00070\}\ {\rm N\,m}
\]

第一 profile：

\[
0.00055\ {\rm N\,m}
\]

规则：

```text
TOO_SOFT  → 0.00070
TOO_STIFF → 0.00040
UNSTABLE / ENGINEERING_BLOCKED → stop
```

最多两个 profile。

---

## 6. 数值积分

保持：

\[
f_{\rm outer}=240\ {\rm Hz}
\]

\[
M=8
\]

不重新搜索 M4/M32。

由于 angle term 是新力项，只对最高 angle profile 做一次最小三节点 angle-cell 的 M8/M16 confirmation：

- finite；
- zero-cap；
- 角误差衰减；
- M8/M16 主要响应误差不超过 5%。

通过后立即进入 coupon，不建立独立长期审计阶段。

---

## 7. 完整路径

```text
实现 angle force
→ focused tests
→ A070 三节点 M8/M16 confirmation
→ A055 axial
→ A055 shear
→ 必要时一个相邻 angle profile
→ table-settle
→ freeze material
→ 下一阶段 strict Pair
```

---

## 8. 不改变的内容

不改变：

- manual microstep integrator；
- aligned energy；
- node topology、mass、spacing；
- coupon load、时长和 gates；
- force-cap 规则；
- HLF-SBP patch/pusher/goal；
- State Diff、IDM、CFPM；
- DeformableRavens gitlink。

---

## 9. 工程风格

- 不建设复杂安全审计；
- 不重复审计已通过 mechanics；
- 不建立多层 frozen/resume contract；
- 不生成 tensor/report SHA256 图；
- 不加入大规模参数搜索；
- 新 angle term 通过后直接回到 coupon 和 strict Pair 主线。

---

## 10. 完成标准

只有同一 profile 同时通过：

- angle-cell confirmation；
- axial；
- shear；
- table-settle；

才输出：

```text
PHASE0B_R3_MATERIAL_CALIBRATION_COMPLETE
```

完成后下一任务是将冻结的 edge+angle material 接入 strict HLF-SBP Pair。
