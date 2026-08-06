# Phase 0B-R2-R1 restricted low-stiffness calibration

Phase 0B-R2 已证明真正 manual microstep integration、向量化 Kelvin–Voigt 力和 `M=8` 数值设置通过验证。当前问题仅是 A4/A6 profile family 偏硬，而不是积分不稳定。

R2 axial 数据近似满足：

\[
d(\alpha) \approx \frac{3.528}{\alpha}\ \mathrm{mm}
\]

本修订仅允许 A1.0、A1.25、A1.5，先以 A1.5 对 `M=8/16` 做 family confirmation，并保持 `M=8` 冻结。标定顺序固定为 axial → shear → table-settle；最多运行两个 profile。

本阶段不运行 Pair，不调整 patch、pusher、goal 或 motion，不运行 development seeds，也不训练 State Diff、IDM 或 CFPM。
