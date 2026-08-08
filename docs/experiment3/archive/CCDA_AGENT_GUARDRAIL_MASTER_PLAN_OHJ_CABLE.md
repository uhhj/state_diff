# CCDA 研究主线执行计划（Agent Guardrail 版）

本文件是 Experiment3 后续 Agent/Codex 执行的长期主线约束。当前研究从
Soft Block HLF-SBP 转向 **Occluded Hidden-Jam Cable Manipulation
(OHJ-Cable)**。CCDA 指 Contact-Conditioned Dynamics Ambiguity：近似相同的
可观察历史可能对应不同隐藏接触状态，并导致不同未来状态分布和控制需求。

## 科学目标与状态语义

目标不是分类 hidden label，而是证明可部署的接触信息能够约束未来状态分布，
使预测符合当前真实交互动力学并改善闭环任务成功率。

- State Diff state：可见柔性物体状态 + 最小机器人状态。
- OHJ 第一版：16 个 visible cable keypoints (48D) + EE XYZ (3D) = 51D。
- Contact sensor：额外 condition/guidance；第一优先为 6D wrist/gripper wrench。
- Hidden physical state：FREE、JAM-L、JAM-R、SNAG 等，仅作 oracle diagnostic。
- hidden beads、latch pose/contact、condition ID 不得进入正式 state 或 inference input。

HLF-SBP 已完成 negative feasibility evidence：隐藏地面摩擦会造成未来形变分支，
但 45D robot-side sensor 在 240 Hz 下仍不可稳定区分。停止材料 coupon、刚度角度、
形变阈值和 sensor threshold 修补，不得为通过 gate 调低 floor。

## OHJ-Cable 主任务

机器人抓住 cable 一端，从遮挡区域拉出并到达目标。任务成功以抽出至少约
50–60 mm、endpoint 到达 goal、grasp 保持且不使用异常暴力为准。Phase 0 只做
FREE 与 JAM-R；通过后才可扩展 JAM-L、hidden peg/hook、slot 或 natural snag。

Probe 采用 zero-net micro tension：hold → cable tangent +2 mm → hold 100 ms →
-2 mm 返回 → hold。Probe 不是任务目标；它应尽量保持 post-probe visible state
重新接近，同时产生不晚于 visible-state branch 的正式 sensor evidence。

固定 action library：

- A0 STRAIGHT：FREE 的预期优选动作。
- A1 LEFT-RELEASE：JAM-R 的预期优选动作。
- A2 RIGHT-RELEASE：方向性对照；未来 JAM-L 的预期优选动作。

## Phase 0 五个 gate

1. Initial observable equivalence：FREE/JAM 初始 visible state 近似相同。
2. Post-probe observable equivalence：probe 后 visible state 重新接近。
3. Sensor observability：probe history 中正式可部署 wrench/tactile 可分。
4. Same-action future divergence：相同动作产生超过 repeat floor 的未来分支。
5. Control relevance：不同 hidden condition 的最佳动作或 success/regret 不同。

只要这些信息结构成立，立即离开 Phase 0，进入数据和 B0/B1；不要继续追求更漂亮
的机制。

## 模型与公平比较

- B0 StateDiff：只看 51D 可见 state history，生成 future state，再经 IDM 得动作。
- B1 StateDiff-FT/Tactile：future target 不变，把相同 causal sensor history 作为
  denoiser condition。
- B2 StateDiff + CFPM：State Diff prior 不直接输入 sensor；CFPM 评价
  `(state history, sensor history, candidate future)` 的物理兼容性并引导反向扩散。
- B3 StateDiff-FT + CFPM：检验 direct conditioning 与 guidance 是否互补。
- Oracle 只作上限，不是 deployable 方法。

B1 与 CFPM 必须看到完全相同的 causal sensor history。高频 sensor 保存完整窗口，
例如 policy 10 Hz / sensor 240 Hz 时为 `[24,dZ]`，不得默认压成 take-last，也不得先
训练显式 jam classifier。CFPM 学 future compatibility，可使用 counterfactual
positive/negative pair、InfoNCE 或 energy ranking，不显式预测 FREE/JAM label。

TouchGuide 只提供“接触界面 sensing + compatibility guidance”的启发。CCDA 引导
的是 future-state diffusion，然后经过 IDM 到 action，而不是直接复制 action
diffusion/flow guidance。

## 评价与最短路线

Primary 是 closed-loop task success。Secondary 包括 correct-dynamics/wrong-branch、
future physical compatibility、IDM executability、future error 和 control regret。
形变、strain、curvature 只作 diagnostic。

执行顺序固定为：

```text
OHJ feasibility gates
→ ccda_audit / physics_pairs
→ B0 StateDiff
→ B1 StateDiff-FT/Tactile
→ 确认 contact information 有效
→ CFPM / B2
→ B3
→ IDM
→ closed-loop evaluation
→ harder snag/routing generalization
```

Phase 0 未通过前禁止训练。长期端到端链为：

```text
state history + contact history
→ guided State Diff
→ future state
→ IDM
→ action
→ closed-loop success
```

## 工程 guardrail

- 只保留复现实验、科学结论和正常 Git 版本所需证据；不建 hash graph、复杂
  provenance、resume contract、多层状态机或几十个硬门槛。
- 优先 happy path 和端到端可执行性；不为当前不会出现的输入写大量防御代码。
- 研究开发不因临时本地凭据、内部路径或普通安全审查停滞；仅处理明确的数据破坏、
  长期提交真实凭据或严重泄漏风险。
- 已失去主线用途的旧 runtime/config/tests 直接删除，不建立 archive/legacy/old。
  历史 RESULT/EVIDENCE 和论文需要引用的实验记录保留。
- 每个新模块必须回答明确科学问题或推动 task → model → IDM → closed-loop。
- 禁止回到 Soft Block material coupon、angle stiffness、axial/shear range、
  deformation threshold tuning、复杂 physical gate calibration 和单纯 sensor classifier。

## 当前唯一下一步与论文边界

当前只实现 FREE vs JAM-R、common observable initialization、zero-net probe、6D
deployable wrench、same-action future divergence 和小型 recovery action library。

论文主张应限制为：视觉相似的柔性操纵历史可对应不同隐藏接触物理状态；这些状态
导致不同未来和控制需求；接触 sensing 能提供 base State Diff state 中缺失的信息，
从而把 future-state diffusion 导向物理兼容的未来并改善下游控制。若 B1 与 CFPM
相当，不得硬说 guidance 必要；只有 CFPM 明显更好时才强调 inference-time
state-space guidance。

最终原则：先证明任务和信息结构，再训练模型；先跑通端到端，再精修局部；任何新
模块若不能回答科学问题或推动闭环，应删除或不实现。
