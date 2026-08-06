# 附录 A：CCDA 研究执行主计划


# CCDA 研究执行主计划（第三版：State Diff / CFPM / TouchGuide 对齐）

> **用途**：本文件作为研究 Agent 的主执行依据，防止后续工程实现、实验设计和结论分析偏离核心科学问题。  
> **主线原则**：DeformableRavens 只负责数据生成、反事实配对和物理真值 rollout；`coord_bimanual` 是主要训练、推理、模型改造和端到端执行环境。CCDA 的目标不是对完整候选未来做事后选择或排序，而是在 State Diff 的反向扩散过程中，利用接触条件未来物理一致性模型（CFPM）的可微梯度，把原始未来状态分布连续地引导到更符合当前隐藏接触条件、柔性物体动力学和机器人可执行性的后验分布。

---

# 1. 项目定位

## 1.1 核心科学问题

本项目研究：

> **Contact-Conditioned Deformation Branch Ambiguity（CCDA）**  
> 接触条件导致的未来形变分支歧义。

目标现象：

- 柔性物体的可见状态历史近似相同；
- 机器人本体状态和过去动作近似相同；
- 遮挡区或内部区域的接触状态不同；
- 接触状态能够通过力觉、本体反馈或接触代理信号提前推断；
- 相同后续动作会产生不同的未来形变分支；
- 不同分支导致不同任务结果或成功率。

核心问题不是“模型能否从候选中挑出正确轨迹”，而是：

> 当原始状态扩散模型面对多分支未来时，能否利用接触条件和物理一致性梯度，直接改变反向扩散过程，使生成分布本身向正确、可执行且物理一致的未来分支迁移。

---

# 2. 系统角色划分

## 2.1 DeformableRavens：数据生成器

DeformableRavens 的职责仅限于：

- 构建电缆场景；
- 生成 `free` 与 `hidden_jam` 配对轨迹；
- 保存电缆节点状态；
- 保存机器人状态；
- 保存动作；
- 保存接触、力、力矩或其代理信号；
- 生成未来形变监督；
- 提供 oracle 接触标签；
- 执行固定探测动作和固定未来动作；
- 生成训练、验证和测试数据。

DeformableRavens **不是主要模型环境**，也不是最终论文方法的核心执行框架。

禁止把工作重心长期放在：

- 扩建 DeformableRavens 的复杂策略框架；
- 在其中重写完整 State Diff；
- 为其建立复杂训练体系；
- 把 DeformableRavens 本身包装成主要贡献。

其任务是尽快、可靠地提供 CCDA 所需的配对数据。

## 2.2 `coord_bimanual`：主环境与基线模型

`coord_bimanual` 是主要改造对象，负责：

- 原始 State Diff 基线复现；
- 数据加载和状态编码；
- 联合未来状态扩散；
- 接触条件编码；
- 物理一致性引导；
- 逆动力学；
- 动作生成；
- 闭环推理；
- 模拟器 rollout；
- 最终端到端任务成功率评估。

研究方法必须最终落到 `coord_bimanual` 的真实训练和推理路径中，而不能只停留在离线数据分析或独立候选打分脚本中。

---

# 3. 总体架构

完整系统分成四层：

```text
DeformableRavens
    ↓
CCDA 配对数据
    ↓
coord_bimanual State Diff 基线
    ↓
接触条件物理一致性引导
    ↓
逆动力学与闭环执行
```

具体流程：

1. DeformableRavens 生成严格配对的 `free/jam` 轨迹；
2. 数据转换为 `coord_bimanual` 可直接使用的统一 schema；
3. 原始 State Diff 学习无接触引导的未来联合状态分布；
4. 接触条件模型对反向扩散中的未来状态提供物理一致性梯度；
5. 反向扩散分布被连续改变；
6. 最终未来状态通过逆动力学产生动作；
7. 在 `coord_bimanual` 中闭环执行并验证成功率。

## 3.1 与 TouchGuide 的关系

TouchGuide 已经属于推理期分布引导，而不是候选排序方法。其核心做法是：

\[
\hat\epsilon_\theta
=
\epsilon_\theta(A_k,V)
-
\eta\sqrt{1-\bar\alpha_k}
\nabla_{A_k}s_\phi(V,T,A_k)
\]

即使用 Contact Physical Model（CPM）对带噪动作轨迹给出可微可行性分数，并在动作扩散的后期去噪步骤中直接修改噪声预测。

CCDA 不得把“我们进行分布引导而不是排序”写成相对 TouchGuide 的创新点。两者真正的区别必须保持为：

| 维度 | TouchGuide | CCDA |
|---|---|---|
| 基础生成变量 | 动作轨迹 | 柔性物体—机器人联合未来状态 |
| 基础模型 | 动作扩散或动作 flow matching | State Diff |
| 引导空间 | 动作空间 | 未来状态空间 |
| 接触语义 | 当前观察—动作接触可行性 | 隐藏接触条件—未来形变动力学一致性 |
| 核心数据 | 专家观察—动作匹配 | 严格配对的反事实接触—未来分支 |
| 输出后处理 | 动作直接执行 | 未来状态经 IDM 生成动作 |
| 科学目标 | 让动作更符合当前接触物理 | 消解隐藏接触导致的未来分支歧义 |

如果实现仅把 TouchGuide 的 noisy action 替换成 noisy future state，并照搬普通时间错配对比学习，则创新性不足。强版本必须包含：

1. CCDA 现象的正式定义和因果配对；
2. 隐藏接触变量的反事实干预；
3. 接触条件与未来形变分支的连续一致性建模；
4. 状态空间后验分布引导；
5. 联合未来状态到 IDM 动作的可执行性验证；
6. 正确分支概率质量和闭环成功率的分布级证据。

---

# 4. 主任务：OCCP

## 4.1 任务名称

> **Occluded Contact-Conditioned Cable Pulling（OCCP）**  
> 遮挡接触条件下的电缆拉动。

CCDA 是科学问题；OCCP 是主实验任务。

## 4.2 隐藏条件

第一版只实现：

- `free`
- `right_hidden_jam`

暂不加入：

- 左侧 Jam；
- 多圆柱；
- 隐藏高摩擦；
- 多材料；
- 双臂；
- 随机抓取点；
- 触觉图像；
- 多目标形状。

## 4.3 几何结构

- 电缆穿过不透明遮挡区；
- 机器人抓住一侧可见端点；
- 另一端固定或稳定夹持；
- 遮挡区内放置一个竖直圆柱销；
- 圆柱销偏离电缆中心线；
- Jam 条件下电缆从圆柱一侧经过，形成小包角切向接触；
- Free 条件下电缆与圆柱保留足够间隙。

建议初始范围：

| 参数 | 建议范围 |
|---|---:|
| 圆柱半径 | \(1.5d_c \sim 2.5d_c\) |
| 圆柱距机器人侧遮挡出口 | \(4\ell \sim 6\ell\) |
| Jam 间隙 | \(0 \sim 0.25d_c\) |
| Free 间隙 | 不小于 \(0.75d_c\) |
| Jam 初始包角 | \(20^\circ \sim 30^\circ\) |
| 初始松弛量 | \(2\ell \sim 4\ell\) |
| 被动端 | 固定 |
| 可见读出区 | 遮挡出口外 3–8 个节点 |

其中：

- \(d_c\)：电缆直径；
- \(\ell\)：节点间距。

## 4.4 动作结构

### 探测阶段

沿电缆轴线小幅预加载：

\[
\Delta x_{\text{probe}}
\approx 0.5\ell \sim 1.0\ell
\]

目标：

- Jam 条件建立接触和张力；
- 接触信号开始可分；
- 可见形状仍近似一致。

### 分支放大阶段

执行较长的轻微斜向拉动：

\[
\Delta x_{\text{test}}
\approx 4\ell \sim 8\ell
\]

斜向角：

\[
10^\circ \sim 25^\circ
\]

目标：

- Free：电缆整体平滑跟随；
- Jam：圆柱成为临时支点；
- Jam 的包角、曲率、出口侧移和进度差明显增大；
- 产生宏观、稳定、可见的未来分支。

---

# 5. CCDA 配对数据

## 5.1 配对方式

Free 与 Jam 必须由同一个基础状态或同一仿真快照生成。

保持一致：

- 电缆长度、质量、节点数和材料；
- 可见节点初始位置；
- 机器人关节状态；
- 末端位姿；
- 抓取点；
- 相机；
- 控制器；
- 随机种子；
- 预加载动作；
- 后续测试动作。

主要干预变量：

\[
C \in \{\text{free},\text{hidden\_jam}\}
\]

## 5.2 五个必要条件

正式 CCDA pair 必须满足：

1. 可见历史相同或近似；
2. 机器人本体状态和过去动作相同或近似；
3. 隐藏接触条件不同；
4. 接触条件能够通过传感信息推断；
5. 相同后续动作产生不同未来形变和任务结果。

## 5.3 关键时间关系

\[
t_{\text{contact-separable}}
<
t_{\text{visual-divergence}}
\]

接触或本体反馈必须先于宏观视觉分叉。

## 5.4 数据必须分成三类

不得把所有轨迹混成同一种“训练数据”。

### A. `ccda_audit`

动作来源：

- 固定轴向预加载；
- 固定斜向测试动作；
- Free/Jam 完全相同的动作 chunk。

用途：

- 证明隐藏接触的因果效应；
- 测量接触分离时间和视觉分叉时间；
- 验证相同动作是否产生稳定未来分支；
- 不要求动作成功，也不要求是专家动作。

### B. `physics_pairs`

动作来源：

- 固定脚本动作；
- 覆盖不同方向、距离和速度的动作；
- 可包含成功和失败的真实物理 rollout。

用途：

- 训练 CFPM；
- 构造接触—未来匹配正样本；
- 构造 Free/Jam 反事实错配负样本；
- 构造 State Diff 错误分支和平均化未来的困难负样本。

### C. `expert_policy`

动作来源：

- 脚本专家；
- oracle 状态机；
- 运动规划或优化控制器；
- 后续可加入人工遥操作。

用途：

- 训练成功未来状态分布；
- 训练 IDM；
- 训练闭环恢复策略；
- 验证端到端任务成功率。

## 5.5 是否需要专家演示

需要，但用途必须区分。

### 不需要专家演示的部分

CCDA 因果现象审计使用同一固定动作，不依赖专家策略：

\[
H^{free}\approx H^{jam},\qquad
A^{free}=A^{jam},\qquad
Y^{free}\neq Y^{jam}
\]

### 需要专家演示的部分

State Diff 作为策略模型需要学习成功或合理的未来状态，而 IDM 需要学习未来状态到动作的映射。因此必须提供分支特定的成功轨迹。

第一版允许在数据生成时使用 oracle 条件标签：

```text
Free expert:
probe → continue pull → reach goal

Jam expert:
probe → retract/unload → lateral release → repull → reach goal
```

正式测试策略不得读取 `free/jam` 标签、隐藏圆柱位置或原始 simulator contact pair。

### 专家数据不等于人工演示

仿真中的脚本控制器、状态机、优化器和规划器生成的稳定成功轨迹均可作为 simulator expert demonstrations。不要因暂时没有人工遥操作系统而阻塞主线。

---

# 6. 数据 schema

## 6.1 联合状态

继承 State Diff 的联合状态逻辑：

\[
S_t=[D_t,R_t]
\]

其中：

### 柔性物体状态

\[
D_t=[p_t^1,\ldots,p_t^N]
\]

第一版优先使用节点位置。

### 机器人本体状态

推荐包括：

- 关节位置；
- 关节速度；
- 末端位姿；
- 末端速度；
- 抓取状态；
- 位置跟踪误差。

## 6.2 接触条件

接触信息单独定义：

\[
Z_t=
[F_t^{ee},\tau_t^{ee},F_t^{joint},\tau_t^{joint}]
\]

可加入：

- 接触法向力代理；
- 末端位姿误差；
- 控制误差；
- 接触数量。

原始模拟器 contact pair 和隐藏条件标签只作 oracle，不直接输入正式模型。

## 6.3 动作

\[
A_t
\]

包括实际控制动作，不把相机参数、任务元数据或静态编码混入动作向量。

## 6.4 统一数据结构

DeformableRavens 输出应转换为 `coord_bimanual` 使用的固定 schema，例如：

```text
episode/
    state
    deformable_state
    robot_state
    contact_state
    action
    visible_mask
    branch_label
    success
    pair_id
    condition
    dataset_role
    expert_flag
    goal
```

其中：

- `condition` 和 `branch_label` 只用于监督、审计和消融；
- `dataset_role ∈ {ccda_audit, physics_pairs, expert_policy}`；
- `expert_flag` 只表示轨迹是否为任务成功策略，不表示其物理真值优先级；
- 正式推理时模型不能读取 `condition`、`branch_label` 或 oracle 几何。

---

# 7. 原始 State Diff 基线

## 7.1 基础分布

原始模型学习：

\[
p_\theta
\left(
S_{t+1:t+H}
\mid
S_{t-h:t}
\right)
\]

输入：

- 柔性物体历史；
- 机器人本体状态历史；
- 可选过去动作历史。

输出：

- 柔性物体未来；
- 机器人未来状态。

## 7.2 多分支问题

在 CCDA 数据中，同一可见历史可能对应多个未来分支，因此原始模型可能：

- 生成错误分支；
- 混合两个分支；
- 产生平均化未来；
- 生成几何合理但接触不一致的未来；
- 生成无法由逆动力学执行的未来。

这正是 CCDA 要解决的问题。

---

# 8. CCDA 方法：分布引导，而非选择器

## 8.1 禁止的错误方向

本项目的主方法不是：

- 先生成大量候选；
- 再用分类器排序；
- 再选择最高分候选；
- 用二分类器决定 Free/Jam；
- 将方法退化为候选过滤器。

候选排序只能作为消融或弱基线，不能作为 CCDA 的核心方法。

## 8.2 正确目标

接触条件应直接改变反向扩散分布。

设基础未来分布为：

\[
p_\theta(S^{future}\mid H)
\]

其中 \(H\) 是可见状态与机器人历史。

接触条件为 \(Z\)。

目标分布可写为：

\[
p_{\text{CCDA}}(S^{future}\mid H,Z)
\propto
p_\theta(S^{future}\mid H)
\exp\left(
\lambda\,G_\phi(S^{future},H,Z)
\right)
\]

其中：

- \(G_\phi\) 是可微的接触—物理一致性评分、对数似然或负能量；
- \(\lambda\) 是引导强度；
- 该形式表示基础分布被连续重加权并迁移；
- 最终样本来自被引导后的分布，而不是事后排序。

## 8.3 反向扩散中的引导

基础 score：

\[
s_\theta(S_k,k,H)
\]

引导后的 score：

\[
s_{\text{guided}}
=
s_\theta
+
\lambda_k
\nabla_{S_k}
G_\phi(S_k,H,Z)
\]

也可以采用能量形式：

\[
s_{\text{guided}}
=
s_\theta
-
\lambda_k
\nabla_{S_k}
E_\phi(S_k,H,Z)
\]

其中：

- \(S_k\)：第 \(k\) 个去噪时刻的未来状态；
- \(E_\phi\)：接触物理不一致性能量；
- \(\lambda_k\)：随去噪时间变化的引导强度。

引导应直接作用于：

- 噪声预测；
- score；
- 预测的 \(x_0\)；
- 或反向扩散均值。

最终实现选择应以当前 State Diff 代码结构和稳定性为准。

## 8.4 引导对象

引导对象是未来联合状态：

\[
S^{future}
=
[D^{future},R^{future}]
\]

不只是动作，也不只是分支标签。

引导应促使未来满足：

- 接触条件一致；
- 电缆形变连续；
- 节点拓扑和顺序一致；
- 长度和局部段长合理；
- 圆柱接触几何一致；
- 机器人运动与柔性物体响应协调；
- 未来能够通过逆动力学实现。

---

# 9. 物理一致性引导模型

## 9.1 模型输出

优先设计为以下一种：

### A. 可微对数一致性

\[
G_\phi(S^{future},H,Z)
\]

值越大表示未来与接触信息越一致。

### B. 可微能量

\[
E_\phi(S^{future},H,Z)
\]

值越小表示未来越物理一致。

### C. 接触观测似然

\[
\log p_\phi(Z\mid S^{future},H)
\]

它直接提供：

\[
\nabla_{S^{future}}
\log p_\phi(Z\mid S^{future},H)
\]

## 9.2 不应仅预测分支标签

只预测 `free/jam` 标签的分类器信息量太弱，容易把方法退化为分支选择器。

正式引导模型应至少对以下连续量建模：

- 接触力趋势；
- 力矩趋势；
- 张力代理；
- 末端跟踪误差；
- 节点曲率；
- 节点位移传播；
- 接触区域的局部几何；
- 机器人和电缆的联合动力学一致性。

## 9.3 可训练目标

可采用组合目标：

\[
L_{\text{guide}}
=
\alpha L_{\text{contact}}
+
\beta L_{\text{geometry}}
+
\gamma L_{\text{dynamics}}
+
\eta L_{\text{executability}}
\]

第一版不要一次加入过多复杂项。推荐最先实现：

1. 接触信号与末端跟踪误差的一致性；
2. 未来局部几何和段长一致性；
3. 端点进度、曲率传播和分支方向一致性。

## 9.4 CFPM 的定位

本项目中与 TouchGuide CPM 对应的模块暂称：

> **Contact-Conditioned Future Physical Consistency Model（CFPM）**  
> 接触条件未来物理一致性模型。

CFPM 输入：

\[
G_\phi(H,Z,Y_k,k)
\]

其中：

- \(H\)：联合状态历史；
- \(Z\)：接触历史；
- \(Y_k\)：State Diff 当前带噪未来联合状态；
- \(k\)：扩散时间步。

CFPM 输出一个未经过 sigmoid 的标量一致性分数。推理时使用：

\[
g_k=\nabla_{Y_k}G_\phi(H,Z,Y_k,k)
\]

CFPM 不是：

- Free/Jam 选择器；
- 事后候选排序器；
- IDM；
- 二值几何 gate；
- 不可微过滤器。

## 9.5 CFPM 编码结构

第一版采用四分支编码：

```text
历史联合状态 H ── history encoder ─┐
接触历史 Z ───── contact encoder ──┼─ condition embedding
扩散时间步 k ─── timestep encoder ─┘
                                         ↓
带噪未来 Y_k ─── future encoder ── FiLM / cross-attention
                                         ↓
                                  temporal pooling
                                         ↓
                                  scalar score G
```

### 历史状态编码器

输入：

\[
H=S_{t-h:t}
\]

建议使用：

- temporal MLP；
- 1D CNN；
- 或 2–4 层小型 Transformer。

必须保留时间顺序，不应简单全局平均。

### 接触历史编码器

接触信号独立编码，第一版可采用：

```text
LayerNorm → Conv1D → SiLU → Conv1D → temporal pooling
```

输入可包括：

- joint reaction force/torque；
- 末端等效力；
- 跟踪误差；
- 接触代理信号。

### 带噪未来编码器

输入：

\[
Y_k\in\mathbb R^{H_f\times D_s}
\]

应保持 State Diff 的未来长度、状态通道顺序和 mask 语义。CFPM 参数不必与 State Diff U-Net 共享；第一版禁止为了“复用”而耦合基础生成模型与引导模型。

### 时间步编码器

时间步必须与 State Diff 使用相同的噪声语义，可使用独立参数的 sinusoidal embedding。重点是 \(k\) 对应同一个 scheduler 噪声水平，而不是强制共享 embedding 权重。

## 9.6 编码与解码原则

低维 State Diff 没有额外 VAE decoder。其完整链路为：

```text
物体状态 + 机器人本体状态
        ↓
联合状态拼接
        ↓
State Diff normalizer
        ↓
前向加噪 / 反向去噪
        ↓
未来联合状态
        ↓
IDM
        ↓
动作
```

CFPM 不负责解码未来，也不直接生成动作。它只在反向扩散中提供梯度。真正的“状态解码”仍由 State Diff scheduler 完成，动作解码仍由 IDM 完成。

显式物理损失可以在 CFPM 内部对 \(Y_k\) 做可微反归一化，再用米、弧度、牛顿等物理单位计算。梯度必须继续传回归一化空间中的 \(Y_k\)。

## 9.7 与 State Diff 必须一致的内容

凡是决定 \(Y_k\) 数学含义的设置必须一致：

- 联合状态维度；
- 通道顺序；
- 物体/机器人切片；
- 历史长度与未来 horizon；
- condition mask 和 visible mask；
- 同一套 State Diff normalizer；
- 前向加噪公式；
- beta schedule；
- \(\alpha_k\) 和 \(\bar\alpha_k\)；
- timestep 集合与顺序；
- prediction type；
- variance parameterization；
- scheduler.step；
- inpainting/conditioning 顺序；
- inference sampling path。

CFPM 接触模态可以使用自己的 normalizer，但 noisy future 不能使用一套与 State Diff 不同的坐标系统。

## 9.8 不需要与 State Diff 相同的内容

以下属于 CFPM 独立超参数：

- CFPM encoder 架构；
- hidden dimension；
- Transformer 层数；
- optimizer；
- learning rate；
- batch size；
- epoch；
- 负样本比例；
- 对比学习温度；
- timestep 采样分布；
- guidance window；
- guidance scale；
- gradient normalization；
- guidance channel mask；
- CFPM EMA。

第一版建议 CFPM 小于基础 U-Net，例如：

- token dimension：128 或 256；
- temporal blocks：2–4；
- attention heads：4；
- scalar head：2 层 MLP。

## 9.9 CFPM 的噪声训练

CFPM 训练时必须使用与 State Diff 相同的前向加噪过程：

\[
Y_k
=
\sqrt{\bar\alpha_k}Y_0
+
\sqrt{1-\bar\alpha_k}\epsilon
\]

这是从 TouchGuide 中应继承的关键做法：辅助引导模型必须在基础生成模型实际经过的 noisy-variable space 中训练。

时间步集合与 State Diff 相同，但采样分布可以不同。推荐：

\[
p_{\text{CFPM}}(k)
=
\rho p_{\text{uniform}}(k)
+
(1-\rho)p_{\text{guidance-window}}(k)
\]

第一版可用：

- 50% 全时间步均匀采样；
- 50% 从实际计划引导的低噪声窗口采样。

不要机械复用 State Diff 的 loss-second-moment sampler，因为 CFPM 优化目标不是扩散噪声回归。

## 9.10 CFPM 训练样本

### 正样本

\[
(H,Z^{free},Y^{free})
\]

\[
(H,Z^{jam},Y^{jam})
\]

### 关键反事实负样本

\[
(H,Z^{free},Y^{jam})
\]

\[
(H,Z^{jam},Y^{free})
\]

### 困难负样本

- State Diff 生成的错误分支；
- Free/Jam 平均化未来；
- 局部段长异常未来；
- 机器人与物体运动不协调未来；
- Jam 条件下错误的自由滑动未来；
- Free 条件下错误的受阻未来。

第一版可使用 pairwise energy loss：

\[
L_{\text{pair}}
=
-\log\sigma
\left[
G_\phi(H,Z,Y^+_k,k)
-
G_\phi(H,Z,Y^-_k,k)
\right]
\]

训练时使用 pairwise loss不意味着推理时排序。推理时只使用能量面的梯度连续改变去噪分布。

## 9.11 推理期引导

对于 epsilon prediction 的 State Diff，形式为：

\[
\hat\epsilon_\theta
=
\epsilon_\theta
-
\lambda_k c_k
\nabla_{Y_k}G_\phi(H,Z,Y_k,k)
\]

其中 \(c_k\) 必须根据当前 State Diff scheduler 和 epsilon/sample 参数化推导，不能机械复制 TouchGuide 的动作空间系数。

插入位置：

```text
State Diff U-Net 预测
        ↓
CFPM 计算 score 与 grad
        ↓
修改 noise prediction
        ↓
原 scheduler.step
```

必须对以下位置做 gradient mask：

- 已固定的历史条件步；
- 不允许改变的状态通道；
- 第一版可只保留柔性物体未来通道。

第一版建议：

\[
g_k=[\nabla_{D_k}G,\;0_R]
\]

即先只引导柔性物体未来，验证有效后再比较联合状态引导。

## 9.12 引导窗口与强度

基础 State Diff 的 checkpoint、scheduler、随机种子集合、采样步数、temperature、horizon、conditioning 和 IDM 在首轮比较中必须保持不变。

CFPM 首先只在低噪声后期阶段引导。若总采样步数为100，建议扫描：

\[
K_{\text{guide}}\in\{5,10,20,30\}
\]

不要直接套用 TouchGuide 的 guidance scale，因为动作空间与高维状态空间的梯度尺度不同。优先采用 RMS 或相对范数归一化：

\[
\bar g_k=
\frac{g_k}{\operatorname{RMS}(g_k)+\epsilon}
\]

或控制：

\[
\|\Delta\epsilon_{\text{guide}}\|
=
r\|\epsilon_\theta\|
\]

第一版可扫描：

\[
r\in\{0.01,0.03,0.1,0.3\}
\]

## 9.13 TouchGuide 的实现参考

TouchGuide 的 CPM 做法可作为工程参考：

- 冻结的 DINOv2 编码视觉和触觉；
- Transformer 融合观察模态；
- 1D CNN + MLP 编码动作；
- L2 normalization；
- 观察 embedding 与动作 embedding 的余弦相似度作为可行性分数；
- 同时刻专家观察—动作作为正样本；
- 时间错配观察—动作作为负样本；
- CPM 在与基础动作扩散一致的噪声空间中训练；
- 后期去噪阶段使用分数梯度修改动作 noise prediction；
- Diffusion Policy 使用100步 DDPMScheduler，并在最后约10–20步引导；
- 论文中的动作空间 guidance scale 约3–4，但不得直接移植到 CCDA。

CCDA 应继承其“独立引导模型 + noisy-space training + 后期去噪梯度”的工程骨架，不应照搬其“观察—专家动作匹配”语义。

---

# 10. 训练与推理阶段

## Phase 0：数据生成最小闭环

目标：

- DeformableRavens 能生成 Free/Jam 配对数据；
- 力觉通道非零；
- 预加载后接触可分；
- 主拉动后未来明显分叉；
- 数据能被转换为 `coord_bimanual` schema；
- 同时建立 `ccda_audit`、`physics_pairs` 和最小 `expert_policy` 数据路径。

不训练复杂模型。

## Phase 1：`coord_bimanual` 原始基线恢复

目标：

- 原始 State Diff 能在新数据上训练；
- 联合状态维度正确；
- 逆动力学能训练；
- 单 batch 前向和反向通过；
- 能完成开放环预测；
- 能完成最小 rollout。

这一步优先于复杂引导。

## Phase 2：CCDA 现象验证

验证：

- 早期可见状态近似；
- 接触信息提前分离；
- 相同动作产生稳定未来分支；
- 分支差异大于随机扰动；
- Free/Jam 任务结果明显不同。

## Phase 3：专家未来、基础分布与 IDM 训练

先用 Free/Jam 分支特定的 simulator expert demonstrations 训练成功未来状态分布和 IDM，再使用 `physics_pairs` 补充真实响应覆盖。

训练成功未来分布：

\[
p_\theta(S^{future}\mid H,G)
\]

训练逆动力学：

\[
p_\psi(A_t\mid S_t,S_{t+1:t+m})
\]

评估：

- 分支覆盖；
- 错误分支率；
- 平均化轨迹；
- 多样性；
- ADE/FDE；
- 物理违规；
- IDM 动作误差；
- IDM 可执行率；
- oracle future 经 IDM 后的任务成功率。

## Phase 4：CFPM 训练

训练：

\[
G_\phi(H,Z,Y_k,k)
\quad\text{或}\quad
E_\phi(H,Z,Y_k,k)
\]

要求：

- 使用与 State Diff 相同的 noisy future 坐标系；
- 使用反事实分支错配作为核心困难负样本；
- 同时覆盖实际计划使用的低噪声引导窗口；
- 不依赖 `free/jam` 标签作为正式推理输入。

检查：

- 对真实未来配对的一致性分数更高；
- 对反事实错配未来的分数更低；
- 梯度方向稳定；
- 梯度对错误分支有纠正作用；
- 接触打乱后引导效果消失；
- CFPM 不仅识别轨迹真伪，而是学习接触条件—未来分支关系。

## Phase 5：去噪分布引导

在反向扩散中加入：

\[
\nabla_{S_k}G_\phi
\]

重点调试：

- 引导时段；
- 引导强度；
- 梯度归一化；
- 是否作用于全部状态或仅柔性物体部分；
- 是否保留机器人—柔性物体协调；
- 是否导致样本坍缩；
- 是否破坏基础分布的可行性。

## Phase 6：逆动力学与端到端执行

完整链路：

```text
状态历史 + 接触历史
        ↓
CCDA 引导状态扩散
        ↓
未来联合状态
        ↓
逆动力学
        ↓
动作
        ↓
coord_bimanual 闭环 rollout
```

最终评价以：

- 真实任务成功率；
- Jam 恢复成功率；
- 动作可执行率；
- 错误分支率；
- 最大接触力；
- 闭环稳定性；

为核心。

---

# 11. 基线与消融

## 11.1 必须包含的基线

1. 原始 State Diff；
2. State Diff + 机器人本体状态；
3. State Diff + 过去动作；
4. 接触直接拼接条件化；
5. 接触 classifier-free conditioning；
6. 候选排序/选择器，仅作为弱基线；
7. TouchGuide-style 动作空间接触引导；
8. 仅把 TouchGuide CPM 从动作替换到状态的直接迁移基线；
9. CCDA 反事实未来一致性分布引导；
10. Oracle contact；
11. Oracle future + IDM。

## 11.2 关键消融

- 无接触；
- 接触打乱；
- 仅分支标签；
- 连续接触信号；
- 早期去噪引导；
- 后期去噪引导；
- 固定 \(\lambda\)；
- 时间变化 \(\lambda_k\)；
- 仅柔性物体状态引导；
- 联合状态引导；
- 无 IDM；
- 有 IDM；
- 无闭环重规划；
- 有闭环重规划。

---

# 12. 评价指标

## 12.1 CCDA 数据质量

- 预加载可见状态差异；
- 机器人状态差异；
- 动作差异；
- 接触信号分离度；
- 视觉分叉时间；
- 接触分离时间；
- 未来分支距离；
- 任务进度差；
- 成功率差；
- 分支放大率。

## 12.2 分布质量

- 正确分支概率质量；
- 错误分支概率质量；
- 分支覆盖；
- 模式平均化程度；
- 条件熵下降；
- 接触打乱后的性能下降；
- 样本多样性；
- 分布坍缩情况。

不要只报告“选中正确候选的比例”。

## 12.3 物理一致性

- 节点顺序；
- 局部段长误差；
- 曲率合理性；
- 接触几何一致性；
- 机器人—物体协调性；
- 力和形变趋势一致性；
- 未来状态的 IDM 可恢复率。

## 12.4 端到端执行

- 闭环任务成功率；
- Free/Jam 分开成功率；
- Jam 恢复成功率；
- 动作可执行率；
- 抓取保持率；
- 关节限制违规率；
- 工作空间违规率；
- 最大接触力；
- 平均恢复步骤数；
- 失败类型分布。

---

# 13. 工程执行原则

## 13.1 主线优先

所有工作应优先回答：

1. 是否帮助生成有效 CCDA 数据？
2. 是否帮助恢复原始 State Diff？
3. 是否帮助实现分布级接触引导？
4. 是否帮助逆动力学和闭环执行？
5. 是否帮助形成可信科学结论？

若答案均为否，应停止该支线。

## 13.2 不过度关注安全问题

- 不因本地测试密码、临时路径或内部开发信息阻塞研究；
- 不主动把真实长期凭证提交到公开仓库；
- 只处理会直接造成仓库破坏、数据泄露或实验不可运行的明显问题；
- 不建设与当前研究规模不匹配的安全审计系统。

## 13.3 不过度审计

- 只保留支持复现和科学结论所需的日志；
- 不反复审计已经通过且未变化的组件；
- 不把审计结果当作主要研究产出；
- 不因非关键 provenance 信息缺失停止实验。

## 13.4 不设置复杂执行合同

- 每阶段只保留少量可操作的完成标准；
- 阈值用于决策，不是不可修改的永久合同；
- 不建立多层 frozen contract、resume contract 和复杂 gate；
- 出现问题优先修复主路径并继续推进。

## 13.5 禁止过度防御

- 不为基本不可能出现的输入写大量分支；
- 不为低概率异常增加复杂恢复系统；
- 不用大量断言保护简单且已知的数据流；
- 只处理会破坏数据、训练、模型或结论的常见失败。

## 13.6 不过度使用哈希

- 正常 Git commit 足以管理版本；
- 不为每个张量、缓存、模型、报告和阈值生成 SHA256；
- 只在数据传输、发布冻结或确有损坏风险时使用哈希；
- 不建立哈希依赖图。

## 13.7 端到端优先

不得长期停留在：

- 数据审计；
- 传感分类；
- 离线未来预测；
- 候选排序；
- 单独物理判别器；
- 未接入真实推理路径的工具代码。

每个模块最终必须接入：

\[
\text{去噪}
\rightarrow
\text{未来状态}
\rightarrow
\text{逆动力学}
\rightarrow
\text{闭环执行}
\]

---

# 14. 禁止偏离的方向

研究 Agent 不得将项目改造成：

- DeformableRavens 主环境研究；
- 单纯接触分类；
- Free/Jam 选择器；
- 候选轨迹排序器；
- 事后过滤系统；
- 只优化 ADE/FDE 的预测任务；
- 只优化物理约束损失但不闭环执行；
- 复杂安全和审计工程；
- 复杂哈希和证据链工程；
- 大量低概率防御代码；
- 在主任务未成立前扩展多任务、多模拟器和双臂。

---

# 15. 最短推进路线

1. 在 DeformableRavens 中完成 OCCP 的 `ccda_audit` 配对数据；
2. 固定主几何配置并验证接触先于视觉分叉；
3. 生成覆盖真实响应的 `physics_pairs`；
4. 用 oracle 状态机生成最小 `expert_policy` 成功轨迹；
5. 建立到 `coord_bimanual` 的数据转换；
6. 恢复原始 State Diff 训练；
7. 恢复 IDM；
8. 建立基础开放环和闭环 rollout；
9. 实现 CFPM 的 history/contact/noisy-future/timestep 编码；
10. 使用与 State Diff 一致的前向加噪训练 CFPM；
11. 验证真实配对分数高于反事实错配；
12. 验证 CFPM 梯度能纠正错误分支方向；
13. 将梯度接入 State Diff 反向扩散；
14. 验证正确分支概率质量增加，而非仅候选排序收益；
15. 验证物理一致性和 IDM 可执行率改善；
16. 验证 `coord_bimanual` 闭环任务成功率改善；
17. 对比 TouchGuide-style 动作引导和状态直接迁移基线；
18. 做必要消融后再扩展几何和任务。

---

# 16. 主线完成标准

项目主线完成需要形成以下证据链：

1. DeformableRavens 可生成严格配对的隐藏接触数据；
2. 接触信号早于宏观视觉分叉；
3. simulator expert demonstrations 能支持成功未来和 IDM 训练；
4. 原始 State Diff 在该数据上呈现真实多分支、错误分支或平均化问题；
5. CFPM 与 State Diff 使用同一 normalizer、scheduler、noisy future 定义和 timestep 语义；
6. CFPM 能区分真实接触—未来配对与反事实错配，并提供稳定、有效的状态梯度；
7. 引导发生在反向扩散内部；
8. 引导后的分布增加正确物理分支的概率质量；
9. 结果不是简单候选排序或 Free/Jam 分类造成的；
10. 未来联合状态具有更高接触动力学和几何一致性；
11. IDM 能将其转化为可执行动作；
12. `coord_bimanual` 闭环成功率得到可靠提升。

最终优先级：

\[
\boxed{
\text{可信的分布级物理引导}
+
\text{可执行的未来状态}
+
\text{端到端闭环成功}
}
\]


---

# 17. 当前不可随意偏离的已确定决策

1. DeformableRavens 只承担数据生成、配对和物理真值 rollout。
2. `coord_bimanual` 是 State Diff、CFPM、IDM 和闭环执行的主环境。
3. CCDA 主方法是反向扩散内部的未来状态分布引导，不是候选选择器。
4. TouchGuide 本身已经是分布引导，因此不能把“不排序”当作相对创新。
5. CCDA 的主要差异是隐藏接触导致的未来形变分支问题、反事实配对数据、未来状态空间后验引导和 IDM 可执行性。
6. 需要 simulator expert demonstrations 训练成功未来和 IDM，但 CCDA 审计数据不需要专家动作。
7. CFPM 必须输入 `history + contact + noisy future + timestep`，输出可微标量一致性。
8. CFPM 与 State Diff 必须共享状态布局、normalizer、前向噪声、scheduler 和 timestep 语义。
9. CFPM 网络规模、损失、优化器、负样本和 guidance 超参数独立设计。
10. 第一版只在低噪声后期引导，并优先只引导柔性物体未来通道。
11. 第一版比较必须冻结基础 State Diff、IDM、scheduler、采样步数、temperature 和随机种子集合。
12. 评价必须覆盖正确分支概率质量、物理一致性、IDM 可执行率和闭环成功率。
13. 不得重新把项目导向复杂 gate、候选过滤、过度审计、哈希链或大量低概率防御。

