# Provable Scaling Laws of Feature Emergence from Learning Dynamics of Grokking 阅读笔记

## 0. 封面信息与一句话摘要

**论文标题**：Provable Scaling Laws of Feature Emergence from Learning Dynamics of Grokking

**早期标题 / alternative title**：Li₂: A Framework on Dynamics of Feature Emergence and Delayed Generalization

**作者**：Yuandong Tian

**arXiv**：2509.21519

**当前核对版本**：arXiv v5。根据 arXiv 页面，初始提交日期为 2025-09-25，最近修订为 2025-12-02。v5 版本摘要备注中还指出：新增机制说明初始阶段的 \(G_F\) 也携带有用信号，从而减少理论对 weight decay 的依赖；同时加入 zero-initialized output layers 可显著加速 grokking 的实验。本文笔记以 v5 的公开信息和论文主体内容为准，但定理编号、精确常数、sample complexity 的 scaling law 仍需要回到原文 PDF/TeX 逐条核对。

**关键词**：grokking；delayed generalization；memorization-to-generalization transition；feature emergence；learning dynamics；two-layer nonlinear networks；group arithmetic；energy landscape；nonlinear CCA；irreducible representation；Fourier feature；mechanistic interpretability。

**阅读目的**：理解 grokking 中“训练集已拟合但测试集迟迟不泛化”的内部动力学机制，并把论文中的数学框架转化为可复现、可诊断、可扩展到 mechanistic interpretability 的研究问题。

**与我的研究方向的关系**：这篇论文直接连接 Mechanism-Guided Trustworthy AI Systems、mechanistic interpretability、representation learning 和 trustworthy generalization。它不是只问模型是否在测试集上答对，而是追问模型内部到底学到了 sample-specific memory、shortcut，还是稳定、可组合、可解释的结构表征。

**一句话摘要**：

> 这篇论文试图把 grokking 从一个经验现象，推进为一个可以由反向传播梯度结构、能量函数局部极值、群表示理论和样本量相变共同解释的数学框架；它说明从记忆到泛化不是神秘顿悟，而是模型从随机特征插值逐渐转向结构表征学习的动力学过程。

**适用边界**：

> 严格证明主要在两层非线性网络和 group arithmetic 任务中成立；迁移到 LLM 是机制类比，而不是完整证明所有大语言模型都按此机制 grok。

这条边界非常重要。论文的价值在于给出一个可证明的 toy-to-theory 框架，而不是声称真实 Transformer、真实语言建模和真实开放域推理都已经被同一个定理完全解释。对 LLM 的意义应理解为：它提供了观察表征涌现、分辨记忆与机制化泛化、设计内部诊断工具的理论模板。

---

## 1. 论文要解释的问题：grokking 到底是什么？

Grokking 或 delayed generalization 描述的是一种特殊训练现象：

1. 训练集准确率很早达到高水平；
2. 测试集准确率长期很低；
3. 继续训练很久以后，测试集准确率突然跃升；
4. 从行为曲线上看，模型似乎先“记忆”，后“顿悟”；
5. 关键科学问题是：这个“顿悟”在模型内部对应什么表征变化？

传统泛化讨论常常把问题压缩为训练误差和测试误差的差距。但 grokking 的特殊性在于，训练误差已经很低以后，模型仍会继续发生内部重组。也就是说，训练集拟合完成并不等于表征学习完成。模型可能先用大量随机特征完成训练集插值，随后在长时间优化中逐渐发现更低复杂度、更结构化、更可泛化的特征。

本文试图回答的研究问题包括：

1. 什么样的特征会在训练过程中涌现？
2. 这些特征是怎么从反向传播中出现的？
3. 为什么早期是记忆，后期变成泛化？
4. 数据量和数据分布如何改变模型最终学到的表征？
5. 能否用数学证明记忆型表征和泛化型表征的区别？
6. 这个理论如何连接 mechanistic interpretability？

这里的“顿悟”不应被理解为心理学意义上的突然理解，而应被理解为动力学相变：早期 loss 已经能被随机特征空间中的线性读出拟合；后期隐藏层参数继续移动，使隐藏激活从 sample-specific interpolation 逐渐转向 structure-specific representation。当结构表征达到足够强度，测试集性能才表现为突然跃升。

---

## 2. 基本模型设定：两层非线性网络

论文的核心理论以两层非线性网络为主要分析对象。设训练输入矩阵为：

$$
X \in \mathbb{R}^{N \times d},
$$

其中 \(N\) 是样本数，\(d\) 是输入特征维度。隐藏层权重为：

$$
W \in \mathbb{R}^{d \times m},
$$

其中 \(m\) 是隐藏层宽度，即隐藏神经元数量。隐藏层激活为：

$$
F = \sigma(XW),
$$

输出预测为：

$$
\hat Y = FA.
$$

这里：

- \(F \in \mathbb{R}^{N \times m}\)：隐藏层激活矩阵；
- \(A\)：输出层权重；
- \(\hat Y\)：模型预测；
- \(Y\)：训练标签。

第 \(j\) 个隐藏神经元的权重向量记作：

$$
w_j \in \mathbb{R}^d.
$$

它在第 \(i\) 个样本上的激活是：

$$
h_{ij} = \sigma(x_i^\top w_j).
$$

它在所有训练样本上的激活向量是：

$$
h_j =
\begin{bmatrix}
\sigma(x_1^\top w_j)\\
\sigma(x_2^\top w_j)\\
\vdots\\
\sigma(x_N^\top w_j)
\end{bmatrix}.
$$

于是隐藏层激活矩阵可以写成列向量拼接：

$$
F = [h_1,h_2,\dots,h_m] \in \mathbb{R}^{N \times m}.
$$

需要强调的是，权重 \(w_j\) 是同一个；输入 \(x_i\) 不同，所以内积 \(x_i^\top w_j\) 不同；经过非线性函数 \(\sigma\) 以后，同一个神经元会对不同样本产生不同强度的响应。因此一个隐藏神经元本质上是一个“特征探测器”，而它在整个训练集上的激活向量 \(h_j\) 就是一个“特征模式”。

**用户疑问**：虽然权重相同，为什么对不同输入会强弱激活不同？

**回答**：因为同一个线性/非线性探测器面对不同输入，其投影值不同；这不是权重变化，而是响应不同。表征学习要研究的不是单个样本上的标量激活，而是一个神经元在整个数据集上的激活模式 \(h_j\) 是否与标签结构对齐。一个神经元如果只是在个别样本上有大激活，并不一定是好的结构特征；关键是它的整列激活模式是否解释了任务生成规律。

---

## 3. 隐藏层宽度与随机特征过拟合

隐藏层宽度 \(m\) 控制模型拥有多少个隐藏特征坐标。隐藏层激活矩阵为：

$$
F = [h_1,h_2,\dots,h_m] \in \mathbb{R}^{N \times m}.
$$

如果 \(m\) 很大，尤其是 \(m \ge N\)，并且随机初始化得到的随机特征矩阵 \(F\) 秩足够高，那么输出层可以找到某个 \(A\)，使得：

$$
FA \approx Y,
$$

甚至在理想插值情形下：

$$
FA = Y.
$$

这就是“随机特征空间里的插值”。它的关键含义是：训练集上表现好并不自动说明隐藏层已经学到了任务结构。输出层可能只是利用大量随机特征作为基函数，在有限训练点上构造了一个能够穿过标签的线性组合。

直观地说，宽度足够大时，模型有足够多随机坐标轴。输出层可以在这些随机坐标轴上做线性组合，把训练集答案背下来。这是训练集插值，不是结构理解。

因此 Stage I 的低 train loss 不能被简单解释为“模型已经学会算法，只是测试集还没显示出来”。更准确的说法是：模型首先找到了一套在训练点上有效的随机特征读出方式；这种读出可能没有捕获任务的生成结构，所以测试集泛化仍然差。

---

## 4. 三个 Stage 的严格解释：不是数据逐步增加，而是训练动力学阶段

### 4.1 Stage 不是什么

Stage I / II / III 不是数据逐步增大的三个阶段。

它们不是：

- 先喂少量数据、再喂更多数据；
- 先冻结 \(W\) 训练 \(A\)，再冻结 \(A\) 训练 \(W\)；
- 人为把 epoch 均匀切成三段；
- 一套独立于模型内部状态的固定时间标签。

真实训练仍然是端到端反向传播，\(A\) 和 \(W\) 同时更新。Stage 描述的是同一个固定训练集上的训练时间动力学 regime：在不同训练时段，反向传播到隐藏层的梯度结构、隐藏节点之间的相关性、输出层和隐藏层的相对更新速度，以及表征与任务结构的 alignment 会发生系统变化。

数据量变化是另一个实验维度：改变训练集比例或数据分布，重新训练模型，再观察由该训练集诱导出的经验能量景观和表征变化。

### 4.2 Stage 的划分依据

Stage 不是简单按 epoch 人为切分，而是按反向传播到隐藏层的梯度结构变化划分。核心对象是：

$$
G_F = \frac{\partial L}{\partial F}.
$$

可以用以下指标诊断训练处在哪个 regime：

1. \(\lVert \Delta A_t\rVert\) 与 \(\lVert \Delta W_t\rVert\) 的相对大小；
2. \(G_F\) 是否近似噪声；
3. \(G_F\) 是否开始携带标签结构；
4. 隐藏节点相关矩阵是否近似对角；
5. \(-\nabla_{w_j}L\) 是否与 \(\nabla E(w_j)\) 方向对齐；
6. 隐藏神经元是否开始对齐 Fourier / irrep 结构；
7. test accuracy 是否在后期跃升。

严谨地说，论文没有给出一个对所有任务、所有模型、所有优化器通用的精确 transition time，即不能说第几 epoch 必然从 Stage I 进入 Stage II。Stage 是由梯度结构和表征演化定义的动力学区间，而不是硬阈值时间标签。这一点既是理论的边界，也是未来研究方向：如果能把 transition time 与样本量、宽度、初始化尺度、输出层学习速度、regularization 和 optimizer 参数联系起来，将会显著增强理论的预测性。

---

## 5. Stage I: Lazy Learning / Output Layer Memorization

Stage I 中，隐藏层基本不动：

$$
W \approx W_0.
$$

因此隐藏层仍然近似是随机特征：

$$
F \approx \sigma(XW_0).
$$

输出层快速学习：

$$
\hat Y = FA.
$$

训练目标可以近似为固定随机特征上的 ridge regression：

$$
\min_A \; \lVert Y - FA\rVert_F^2 + \lambda \lVert A\rVert_F^2.
$$

Stage I 的核心动力学特征是：

$$
\lVert \Delta A\rVert \gg \lVert \Delta W\rVert.
$$

对应训练曲线表现为：

- train loss 快速下降；
- train accuracy 很快升高；
- test accuracy 仍然低；
- 模型看起来像是在记忆训练集。

这是一种 lazy learning 或 random-feature learning：隐藏层提供随机非线性特征，输出层在这些特征上做线性组合。只要随机特征维度足够大，有限训练集就可以被拟合。

**用户疑问**：如果 Stage I 是完整训练完成，那么测试不好是不是因为训练数据太少？

**回答**：Stage I 不是完整训练结束，而是训练早期。此时模型已经能够拟合训练集，但内部机制主要是输出层在随机特征空间中插值，而不是隐藏层已经学到泛化结构。同一个固定训练集继续训练，后续可能进入 feature learning 并泛化。因此“训练集准确率已高”不等于“训练已经完成”，更不等于“泛化结构已经形成”。

---

## 6. 为什么早期 \(A\) 比 \(W\) 变化快？

设平方损失为：

$$
L = \frac12\lVert FA - Y\rVert_F^2
+ \frac{\lambda}{2}\lVert A\rVert_F^2
+ \frac{\lambda}{2}\lVert W\rVert_F^2,
$$

其中：

$$
F = \sigma(XW).
$$

输出层梯度为：

$$
\nabla_A L = F^\top(FA-Y) + \lambda A.
$$

记 \(a_j\) 为输出层中与第 \(j\) 个隐藏神经元对应的输出权重，即 \(A\) 的第 \(j\) 行或相应输出通道权重。第 \(j\) 个隐藏神经元的梯度为：

$$
\nabla_{w_j}L
=
X^\top\left[((FA-Y)a_j)\odot \sigma'(Xw_j)\right]
+\lambda w_j.
$$

早期初始化时：

- \(F\) 是随机特征；
- \(A\) 随机；
- \(a_j\) 随机；
- \(\sigma'(Xw_j)\) 是随机 gating；
- \(W\) 收到的样本贡献方向不相干，近似噪声。

对 \(A\) 来说，若初始化下 \(FA\) 还很小，可以近似得到：

$$
\nabla_A L \approx -F^\top Y.
$$

只要随机特征矩阵 \(F\) 足够宽、秩足够高，\(F^\top Y\) 就能给出直接降低训练误差的方向。输出层看到的是已经展开好的随机特征坐标，它只需要在线性空间中组合这些坐标。

对 \(W\) 来说：

$$
\nabla_{w_j} L \approx
X^\top\left[(-Y a_j)\odot \sigma'(Xw_j)\right].
$$

这个梯度额外乘了随机输出权重 \(a_j\) 和随机 gating \(\sigma'(Xw_j)\)。因此即使标签 \(Y\) 本身有结构，传到单个隐藏神经元时也可能被随机权重和随机 gating 打散，导致方向性差、信噪比低。

所以早期有效动力学是：

$$
A \text{ learns fast},
$$

$$
W \text{ receives noisy / incoherent gradients}.
$$

这不是训练算法真的先训练 \(A\) 后训练 \(W\)。真实优化仍是端到端更新，只是两个参数块的有效更新速度不同。也不能说 \(W\) 的速度单调变慢。典型情形是：

1. 早期 \(W\) 因反传信号噪声大而慢；
2. 中期 \(A\) 已经吸收标签信息，\(G_F\) 开始携带结构，\(W\) 的有效更新变强；
3. 后期随整体收敛，梯度再次变小。

**用户质疑**：如果模型在收敛，梯度不应该越来越快。

**回答**：普通收敛过程的梯度通常会变小，但 grokking 的特殊性在于隐藏层的有效学习信号一开始不是强，而是噪声。当输出层 \(A\) 在 Stage I 中吸收了标签信息后，传回隐藏层的 \(G_F\) 反而变得更有结构，所以 \(W\) 的有效更新可能先弱后强。这是 hidden-layer effective signal-to-noise 的变化，不是传统单调收敛曲线。

---

## 7. Stage II: Independent Feature Learning

Stage I 后，输出层 \(A\) 不再随机，它已经通过随机特征拟合过训练标签。因此反向传播到隐藏层的梯度：

$$
G_F = \frac{\partial L}{\partial F}
$$

开始携带标签信息。

Stage II 的核心近似是：

$$
G_{F,j}
\text{ mainly depends on }
h_j,
$$

也就是说，第 \(j\) 个隐藏神经元的梯度主要依赖自己的激活模式，而不强依赖其他隐藏节点。此时每个隐藏节点可以近似独立地学习特征。

论文将这一阶段解释为：原始 loss 上对 \(w_j\) 的梯度下降，近似等价于某个有效能量函数 \(E(w_j)\) 上的梯度上升：

$$
-\nabla_{w_j} L \approx c \nabla_{w_j} E(w_j),
$$

其中 \(c>0\) 是尺度系数。于是：

$$
w_j^{t+1}
=
w_j^t - \eta \nabla_{w_j} L
\approx
w_j^t + \eta c \nabla_{w_j}E(w_j).
$$

这句话的精确含义是：

- 模型没有显式计算 \(E\)；
- 模型没有额外做搜索；
- \(E\) 是理论家从原始 loss 的隐藏层梯度中抽象出的有效目标；
- 这个等价是梯度层面的等价，不是说全局 \(L(W,A)=-E(W)\)；
- 该等价依赖 Stage II 的近似条件，例如隐藏节点相关性较弱、反传信号与标签结构对齐、输出层读出已形成有用反馈。

Stage II 是本文最关键的理论桥梁：它把“隐藏特征怎么涌现”从经验观察转化为一个可分析的单神经元能量景观问题。局部能量峰对应可能被神经元学到的特征方向；不同数据分布改变经验能量景观，进而改变模型更可能学到记忆型特征还是结构型特征。

---

## 8. 隐藏节点为什么近似独立：随机初始化 + 宽度集中

**用户质疑**：隐藏节点近似独立到底是因为随机初始化正交，还是因为宽度？

答案是：两个都需要，但作用不同。

### 8.1 随机初始化提供独立来源

不同隐藏节点：

$$
w_1,w_2,\dots,w_m
$$

独立随机初始化。因此：

$$
h_j=\sigma(Xw_j)
$$

在统计上近似独立或弱相关。换句话说，随机初始化让不同神经元最开始从不同方向观察数据。若所有 \(w_j\) 初始化完全相同，那么无论宽度多大，所有 \(h_j\) 都相同，宽度不会产生有效多样性。

### 8.2 宽度提供集中性

考虑经验神经切线或随机特征 Gram 的一个典型对象：

$$
\frac{1}{m}FF^\top
=
\frac{1}{m}\sum_{j=1}^{m}h_jh_j^\top.
$$

这是很多独立随机矩阵的平均。大宽度下，根据集中现象：

$$
\frac{1}{m}FF^\top
\approx
\mathbb{E}[hh^\top].
$$

交叉项相对变小，相关矩阵趋向简单结构，例如 identity 的倍数或近似对角。用一句话概括：

$$
\text{随机初始化}
\Rightarrow
\text{交叉项期望小},
$$

$$
\text{大宽度}
\Rightarrow
\text{交叉项以高概率真的小}.
$$

没有随机初始化，宽度大也可能所有节点相同；没有足够宽度，随机初始化也可能偶然相关较强。Stage II 的独立特征学习近似依赖二者共同成立。

一个可测指标是隐藏特征相关矩阵的 off-diagonal ratio。令 \(\tilde F\) 为中心化或标准化后的隐藏激活，则：

$$
\frac{
\left\lVert \tilde F^\top \tilde F-\mathrm{diag}(\tilde F^\top \tilde F)\right\rVert_F
}{
\left\lVert \tilde F^\top \tilde F\right\rVert_F
}.
$$

如果该量小，说明隐藏特征之间的相关性较弱，独立特征学习近似更可信；如果该量大，则 Stage II 近似不成立或正在进入 Stage III。
### 8.X 大宽度不是让一对神经元更独立，而是让整体随机特征统计量集中

之前容易混淆的一点是：**大宽度并不是让两个已经给定的隐藏神经元变得更独立**。

更准确的逻辑是：

$$
\boxed{
\text{independence comes from iid initialization; decoupling comes from width-induced concentration.}
}
$$

即：

$$
\boxed{
\text{随机初始化产生隐藏节点作为随机变量的独立性。}
}
$$

$$
\boxed{
\text{大宽度让许多独立随机特征的平均统计量集中到稳定平均场。}
}
$$

设固定训练集：

$$
X\in \mathbb R^{n\times d},
$$

隐藏层宽度为：

$$
m,
$$

隐藏权重为：

$$
W=[w_1,w_2,\dots,w_m]\in \mathbb R^{d\times m}.
$$

第 \(j\) 个隐藏节点在所有样本上的激活向量为：

$$
h_j=\sigma(Xw_j)\in \mathbb R^n,
$$

隐藏层表示矩阵为：

$$
F=[h_1,h_2,\dots,h_m]\in \mathbb R^{n\times m}.
$$

随机初始化给出：

$$
w_1,w_2,\dots,w_m \overset{iid}{\sim} P_W.
$$

因此：

$$
h_1,h_2,\dots,h_m
$$

作为随机向量也是 iid 或近似 iid。但是需要强调：

$$
\text{statistical independence} \neq \text{orthogonality}.
$$

也就是说，随机初始化给出的是概率意义上的独立，不意味着一次具体初始化中：

$$
h_i^\top h_j=0.
$$

大宽度的作用不是让任意一对 \(h_i,h_j\) 更正交，而是让许多随机特征共同构成的平均矩阵更稳定。考虑：

$$
K_m=\frac{1}{m}FF^\top
=
\frac{1}{m}\sum_{j=1}^{m}h_jh_j^\top.
$$

因为每个 \(h_jh_j^\top\) 是由独立随机隐藏节点产生的随机矩阵，所以：

$$
K_m
=
\frac{1}{m}\sum_{j=1}^{m}h_jh_j^\top
\approx
\mathbb E_w[\sigma(Xw)\sigma(Xw)^\top]
=
K_\infty.
$$

随机波动大致满足：

$$
K_m-K_\infty=O_{\mathbb P}\left(\frac1{\sqrt m}\right).
$$

这就是所谓的“稳定”：

$$
\boxed{
K_m \text{ 不再强烈依赖某一次随机初始化，而是接近确定的无限宽随机特征核 }K_\infty。
}
$$

同时，单个隐藏节点对整体平均核的贡献只有：

$$
\frac1m h_jh_j^\top,
$$

量级是：

$$
O\left(\frac1m\right).
$$

因此当 \(m\) 很大时，第 \(j\) 个节点对整体平均背景的反作用很小，可以近似把其他所有节点看成一个稳定背景。这就是 Stage II 中“独立特征学习”的真正含义：

$$
\Delta w_j
=
\Phi(w_j;w_1,\dots,w_m).
$$

在大宽度稳定背景下近似变成：

$$
\Delta w_j
\approx
\Phi(w_j;K_\infty).
$$

即：

$$
\boxed{
\text{每个隐藏节点不是物理上完全独立，而是在稳定平均场中近似解耦。}
}
$$

这里必须明确区分 \(n\) 和 \(m\)：

| 符号 | 含义 | 增大后稳定什么 | 在论文中的角色 |
| --- | --- | --- | --- |
| \(n\) | 训练样本数 | 经验能量 \(E_S\) 接近真实任务能量 \(E_{\mathcal D}\) | 决定记忆峰 / 结构峰 |
| \(m\) | 隐藏层宽度 / 随机特征数 | 随机特征核 \(K_m\) 接近 \(K_\infty\) | 支撑 Stage II 的独立节点动力学 |

不能把宽度增加写成“提供更多数据”。正确表述是：

$$
\boxed{
\text{更大宽度提供更多随机特征，不提供更多训练数据。}
}
$$

### 8.Y \(K_m=\frac1mFF^\top\) 的含义：样本-样本相似性，而不是神经元独立性

$$
K_m=\frac1mFF^\top
$$

不是用来直接检查“两个隐藏神经元是否独立”的矩阵。它的真正含义是：

$$
\boxed{
K_m \text{ 是隐藏层随机特征诱导出的样本-样本相似性矩阵。}
}
$$

因为：

$$
F\in\mathbb R^{n\times m},
$$

其中第 \(i\) 行：

$$
F_i=
[\sigma(x_i^\top w_1),\sigma(x_i^\top w_2),\dots,\sigma(x_i^\top w_m)]
$$

表示第 \(i\) 个样本在所有隐藏节点上的表示。于是：

$$
(K_m)_{ab}
=
\frac1m(FF^\top)_{ab}
=
\frac1mF_a\cdot F_b.
$$

展开为：

$$
(K_m)_{ab}
=
\frac1m\sum_{j=1}^{m}
\sigma(x_a^\top w_j)\sigma(x_b^\top w_j).
$$

它表示：

$$
\boxed{
\text{样本 }x_a\text{ 和 }x_b\text{ 在隐藏表示空间中的平均内积相似度。}
}
$$

对角线：

$$
(K_m)_{aa}
=
\frac1m\sum_{j=1}^{m}
\sigma(x_a^\top w_j)^2
$$

表示样本 \(x_a\) 经过隐藏层后的平均表示长度平方。非对角线：

$$
(K_m)_{ab},\quad a\ne b
$$

表示样本 \(x_a,x_b\) 在隐藏空间中有多相似。因此，\(K_m\) 的作用是刻画：

$$
\boxed{
\text{随机隐藏层给训练样本制造了怎样的几何结构。}
}
$$

Stage I 中，隐藏层仍近似随机：

$$
F\approx F_0.
$$

输出层在随机特征上学习：

$$
\hat Y=FA.
$$

输出层能否在随机特征上快速拟合训练集，取决于随机隐藏表示给样本制造出的几何结构，也就是：

$$
K_m=\frac1mFF^\top.
$$

如果 \(K_m\) 让不同样本在隐藏空间中足够可分，输出层就能在这些随机特征上插值训练标签。Stage II 中，大宽度让：

$$
K_m\approx K_\infty,
$$

使随机隐藏表示的整体背景可分析，从而支持把单个隐藏节点拿出来分析其 feature learning 动力学。

精确总结是：

$$
\boxed{
K_m \text{ 的角色是刻画样本几何与随机特征平均背景，不是直接刻画隐藏节点之间是否相似。}
}
$$

### 8.Z \(FF^\top\) 与 \(F^\top F\)：样本几何 vs. 特征冗余

同一个隐藏表示矩阵：

$$
F\in \mathbb R^{n\times m}
$$

可以从两个方向相乘。

**1. \(FF^\top\)：样本-样本相似性**

$$
FF^\top\in\mathbb R^{n\times n}.
$$

其第 \((a,b)\) 项：

$$
(FF^\top)_{ab}
=
F_a\cdot F_b
=
\sum_{j=1}^{m}F_{aj}F_{bj}.
$$

含义：

$$
\boxed{
\text{第 }a\text{ 个样本和第 }b\text{ 个样本在所有隐藏节点上的表示有多像。}
}
$$

它回答的问题是：隐藏层把训练样本映射成了怎样的几何结构？因此 \(FF^\top\) 主要用于理解 Stage I / Stage II 中随机隐藏表示的样本几何、随机特征 kernel 和输出层插值能力。

**2. \(F^\top F\)：隐藏节点-隐藏节点相似性**

$$
F^\top F\in\mathbb R^{m\times m}.
$$

其第 \((i,j)\) 项：

$$
(F^\top F)_{ij}
=
f_i^\top f_j
=
\sum_{a=1}^{n}F_{ai}F_{aj}.
$$

含义：

$$
\boxed{
\text{第 }i\text{ 个隐藏节点和第 }j\text{ 个隐藏节点在所有训练样本上的激活模式有多像。}
}
$$

它回答的问题是：不同隐藏神经元是不是在学重复的东西？因此 \(F^\top F\) 主要用于理解 Stage III 中的 feature redundancy 和 similar-feature repulsion。

| 矩阵 | 维度 | 比较对象 | 含义 | 在论文中的作用 |
| --- | ---: | --- | --- | --- |
| \(FF^\top\) | \(n\times n\) | 样本 vs 样本 | 样本在隐藏空间中的相似性 | 随机特征 kernel，解释 Stage I/II |
| \(F^\top F\) | \(m\times m\) | 节点 vs 节点 | 隐藏节点激活模式的相似性 | 特征冗余，解释 Stage III repulsion |

最后可以压缩成一句：

$$
\boxed{
FF^\top=\text{样本几何};\qquad F^\top F=\text{特征冗余。}
}
$$

---

## 9. 能量函数 \(E(w)\) 与 nonlinear CCA

定义隐藏神经元激活：

$$
h_w = \sigma(Xw).
$$

令 \(P\) 为样本中心化投影，例如：

$$
P = I - \frac{1}{N}\mathbf{1}\mathbf{1}^\top.
$$

中心化后：

$$
\tilde h_w = P h_w,
$$

标签中心化为：

$$
\tilde Y = P Y.
$$

可以用一个简化形式理解能量函数：

$$
E(w)
=
\frac12
\left\lVert \tilde Y^\top \tilde h_w\right\rVert^2.
$$

等价写成：

$$
E(w)
=
\frac12
\tilde h_w^\top \tilde Y \tilde Y^\top \tilde h_w.
$$

解释如下：

- \(\tilde Y\tilde Y^\top\) 是标签结构诱导出的样本相似性 kernel；
- \(E(w)\) 衡量隐藏神经元激活模式 \(\tilde h_w\) 是否与标签结构对齐；
- 如果 \(\tilde h_w\) 与标签结构无关，能量低；
- 如果 \(\tilde h_w\) 能解释标签结构，能量高。

对这个简化能量求梯度。记：

$$
D_w=\mathrm{diag}(\sigma'(Xw)).
$$

由于：

$$
\nabla_w h_w = X^\top D_w,
$$

可得到：

$$
\nabla_w E
=
X^\top D_w P\tilde Y\tilde Y^\top P h_w.
$$

隐藏层 loss 梯度的一般形式是：

$$
\nabla_w L =
X^\top D_w G_w,
$$

其中 \(G_w\) 是反传到该神经元激活向量上的有效信号。如果 Stage II 中：

$$
G_w \propto -P\tilde Y\tilde Y^\top P h_w,
$$

则：

$$
-\nabla_w L \propto \nabla_w E.
$$

这就是等效证明的核心：不是模型显式最大化相关性，而是原始 loss 的反向传播在 Stage II 近似产生了同一个方向。

### nonlinear CCA 解释

普通 CCA 问：

$$
\max_{u,v}\mathrm{Corr}(Xu,Yv),
$$

即找输入方向和标签方向，使相关性最大。这里输入侧不是线性投影 \(Xu\)，而是非线性特征：

$$
h_w = \sigma(Xw).
$$

因此 nonlinear CCA 可以理解成：

$$
\max_w \left\lVert \tilde Y^\top \tilde h_w\right\rVert^2,
$$

即找一个非线性输入特征，使它和标签结构最对齐。

必须写清楚三点：

1. 这不是模型额外做 Pearson 相关性分析；
2. 这是从原始 loss 的反向传播梯度中推导出的有效目标；
3. “相关性”在这里指隐藏激活模式与标签结构的对齐，而不是外加统计步骤。

因此 nonlinear CCA 在本文中是机制解释，不是后处理技巧。它把隐藏神经元的 feature emergence 表述为：在非线性特征族 \(\{\sigma(Xw):w\in\mathbb{R}^d\}\) 中，寻找与标签诱导 kernel 最对齐的方向。

---

## 10. 什么是高能量特征？

高能量特征不是“激活值大”：

$$
\text{高能量特征}
\neq
\text{激活值大}.
$$

更准确地说：

$$
\text{高能量特征}
=
\text{隐藏神经元在所有样本上的激活模式，与标签结构高度对齐}.
$$

如果 \(E(w)\) 高，说明：

$$
h_w=\sigma(Xw)
$$

能够被输出层高效用来解释 \(Y\)。注意这里的对象不是单样本激活，而是整列激活向量 \(h_w\)。某个神经元在一个样本上特别兴奋并不重要；重要的是它在所有样本上的响应模式是否捕捉了标签变化规律。

在 modular addition 任务中：

$$
y=a+b \pmod p,
$$

记忆型特征可能是：

$$
\mathbf{1}\{(a,b)=(a_0,b_0)\},
$$

即只识别某一个训练 pair。结构型特征则可能是 Fourier / group representation，例如：

$$
\cos\left(\frac{2\pi k a}{p}\right),
\quad
\sin\left(\frac{2\pi k a}{p}\right),
$$

以及 \(b\) 上相应的频率组合，或者更一般的 irreducible representation。

直觉上，高能量不是“神经元兴奋”，而是“神经元的激活模式能不能抓住生成标签的结构”。如果一个神经元的响应只对应训练集中某几个孤立样本，它可能有助于训练集拟合，但不一定能泛化；如果它对应 Fourier frequency 或 irrep，它能压缩地表示整个群运算规律，因此更有泛化价值。

---

## 11. Stage III: Interactive Feature Learning

Stage II 的独立近似会在训练一段时间后失效。原因是隐藏节点已经不再随机，它们可能学到相似特征，因此：

$$
h_i^\top h_j
$$

不再可以忽略。

完整 loss 为：

$$
L=\frac12\lVert FA-Y\rVert_F^2.
$$

它依赖整个隐藏激活矩阵：

$$
F=[h_1,h_2,\dots,h_m].
$$

当多个隐藏节点共同拟合同一个输出目标时，它们不再是独立最大化各自能量的个体，而是通过输出层、Gram 矩阵、残差和 inverse Gram 发生耦合。此时：

- 相似特征会产生冗余；
- 冗余特征在最优读出中会互相竞争；
- 反向传播信号会自动减弱已解释结构；
- 尚未解释的结构在残差中保留下来；
- 新的隐藏节点更倾向学习 missing features。

Stage III 因此解释了两个后期现象：相似神经元之间的 repulsion，以及 top-down modulation 下对缺失结构的补齐。

---

## 12. 相似神经元为什么会彼此推离：数学推导

### 12.Z 从 Independent Feature Learning 到 Interactive Feature Learning：从独立找特征到协同补全特征集合

Stage II 中，每个隐藏节点近似独立地沿能量函数爬升：

$$
-\nabla_{w_j}L\approx c\nabla E(w_j).
$$

因此多个节点可能被同一个高能量特征吸引。例如在 modular addition 中，多个节点可能都学到同一个 Fourier frequency：

$$
f_1\approx f_2\approx f_3.
$$

但泛化需要的是一组完整、互补的结构特征，而不是同一个特征的多份拷贝。当多个节点学到相似结构后：

$$
F^\top F
$$

的非对角项变大，Stage II 的独立近似失效。此时输出层和 residual 开始产生交互：

1. 相似节点通过 inverse Gram 中的负交叉项被去冗余；
2. 已经被解释的结构产生较小残差；
3. 未被解释的结构继续产生较大梯度；
4. 节点逐渐分工，形成更完整的特征集合。

因此 Stage III 的本质是：

$$
\boxed{
\text{从“很多节点各自找到高能量特征”，转向“节点之间协调，形成互补特征集合”。}
}
$$

用 grokking 的语言：

- Stage I：输出层在随机特征上记忆训练集；
- Stage II：隐藏节点独立学习高能量结构特征；
- Stage III：隐藏节点去冗余、补残差、形成完整结构表征；
- 测试泛化在特征集合足够完整后突然跃升。

### 12.X 为什么线性输出层也会让隐藏节点互相影响？

一个常见误解是：输出层是线性连接，所以每个隐藏节点应该独立反向传播，两个节点为什么会互相影响？

关键在于，模型预测不是每个节点单独决定，而是所有隐藏节点共同决定：

$$
\hat Y=FA=\sum_{j=1}^{m}f_j a_j.
$$

loss 是：

$$
L=\frac12\left\lVert Y-\sum_{j=1}^{m}f_j a_j\right\rVert^2.
$$

因此它不是每个节点各自有一个独立 loss，而是所有节点共同解释同一个目标 \(Y\)。残差：

$$
R=FA-Y
$$

包含所有隐藏节点的贡献：

$$
R=\sum_{l=1}^{m}f_l a_l-Y.
$$

对第 \(j\) 个隐藏激活 \(f_j\) 的梯度为：

$$
\nabla_{f_j}L = R a_j^\top.
$$

虽然求的是第 \(j\) 个节点的梯度，但 \(R\) 本身包含所有 \(f_l\)。因此隐藏节点之间通过共同残差产生耦合。

核心句是：

$$
\boxed{
\text{隐藏节点互相影响，不是因为输出层非线性，而是因为它们共同竞争解释同一个残差。}
}
$$

### 12.1 Inverse Gram 中的负交叉项如何产生去冗余信号

固定隐藏表示 \(F\) 时，输出层 ridge regression 的解为：

$$
A^*=(F^\top F+\lambda I)^{-1}F^\top Y.
$$

这里出现关键矩阵：

$$
F^\top F.
$$

其第 \((i,j)\) 项为：

$$
(F^\top F)_{ij}=f_i^\top f_j.
$$

如果两个隐藏节点相似：

$$
f_i^\top f_j>0,
$$

则 \(F^\top F\) 有正的 off-diagonal 项。只看两个节点：

$$
F=[f_1,f_2].
$$

则：

$$
F^\top F+\lambda I
=
\begin{bmatrix}
f_1^\top f_1+\lambda & f_1^\top f_2\\
f_2^\top f_1 & f_2^\top f_2+\lambda
\end{bmatrix}.
$$

记：

$$
a=f_1^\top f_1+\lambda,
$$

$$
c=f_2^\top f_2+\lambda,
$$

$$
b=f_1^\top f_2.
$$

则：

$$
G=
\begin{bmatrix}
a & b\\
b & c
\end{bmatrix}.
$$

其逆为：

$$
G^{-1}
=
\frac1{ac-b^2}
\begin{bmatrix}
c & -b\\
-b & a
\end{bmatrix}.
$$

如果：

$$
b=f_1^\top f_2>0,
$$

则：

$$
(G^{-1})_{12}
=
-\frac{b}{ac-b^2}<0.
$$

因此：

$$
\boxed{
\text{特征正相关}
\Rightarrow
F^\top F\text{ 有正交叉项}
\Rightarrow
(F^\top F+\lambda I)^{-1}\text{ 有负交叉项。}
}
$$

反传信号中会出现类似：

$$
F(F^\top F+\lambda I)^{-1}.
$$

看第 1 列：

$$
\left[F(F^\top F+\lambda I)^{-1}\right]_{\cdot 1}
=
q_{11}f_1+q_{21}f_2.
$$

其中：

$$
q_{21}<0.
$$

所以：

$$
q_{11}f_1+q_{21}f_2
=
q_{11}f_1-|q_{21}|f_2.
$$

这说明第 1 个节点收到的有效信号里，会减去一部分与第 2 个节点相似的方向。这就是 repulsion 的数学来源：

$$
\boxed{
\text{反传梯度中出现了“减去相似特征”的项。}
}
$$

也可以把链条写成：

$$
\text{feature similarity}
\Rightarrow
\text{positive Gram off-diagonal}
\Rightarrow
\text{negative inverse-Gram coefficient}
\Rightarrow
\text{gradient subtracts similar feature}
\Rightarrow
\text{feature specialization}.
$$

必须写清楚：

- 这不是物理力；
- 不是模型主动讨厌相似节点；
- 而是 ridge / inverse Gram 对共线特征进行重叠解释量扣除；
- 因为隐藏特征本身可学习，所以这个扣除项会通过反向传播推动隐藏节点分化。

### 12.Y Repulsion 的动力学直觉：模型没有意识，梯度场自动编码了冗余特征的低边际收益

不要把 repulsion 理解成模型“意识到两个节点相似”。模型没有意识。更准确的说法是：

$$
\boxed{
\text{模型没有意识；loss 的梯度场自动把“重复特征的边际收益低”编码进了更新方向。}
}
$$

如果两个隐藏节点：

$$
f_1\approx f_2,
$$

那么：

$$
f_1a_1+f_2a_2
\approx
f_1(a_1+a_2).
$$

虽然形式上有两个节点，但它们功能上只提供了一个方向：

$$
\mathrm{span}(f_1,f_2)\approx \mathrm{span}(f_1).
$$

如果目标 \(Y\) 中还有未被解释的结构，那么继续让 \(f_2\) 更像 \(f_1\) 并不会显著降低 loss，因为这个方向已经被覆盖了。

设：

$$
Y=\text{feature A}+\text{feature B}.
$$

如果 \(f_1\) 和 \(f_2\) 都在解释 feature A，则预测可能已经覆盖 A：

$$
\hat Y\approx \text{feature A}.
$$

残差：

$$
R=Y-\hat Y\approx \text{feature B}.
$$

反向传播来自残差，所以后续有效梯度更倾向于让某些节点去解释 feature B，而不是继续重复 feature A。

因此，repulsion 更准确地说是：

$$
\boxed{
\text{冗余方向边际收益下降，未解释残差方向边际收益更高。}
}
$$

从线性空间角度，如果：

$$
f_1=f_2=f,
$$

则：

$$
\mathrm{span}(f_1,f_2)=\mathrm{span}(f),
$$

模型只获得一维解释能力。如果让：

$$
f_2=f+\epsilon u,
$$

且 \(u\) 对应残差中尚未解释的结构，那么：

$$
\mathrm{span}(f_1,f_2)
$$

从近似一维扩展到二维，模型能解释更多 \(Y\) 的结构，loss 会下降更多。因此所谓“推离”的动力是：

$$
\boxed{
\text{重复节点不增加表达维度；分化节点增加表达维度，因此更能降低 loss。}
}
$$

Stage III 的 repulsion 可以看成一种隐式、动态的去冗余机制，类似 Gram-Schmidt 中从一个向量中减掉与另一个向量重叠的部分：

$$
f_2^\perp
=
f_2-\frac{f_2^\top f_1}{\lVert f_1\rVert^2}f_1.
$$

但神经网络不是显式执行 Gram-Schmidt，而是通过：

$$
(F^\top F+\lambda I)^{-1}
$$

中的负交叉项，以及残差驱动的反向传播，自动产生类似的去重效果。

总结为：

$$
\boxed{
\text{相似节点之所以分化，是因为在共同最小化同一个 loss 时，重复特征对降低残差的边际贡献会被 inverse Gram / 投影机制扣掉，而未解释的残差方向会产生更大的梯度收益。}
}
$$
---

## 13. Top-down modulation：为什么后期会学“缺失特征”？

用 residual 可以清楚解释 top-down modulation。假设目标可以分解为多个结构成分：

$$
Y = Y_1+Y_2+Y_3.
$$

模型已经学到了前两个结构：

$$
\hat Y \approx Y_1+Y_2.
$$

则残差为：

$$
R=Y-\hat Y\approx Y_3.
$$

反向传播信号主要来自残差：

$$
G_F \sim R.
$$

因此后续隐藏层收到的主要信号是：

$$
\text{继续学习还没有被解释的结构}.
$$

这就是 top-down modulation：

- 已经解释的部分梯度变弱；
- 没有解释的部分继续产生强梯度；
- 有效能量函数被 residual 改写；
- 后续神经元更倾向于学习 missing irreps / missing features。

输出层不是有意识地告诉隐藏层缺什么，而是残差本身告诉反向传播还缺什么。在 group arithmetic 中，如果某些 Fourier modes 或 irreps 已经被表示，那么这些成分在残差中减少；尚未被表示的 modes 在残差中保留，从而塑造后续神经元的有效能量景观。

---

## 14. 能量峰到底是什么？和 loss 有什么关系？

必须区分：

$$
E(w)\text{ 的峰}
\neq
L(W,A)\text{ 的峰}.
$$

原始训练是：

$$
\min L(W,A).
$$

Stage II 单个隐藏神经元的有效动力学是：

$$
\max E(w_j).
$$

二者通过梯度关系连接：

$$
-\nabla_{w_j}L \approx c\nabla_{w_j}E(w_j).
$$

因此：

- \(E\) 的局部最大值对应隐藏神经元可能学到的特征方向；
- 它不是整个原始 loss 的全局最小值；
- 非凸优化通常只能到 local maximum；
- 不需要也不能保证找到全局最高峰；
- 初始化、学习率、噪声、样本分布、吸引盆共同决定模型走向哪个局部峰。

局部能量峰可以定义为：

$$
\nabla E(w)=0,
$$

且：

$$
\nabla^2 E(w)\preceq 0.
$$

在 group arithmetic 任务中，论文证明某些局部极值对应群的 irreducible representations；在 modular addition 中对应 Fourier basis。这个结论的重要性在于，它把“泛化特征”从直觉概念变成了可数学识别的对象：如果任务本身由群运算生成，那么能表达群结构的 irrep / Fourier directions 就是可泛化表征的自然基。

---

## 15. 记忆峰与结构峰：数学上怎么区分？

### 15.1 结构峰

结构峰不是主观定义，而是在 group arithmetic 任务中由群表示理论给出。考虑群运算任务：

$$
(g,h)\mapsto gh.
$$

若数据均匀覆盖群运算结构，则 \(E(w)\) 的局部最大值会落在群的 irreducible representation / Fourier 子空间中。对 Abelian group / modular addition：

$$
y=a+b \pmod p,
$$

结构特征对应 Fourier frequencies。

因此：

$$
\text{结构峰}
=
E(w)\text{ 的局部最大值，且对应 irrep / Fourier representation}.
$$

结构峰能泛化，因为它抓住的是群运算的生成结构，而不是某几个训练样本。换句话说，它把很多样本背后的共同生成规律压缩到少量可组合特征中。

### 15.2 记忆峰

记忆峰对应 sample-specific / pair-specific 表征。例如：

$$
\mathbf{1}\{(a,b)=(a_0,b_0)\},
$$

或者 one-hot pair / target slice indicator。

偏分布例子：只采一个 target slice：

$$
(g,g^{-1}h)\mapsto h.
$$

这种数据没有完整群结构，只包含一个 target slice。在这种情况下，能量函数的全局最优可以变成记忆型解，例如集中到某个 pair / sample coordinate：

$$
w \propto (e_{g^*},e_{g^{*-1}h}).
$$

这说明：

$$
\text{偏数据}
\Rightarrow
\text{sample-specific feature gets highest energy}.
$$

因此：

$$
\text{记忆峰}
=
E(w)\text{ 的最优或局部最优落在样本 / pair indicator 坐标上}.
$$

需要区分两个层面的记忆：

1. Stage I 的记忆是输出层在随机特征上插值；
2. Stage II 的记忆峰是隐藏神经元真的学到了 non-generalizable feature。

二者相关，但不是同一个概念。Stage I 可以在隐藏层不动时通过 \(A\) 完成训练集插值；Stage II 的记忆峰则意味着隐藏层参数也被推向了样本特异方向。后者更危险，因为它会把 non-generalizable structure 固化进表征。

---

## 16. 数据量和数据分布如何改变能量景观？

经验能量可以写成：

$$
E_S(w)
=
\frac{1}{|S|}
\sum_{(x_i,y_i)\in S}
e(w;x_i,y_i).
$$

population energy 为：

$$
E_{\mathcal D}(w)
=
\mathbb E_{(x,y)\sim \mathcal D}
[e(w;x,y)].
$$

有限训练集上：

$$
E_S(w)=E_{\mathcal D}(w)+\epsilon_S(w),
$$

其中：

$$
\epsilon_S(w)
$$

是采样误差和分布偏差。

必须强调：

- 对于固定训练集 \(S\)，Stage II 的能量景观基本固定；
- 训练过程是在这个景观上移动；
- 山峰不是随 epoch 从记忆峰变成结构峰；
- 改变训练数据量或数据分布，才会改变 \(E_S(w)\) 的形状。

### 16.1 均匀且足够多的数据

如果 \(S\) 均匀覆盖真实任务结构：

$$
E_S(w)\approx E_{\mathcal D}(w),
$$

采样噪声变小，真实结构峰稳定存在。在 group arithmetic 中，结构峰对应 irreps / Fourier features。

此时模型更可能学到能解释整个任务生成规律的方向。训练样本越均匀覆盖群结构，经验能量越接近 population energy，spurious sample-specific directions 越难稳定占优。

### 16.2 少量数据

少量数据导致：

$$
\epsilon_S(w)\text{ 大}.
$$

因此：

- 结构峰可能不稳定；
- 经验景观可能出现 spurious / sample-specific 局部峰；
- 不能保证泛化峰存在或吸引盆足够大。

严谨表述是：

$$
\text{少数据}
\not\Rightarrow
\text{必然记忆},
$$

而是：

$$
\text{少数据}
\Rightarrow
\text{结构峰稳定性不能保证}.
$$

少数据仍可能泛化，如果采样恰好覆盖了关键结构；但理论上，经验能量相对于 population energy 的偏差更大，记忆峰或 shortcut 峰更容易竞争。

### 16.3 偏分布数据

偏分布比少数据更强。如果只采某一个 target slice，数据本身不包含完整结构，因此 \(E_S(w)\) 会奖励 sample-specific / target-specific 特征。此时可以证明记忆解成为全局最优或主导局部最优。

因此：

$$
\text{偏分布}
\Rightarrow
\text{记忆峰可能成为 global optimum}.
$$

这说明数据质量不等价于数据数量。即使样本数不少，如果覆盖方式破坏了任务结构，经验能量仍可能被错误地塑造成偏向记忆或 shortcut 的景观。

### 16.4 “两座山此消彼长”的严谨解释

不要把它写成训练过程中山真的在变。更严谨的说法是：对于不同数据集 \(S_1,S_2\)，对应不同经验能量函数：

$$
S_1\Rightarrow E_{S_1}(w),
$$

$$
S_2\Rightarrow E_{S_2}(w).
$$

当数据少或偏时，可能出现：

$$
E_S(w_{\text{mem}})
>
E_S(w_{\text{gen}}),
$$

或者 \(w_{\text{gen}}\) 的局部峰不稳定。

当数据均匀且足够多时：

$$
E_S(w_{\text{gen}})
\approx
E_{\mathcal D}(w_{\text{gen}}),
$$

而记忆特征在不同样本上的贡献互相抵消：

$$
E_S(w_{\text{mem}})
\downarrow.
$$

因此更可能有：

$$
E_S(w_{\text{gen}})
>
E_S(w_{\text{mem}}).
$$

更精确地说：

- 结构峰高度更稳定；
- 结构峰吸引盆变大；
- 记忆峰吸引盆变小；
- 有些记忆峰消失；
- 中间区域可能两类峰共存。

这给数据工程一个机制化解释：增加数据不是为了让模型“见过更多答案”这么简单，而是为了让经验能量景观更接近真实任务结构，使反向传播更可能把隐藏层推向结构峰。

---

## 17. 本次讨论中的关键质疑与澄清

### Q1. Stage 是否只是人为解释？

不是纯粹人为叙事。它可以通过以下指标检验：

- \(\lVert \Delta A\rVert/\lVert \Delta W\rVert\)；
- \(G_F\) 与标签结构的 alignment；
- 隐藏节点相关矩阵是否近似对角；
- \(-\nabla L\) 与 \(\nabla E\) 的 cosine similarity；
- 隐藏特征是否对齐 Fourier / irrep；
- train/test accuracy 曲线是否出现 delayed generalization。

但也要承认：论文没有给出通用 transition time；三阶段是动力学 regime，不是严格 epoch 分段。

### Q2. Stage 是否对应数据逐步增大？

不是。Stage 是固定训练集上的训练时间阶段。数据量变化是另一个实验维度，用来比较不同训练集诱导的 \(E_S(w)\) 如何变化。

### Q3. \(A\) 和 \(W\) 是不是被分开训练？

不是。它们端到端同时更新。但早期 \(\lVert \Delta A\rVert\gg \lVert \Delta W\rVert\)，所以动力学上近似表现为输出层先学习。

### Q4. 为什么 \(A\) 早期更快？

因为：

$$
\nabla_A L=F^\top(FA-Y)+\lambda A
$$

可以直接利用随机特征矩阵 \(F\) 拟合标签；而：

$$
\nabla_{w_j}L
=
X^\top\left[((FA-Y)a_j)\odot \sigma'(Xw_j)\right]
+\lambda w_j
$$

还要乘随机输出权重 \(a_j\) 和随机 gating \(\sigma'(Xw_j)\)，早期方向噪声大。

### Q5. \(W\) 的速度是否越来越慢？

不一定。早期 \(W\) 因反传噪声而慢；Stage I 后 \(G_F\) 携带标签结构，\(W\) 的有效更新可能变强；后期再随收敛变慢。

### Q6. 宽度为什么导致隐藏节点独立？

随机初始化让不同隐藏节点交叉项期望小；大宽度让这些交叉项通过集中现象以高概率小。因此大宽度和随机初始化共同导致 Stage II 中的近似独立。

### Q7. 能量函数和原 loss 是什么关系？

不是全局目标相等，而是 Stage II 中隐藏神经元梯度方向等效：

$$
-\nabla_{w_j}L\approx c\nabla E(w_j).
$$

因此原 loss 上的梯度下降，可以解释为单神经元在有效能量函数上的梯度上升。该解释依赖 Stage II 条件，不能无条件推广到所有训练阶段。

### Q8. 高能量是否就是相关性高？

不是额外相关性分析，而是反传梯度诱导出的 nonlinear CCA 对齐。高能量表示隐藏激活模式能解释标签结构。它对应的是激活向量与标签诱导 kernel 的对齐，而不是单个样本激活值大。

### Q9. 相似神经元为什么排斥？

因为隐藏特征相关时，Gram 矩阵有正 off-diagonal，inverse Gram 中对应交叉项为负，反传信号会扣掉相似特征，从而推动节点分化。最简单的 \(2\times2\) 推导中，若：

$$
G=
\begin{bmatrix}
a & b\\
b & c
\end{bmatrix},
\quad b>0,
$$

则：

$$
(G^{-1})_{12}
=
-\frac{b}{ac-b^2}<0.
$$

这就是 feature repulsion 的线性代数来源。

### Q10. 更多数据如何影响泛化？

更多且更均匀的数据让 \(E_S(w)\) 接近 population energy \(E_{\mathcal D}(w)\)，结构峰稳定存在；少量或偏分布数据会让经验景观偏离真实结构，记忆峰或 shortcut 峰可能主导。关键不是“数据多”本身，而是数据是否覆盖真实生成结构。
### Q: 大宽度为什么与隐藏节点独立有关？

大宽度本身不让一对给定隐藏节点更独立。隐藏节点作为随机变量的独立性来自 iid 随机初始化。大宽度的作用是让许多 iid 随机隐藏节点共同形成的统计量，例如：

$$
K_m=\frac1mFF^\top
$$

集中到稳定期望：

$$
K_\infty=\mathbb E_w[\sigma(Xw)\sigma(Xw)^\top].
$$

从而让每个节点面对一个稳定平均背景，并且单个节点对整体背景的贡献只有 \(O(1/m)\)，因此 Stage II 中可以近似解耦。

### Q: \(K_m\) 是用来衡量隐藏节点独立性的吗？

不是。\(K_m=\frac1mFF^\top\) 衡量的是样本-样本相似性，即不同训练样本经过隐藏层后的表示有多像。它用于理解随机隐藏表示的样本几何、Stage I 的随机特征插值和 Stage II 的稳定平均背景。隐藏节点之间是否相似应看 \(F^\top F\)。

### Q: \(FF^\top\) 和 \(F^\top F\) 有什么区别？

$$
FF^\top
$$

比较样本和样本，是样本几何；

$$
F^\top F
$$

比较隐藏节点和隐藏节点，是特征冗余。

Stage I / II 主要关心 \(FF^\top\)；Stage III 的相似节点 repulsion 主要关心 \(F^\top F\)。

### Q: 线性输出层为什么会让隐藏节点互相影响？

因为输出是所有节点贡献之和：

$$
\hat Y=\sum_j f_j a_j.
$$

loss 的残差：

$$
R=\hat Y-Y
$$

包含所有节点的贡献。因此第 \(j\) 个节点的梯度：

$$
\nabla_{f_j}L=Ra_j^\top
$$

虽然是对 \(f_j\) 求导，但其中的 \(R\) 依赖所有 \(f_l\)。所以隐藏节点通过共同残差耦合，而不是各自独立优化。

### Q: 模型为什么会“意识到”两个节点相似？

模型没有意识。它只是沿着 loss 下降最快的方向更新。两个相似节点解释同一部分目标，重复方向对降低残差的边际贡献低；而未解释残差方向的边际贡献高。输出层 ridge 解中的 inverse Gram 会自动扣除共线特征的重叠解释量，形成负交叉项；该项通过反向传播推动隐藏特征分化。

### Q: Repulsion 应该如何直觉理解？

Repulsion 不是物理排斥，而是隐式去冗余。相似节点功能上只提供一个解释方向，不能扩大特征空间；如果其中一个节点偏向 residual 中尚未解释的方向，则模型能解释更多目标结构，loss 下降更多。因此梯度自然推动重复节点分工。

---

## 18. 机制解释与 mechanistic interpretability 的关系

这篇论文对 mechanistic interpretability 的意义可以概括为以下几点。

1. 它把“模型是否泛化”转化为“模型内部表征是否结构化”。泛化不只是外部行为，而是内部表征是否对齐任务生成机制。

2. 它区分了训练集行为相同但内部机制不同的两类模型：

   - 随机特征插值；
   - 结构表征泛化。

   两者都可能有高 train accuracy，但测试分布、OOD 分布和机制审计结果会不同。

3. 它说明仅看 train/test accuracy 不够，还要看隐藏表征是否对应可解释结构。一个模型可能在训练集上表现完美，但靠的是 pair indicator 或 shortcut；另一个模型可能同样拟合训练集，但内部已经形成 Fourier / irrep-like 特征。

4. 在 group arithmetic 中，结构表征可以用 Fourier / irrep 明确识别。这使 mechanistic interpretability 从“观察 neuron pattern”升级为“把 neuron pattern 映射到任务的数学结构”。

5. 在 LLM 中，类比对象可能是：

   - induction heads；
   - syntax circuits；
   - entity-relation features；
   - algorithmic reasoning circuits；
   - latent concepts；
   - world model states。

6. 它为“机制引导的可信 AI”提供一个理论入口：

   - 可信不是只问模型答对没有；
   - 而是问模型靠 sample-specific memory、shortcut、spurious correlation，还是靠 stable structural representation。

对 Mechanism-Guided Trustworthy AI Systems 来说，这个框架尤其有价值。它提供了一条可操作链条：

$$
\text{data distribution}
\rightarrow
\text{energy landscape}
\rightarrow
\text{representation type}
\rightarrow
\text{generalization behavior}
\rightarrow
\text{trustworthiness}.
$$

这条链条把数据工程、优化动力学、内部机制解释和可信泛化连接起来。

---

## 19. 核心贡献总结

本文的贡献不是简单报告 grokking 现象，而是把 delayed generalization 拆解为可分析的训练动力学机制。

1. 从经验 grokking 现象推进到反向传播动力学机制。论文试图解释为什么 train accuracy 已经很高后，隐藏层仍会继续发生结构性变化。

2. 用 \(G_F\) 的结构解释三阶段。Stage I 中 \(G_F\) 对隐藏层而言噪声较强；Stage II 中 \(G_F\) 开始携带标签结构；Stage III 中 \(G_F\) 受到 feature interaction 和 residual modulation 影响。

3. 解释为什么输出层先在随机特征上过拟合。大宽度随机特征矩阵使输出层可以快速插值训练集，而隐藏层早期收到的梯度信号方向性较弱。

4. 解释为什么隐藏层后续开始学习结构特征。输出层读出形成以后，反传到隐藏层的信号不再纯随机，而是携带标签结构，从而推动特征涌现。

5. 将 Stage II 的隐藏层学习等价为能量函数梯度上升。单个神经元的更新可近似写作：

   $$
   -\nabla_{w_j}L\approx c\nabla E(w_j).
   $$

6. 将能量函数的局部极值和 feature emergence 联系起来。隐藏神经元最终学到什么，取决于它落入哪个局部能量峰的吸引盆。

7. 在 group arithmetic 中用群表示理论证明泛化特征对应 irreps / Fourier bases。这为“结构表征”提供了数学定义，而不是依赖主观解释。

8. 分析样本量和数据分布如何改变经验能量景观。均匀且足够多的数据稳定结构峰；少量或偏分布数据可能产生记忆峰或 shortcut 峰。

9. 分析 Stage III 中隐藏节点交互、相似特征排斥、缺失特征补齐。inverse Gram 解释 repulsion，residual 解释 top-down modulation。

10. 给出可证伪诊断指标，而不仅仅是叙事解释。可测量对象包括 \(G_F\)-label alignment、Gram off-diagonal ratio、Fourier / irrep alignment、gradient-energy cosine similarity 等。

---

## 20. 局限性与批判性讨论

1. 严格证明主要限于两层非线性网络。两层网络足以揭示 feature emergence 的基本机制，但不能直接代表深层 Transformer 的全部训练动力学。

2. 任务主要是 group arithmetic，不等于真实 LLM 预训练。Group arithmetic 有清晰的代数结构，便于定义 Fourier / irrep 特征；自然语言任务的结构更复杂、更稀疏、更混合。

3. 三阶段 transition time 尚未完整理论化。论文提供了 regime-level 描述和诊断指标，但没有给出对所有设置通用的 epoch-level transition law。

4. 能量函数等价依赖 Stage II 条件。若 \(G_F\) 的结构不满足，或隐藏节点已经强耦合，则：

   $$
   -\nabla_{w_j}L\approx c\nabla E(w_j)
   $$

   这一解释不成立或需要修正。

5. 大宽度、随机初始化、相关矩阵近似对角等假设在真实网络中未必完全成立。真实模型中存在 layer norm、residual stream、attention routing、optimizer state 等因素，它们会改变梯度传播结构。

6. 真实 Transformer 有 attention、residual stream、layer norm、MLP、optimizer 等复杂因素。每一层都可能重写特征空间，使单神经元能量景观不再是充分描述。

7. LLM 的“结构峰”未必能像 modular addition 一样用 Fourier / irrep 明确标记。真实语言、知识和推理任务缺少单一、封闭、可枚举的群结构。

8. 真实模型存在 superposition，多个特征可能压缩在同一方向中。一个 neuron 或一个 direction 可能同时承载多个特征，导致高能量峰与可解释单特征之间的对应变得模糊。

9. 记忆与泛化在真实任务中可能不是二分，而是连续谱。例如 entity memory、factual recall、algorithmic abstraction 和 shortcut correlation 可能同时存在，并在不同上下文中发挥作用。

10. 论文提供的是机制框架，不是完整解释所有大模型泛化。它的最大价值是提出一组可检验的内部变量，而不是给出一个包打天下的 LLM generalization theorem。

还需要特别注意 v5 版本的一个变化：arXiv 备注显示，最新版本加入了初始 \(G_F\) 也携带有用信号的机制，并减少了理论对 weight decay 的依赖。因此如果阅读早期版本或早期笔记，需要重新核对“Stage I 是否完全依赖 lazy learning + weight decay”的表述，避免把旧版本结论误认为最新定理。

---

## 21. 未来研究与创新性思考

### 21.1 从 toy group task 到 Transformer circuit

核心研究问题是：类似

$$
-\nabla L \approx \nabla E
$$

的等价能否在 Transformer 的 attention heads、MLP neurons、residual stream subspaces 中被观测？

可以设计如下研究：

- 记录训练中不同层的 representation；
- 计算特征与标签结构 / task structure 的 alignment；
- 观察是否存在 Stage I \(\rightarrow\) Stage II \(\rightarrow\) Stage III 类似动力学；
- 对 attention head 的 query-key-value 子空间分别定义能量或 alignment；
- 对 MLP neuron、residual direction、SAE feature 分别计算 feature-label CCA score；
- 用 causal intervention 验证高能量结构特征是否真的支持泛化。

关键挑战是：Transformer 中的“特征”不一定是单个 neuron，而可能是 residual stream direction、attention pattern、SAE latent 或多层 circuit。因此需要把 \(w_j\) 从单神经元权重推广为可干预的 representation component。

### 21.2 能量函数诊断工具

可以设计一个 empirical toolkit，专门诊断模型是否从随机插值转向结构表征。候选指标包括：

- \(G_F\)-label alignment；
- feature-label CCA score；
- hidden-feature Gram off-diagonal ratio；
- gradient-energy cosine similarity；
- Fourier / irrep / task-basis alignment；
- residual missing-feature spectrum。

对 synthetic tasks，可以直接定义 task-basis；对真实任务，可以用程序生成标签结构、因果变量、语法树、关系图谱或外部 simulator 作为 weak structural target。这样的工具可以服务于 mechanistic interpretability，也可以服务于训练过程监控。

### 21.3 数据分布作为能量景观塑形工具

从数据工程角度，本文暗示：

- 数据量不是唯一关键；
- 关键是数据是否覆盖真实结构；
- 可以设计 curriculum / active sampling，让 \(E_S\) 更接近 \(E_{\mathcal D}\)；
- 可以用数据分布抑制记忆峰，增强结构峰。

具体做法包括：

- 对 algorithmic tasks，主动采样覆盖所有组合结构；
- 对动态系统预测，采样覆盖关键状态转移而非只覆盖高频状态；
- 对安全关键任务，增加反 shortcut 样本，降低 spurious peak 的能量；
- 对 OOD 泛化，设计能暴露不变机制的训练分布，而不是只扩大同分布样本数。

这将“数据选择”从经验工程提升为 energy landscape shaping。

### 21.4 可信 AI 中的应用

将此理论用于：

- 判断模型是 memorization-based 还是 mechanism-based；
- 检测 shortcut learning；
- 评估分布外泛化；
- 设计可干预表征；
- 对高风险系统进行内部机制审计。

一个可信 AI 系统不应只报告预测准确率，还应报告内部机制证据。例如，在医学、金融、工程控制、气候或复杂动态系统预测中，模型可能在历史数据上表现良好，但依赖 spurious correlation。能量景观和表征 alignment 诊断可以帮助区分“历史插值”与“机制泛化”。

### 21.5 与我的研究方向的连接

> 我的长期方向是 Mechanism-Guided Trustworthy AI Systems。这篇论文提供了一个非常重要的理论切入点：模型的可信泛化不是由外部 accuracy 单独决定，而是由内部表征是否收敛到稳定、可解释、可组合的结构特征决定。未来可以把这种“数据分布-优化景观-表征类型-泛化行为”的链条扩展到复杂动态系统中的可靠预测、分布漂移检测、不确定性校准和机制可解释性分析。

进一步说，这篇论文给我的启发是：可信泛化研究可以不只停留在 calibration、robustness benchmark 或 post-hoc explanation，而应深入训练动力学本身，追踪机制性特征何时、为何、在什么数据条件下涌现。

---

## 22. 建议的后续实验设计

### 实验 1：modular addition grokking 复现

任务：

$$
y=a+b \mod p.
$$

实验设置：

- 改变训练比例；
- 记录 train/test accuracy；
- 记录 \(\lVert \Delta A\rVert\)、\(\lVert \Delta W\rVert\)；
- 记录隐藏特征 Fourier alignment；
- 观察 grokking 相变。

预期现象：低训练比例或偏采样时，测试集泛化延迟更明显；均匀且足够多的数据应更稳定地诱导 Fourier features。

### 实验 2：Stage 诊断

记录以下指标：

$$
\mathrm{Align}(G_F, YY^\top F),
$$

$$
\frac{\left\lVert F^\top F-\mathrm{diag}(F^\top F)\right\rVert_F}{\left\lVert F^\top F\right\rVert_F},
$$

$$
\cos(-\nabla_w L,\nabla_w E).
$$

验证 Stage I / II / III 是否成立：

- Stage I：\(\lVert \Delta A\rVert\gg \lVert \Delta W\rVert\)，\(G_F\) 对隐藏层方向性弱；
- Stage II：gradient-energy cosine 上升，Gram off-diagonal ratio 较低；
- Stage III：Gram off-diagonal ratio 上升，repulsion 和 residual modulation 变明显。

### 实验 3：偏分布数据导致记忆峰

只采某个 target slice：

$$
a+b=c.
$$

观察隐藏特征是否变成 sample / pair indicator，而不是 Fourier structure。可用以下指标比较：

- pair indicator alignment；
- Fourier alignment；
- test generalization across unseen pairs；
- energy peak location visualization。

预期：偏分布会把经验能量景观塑造成更偏向记忆峰，使隐藏神经元学习 non-generalizable features。

### 实验 4：从 group task 到真实任务

在小型 Transformer 上设计 synthetic algorithmic task，观察是否存在类似：

- early memorization；
- later feature emergence；
- representation alignment；
- feature specialization。

可选任务包括：

- modular addition / composition；
- finite-state automata；
- Dyck language；
- key-value induction；
- relational composition；
- grid-world transition prediction。

关键是为任务定义可验证的结构基，例如 Fourier basis、automaton state、syntax stack depth、relation graph latent state，然后追踪模型内部表征是否从 sample-specific memory 转向这些结构基。

---

## 23. 最后一页：我的研究性总结

> 这篇论文最重要的地方，不是简单说“模型先记忆后泛化”，而是试图把这一现象拆解为可分析的训练动力学。Stage I 说明输出层可以先在随机高维特征空间里插值训练集；Stage II 说明当反传信号开始携带标签结构时，隐藏神经元的更新可以等价为对能量函数的梯度上升；Stage III 说明当隐藏特征不再独立时，节点之间通过 inverse Gram 和 residual 产生交互，从而推动特征分化和缺失结构补齐。数据量与数据分布则通过改变经验能量函数 \(E_S\) 的形状，决定结构峰还是记忆峰主导。这个框架使“记忆”和“泛化”不再只是行为层面的现象，而成为内部表征类型和优化景观局部极值之间的关系。

> 对 mechanistic interpretability 来说，这篇论文的启发是：真正值得追踪的不是模型什么时候 train accuracy 上升，而是隐藏表征什么时候从 sample-specific interpolation 转向 structure-specific representation。

这次讨论进一步澄清了 Li₂ 框架中“独立”与“交互”的精确含义。Stage II 的 independent feature learning 并不是说隐藏节点之间没有任何数学联系，而是说在 iid 随机初始化和大宽度集中下，随机特征核 \(K_m=\frac1mFF^\top\) 接近稳定平均场，单个节点对整体背景影响为 \(O(1/m)\)，因此每个节点的动力学可以近似解耦。Stage III 则是在节点学到结构后，\(F^\top F\) 的非对角项开始显著，隐藏节点之间的冗余无法再忽略。此时输出层的联合 ridge 解通过 \((F^\top F+\lambda I)^{-1}\) 产生负交叉项，相似节点的重叠贡献被扣除，残差中尚未解释的结构产生更强梯度，于是隐藏特征从重复走向分工。这个过程不是模型“意识到”冗余，而是 loss geometry、projection residual 与 inverse Gram 共同编码出的优化动力学。
这也改变了我对 trustworthy generalization 的理解。一个模型在训练集或普通测试集上表现好，只能说明它在某个分布切片上行为正确；真正可信的模型还需要内部机制证据，说明它的预测来自稳定结构，而不是来自偶然相关、数据泄漏、模板记忆或局部 shortcut。本文提供了一种可以继续扩展的理论路线：用训练动力学解释表征涌现，用能量景观区分记忆峰和结构峰，用机制诊断连接内部表示和外部泛化。

---

## 待进一步核对论文原文的细节

以下内容需要回到最新 arXiv v5 PDF/TeX 逐条确认，避免把早期版本、作者网页摘要或二级资料中的表述误写成最终定理：

1. **Theorem / Lemma 编号是否与最新 arXiv 版本一致**
   当前根据可访问 HTML 信息，论文包含与 backpropagated gradient、energy function、local maxima、target reconstruction、sample complexity、memorization solution、repulsion、top-down modulation、Muon optimizer 等相关的 Lemma / Theorem / Corollary。正式笔记库中应核对每个编号、命题名和假设条件。

2. **能量函数原文精确记号**
   本笔记使用
   $$
   E(w)=\frac12\lVert \tilde Y^\top \tilde h_w\rVert^2
   $$
   作为简化解释。原文可能包含更精确的中心化、归一化、输出维度、regularization、population / empirical 版本以及 nonlinear CCA 表述，需要逐式核对。

3. **group arithmetic 任务中 sample complexity 的精确 scaling law**
   本笔记只保留“样本量和分布改变经验能量景观，从而影响结构峰稳定性”的机制解释。原文中关于保持局部最优、结构峰出现、记忆峰主导或泛化特征可恢复的 scaling law，需要核对具体依赖项，例如群大小、表示维度、样本比例、失败概率、常数和 logarithmic factors。

4. **Muon optimizer 部分是否需要单独展开**
   arXiv v5 页面和论文目录显示包含 Muon optimizer 相关分析。当前笔记重点放在 feature emergence 和 grokking 机制，尚未展开 Muon 如何重平衡梯度更新、为何影响 delayed generalization，以及它与 \(A/W\) 相对更新速度的关系。

5. **multi-layer extension 的具体假设**
   若论文对多层网络或更复杂架构给出 extension，需要核对其假设是否只是 heuristic、实验观察，还是有形式化 theorem。尤其要确认是否要求层间近似独立、局部线性化、固定前层特征、特定初始化尺度或特定 optimizer。

6. **v5 中“初始 \(G_F\) 也携带有用信号”的机制细节**
   早期解释容易把 Stage I 写成纯 lazy learning + weight decay。v5 备注指出理论已加入初始 \(G_F\) 的有用信号并减少对 weight decay 的依赖，因此需要核对这部分如何修改 Stage I / Stage II 的边界。

7. **zero-initialized output layers 加速 grokking 的实验细节**
   需要确认该实验的任务、网络、优化器、学习率、初始化、是否只在 group tasks 中观察到，以及它如何改变 \(G_F\)、\(A/W\) 更新速度和 feature emergence 时间。

---

## 参考与版本来源

- arXiv: [2509.21519, Provable Scaling Laws of Feature Emergence from Learning Dynamics of Grokking](https://arxiv.org/abs/2509.21519)
- ar5iv HTML rendering: [ar5iv.labs.arxiv.org/html/2509.21519](https://ar5iv.labs.arxiv.org/html/2509.21519)
