# KSelector 模块分析与 K 值变动算法逻辑

## 一、KSelector 模块架构

### 1.1 核心组件

```
RGCKSelector
├── DualHeadEncoder   # 双头编码器，用于对比学习
│   ├── lin1: Linear(input_dim → 500)
│   ├── lin2: Linear(input_dim → 500)  
│   ├── activate: identity (无激活函数)
│   └── 输出: L2归一化的 (out1, out2)
│
├── QNetwork          # Q值网络，用于epsilon-greedy决策
│   ├── lin1: Linear(1000 → 256)  # state_pooled + cluster_pooled 拼接
│   ├── lin2: Linear(256 → num_actions)
│   └── 输出: 每个候选K的Q值
│
├── scatter_mean()    # 按聚类标签求均值，构建cluster_state
├── _compute_reward() # 无监督奖励函数
├── _select_action()  # epsilon-greedy 动作选择
├── _clustering_step()# GPU K-Means 聚类
└── _train_q_network()# Q网络批训练
```

### 1.2 数据流（每个RL epoch）

```
 features [N, D]
     │
     ▼
 DualHeadEncoder ──► z1 [N,500], z2 [N,500]
     │
     ▼
 state = (z1+z2)/2  [N, 500]
     │
     ├──► _select_action(state, cluster_state, ε)
     │       │
     │       ├── ε概率: 随机选K
     │       └── 1-ε概率: QNetwork(argmax Q)
     │       ▼
     │    action → K = candidate_ks[action]
     │
     ├──► _clustering_step(state.detach(), K)
     │       │
     │       └── GPU K-Means (kmeans++, init_time次重启)
     │           ▼
     │       predict_labels [N], centers [K,500], dis [N,K]
     │
     ├──► _compute_reward(centers, dis, K)
     │       │
     │       center_dis = pairwise(centers, centers).mean()  # 中心间距离
     │       min_dis_mean = dis.min(dim=1).mean()            # 簇内紧凑度
     │       reward = (center_dis - min_dis_mean) / K^0.3
     │       ▼
     │    reward (标量)
     │
     ├──► scatter_mean(state, labels, K) → cluster_state [K,500]
     │
     ├──► InfoNCE loss (子采样到 max_infonce_nodes=2500)
     │    S = z1_z2 @ z1_z2.T  [2M, 2M] 相似度矩阵
     │    info_loss = -log(pos / (pos+neg))
     │
     ├──► KL clustering loss (分块计算，chunk=2000)
     │    q = soft_cluster_assignment(state, centers)  # Student's t
     │    p = q^2 / sum(q)  # 锐化目标分布
     │    kl_loss = KL(q || p)
     │
     ├──► loss = info_loss + 10 * kl_loss
     │    opt_E.step()  # 只更新编码器
     │
     ├──► 存储到replay_buffer:
     │    [s_pooled, c_pooled, action, s_next_pooled, c_next_pooled, reward]
     │    每个是 graph-level 均值池化向量 [500]
     │
     └──► buffer满(50) → _train_q_network()
          Q(s,c) ← reward + γ * max_a' Q(s', c')
          opt_Q.step()  # Q_epochs=30 轮
```

### 1.3 奖励函数详细分析

```python
center_dis = (centers[K,D] 两两欧氏距离平方).mean()   # 中心分离度，越大越好
min_dis_mean = dis[N,K].min(dim=1).values.mean()       # 点到最近中心的距离，越小越好
reward = (center_dis - min_dis_mean) / K^0.3           # K 惩罚项
```

**问题1：无监督信号与监督目标的错位**

奖励函数只评估"聚类几何质量"，不考虑标签。当数据分布与标签不一致时（例如ground truth是6类但分布呈现12个自然簇），高reward的K ≠ ground truth K。

**问题2：K惩罚力度不足**

指数 0.3 意味着：K=12 时惩罚因子 = 12^0.3 ≈ 2.11，K=6 时 = 6^0.3 ≈ 1.71。两者相差仅 1.23 倍，不足以显著偏好较小的K。

**问题3：负奖励的恶性循环**

当 `center_dis < min_dis_mean` 时，reward 为负。这发生在聚类结构差（中心近、簇松散）时。负奖励导致 Q-network 学到"所有K都差"，TD error 虽小但无辨别力，best_K 陷在初始中位值。

---

## 二、K 值变动算法逻辑（完整时序）

### 2.1 全局流程

```
┌─────────────────────────────────────────────────────────────────┐
│ main.py                                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [Phase 0] 参数验证                                              │
│    ├── --clusters=None & --auto_k=False → 报错退出              │
│    ├── k_warmup_epochs = epoch//5  (如 epoch=50 → 10)          │
│    └── k_interval = epoch//10     (如 epoch=50 → 5)            │
│                                                                  │
│  [Phase 1] --auto_k True?                                       │
│    └── run_standalone_k_selection()                             │
│        ├── 输入: pretrain/{dataset}_feature.emb (Node2Vec静态特征)│
│        ├── RGCKSelector.select_k() 完整模式 E=200               │
│        ├── 创建新编码器 (pretrained_encoder=None)               │
│        └── 输出: initial_K = RL搜索结果的best_k                 │
│                                                                  │
│  [Phase 2] TGC训练 + 动态K选择                                   │
│    ├── TGC.__init__(args)                                       │
│    │   ├── self.clusters = initial_K (从Phase 1或用户指定)      │
│    │   ├── _init_cluster_layer()  # GPU K-Means在Node2Vec特征上 │
│    │   ├── self.k_selector = None  # 首次使用时延迟创建         │
│    │   └── self.k_cooldown = 0                                  │
│    │                                                            │
│    └── TGC.train()                                              │
│        for epoch in range(epochs):                              │
│          │                                                       │
│          ├─ TGC forward/backward (所有batch)                    │
│          │                                                       │
│          ├─ 动态K选择判断 (epoch >= k_warmup)                    │
│          │   │                                                   │
│          │   ├─ cooldown > 0? → 跳过, cooldown--                │
│          │   │                                                   │
│          │   └─ 触发条件:                                       │
│          │       epoch == k_warmup  (第一次)                    │
│          │       或 (epoch - k_warmup) % k_interval == 0        │
│          │       │                                               │
│          │       └─ _k_selection_step(epoch)                    │
│          │           │                                           │
│          │           ├─ 首次调用: 创建 RGCKSelector             │
│          │           │   candidate_ks = range(k_min, k_max+1)   │
│          │           │                                           │
│          │           ├─ embeddings = node_emb.detach().numpy()  │
│          │           │   (当前TGC嵌入, [N,128])                 │
│          │           │                                           │
│          │           └─ select_k_quick(embeddings, labels,      │
│          │                          pretrained_encoder)         │
│          │               │                                       │
│          │               ├─ E_epochs = max(60, 200//2) = 100   │
│          │               ├─ kmeans_init_time = max(5, 20//3)=7 │
│          │               │                                       │
│          │               └─ select_k(labels, pretrained_encoder)│
│          │                   │                                   │
│          │                   ├─ 首次: encoder=None → 新编码器   │
│          │                   ├─ 后续: encoder=上次训练的 → 热启动│
│          │                   ├─ QNetwork: 每次重新创建! (Bug)   │
│          │                   │                                   │
│          │                   └─ 返回 best_k                     │
│          │                                                       │
│          ├─ new_k != self.clusters?                             │
│          │   ├─ _update_clusters(new_k, embeddings)             │
│          │   │   ├─ 用GPU K-Means在新嵌入上初始化cluster_layer  │
│          │   │   ├─ 重建SGD优化器，保留node_emb/delta的momentum │
│          │   │   └─ cluster_layer的momentum: 从0开始 (冷启动)   │
│          │   └─ k_cooldown = 2                                  │
│          │                                                       │
│          └─ 评估: eva(self.clusters, labels, node_emb)          │
│              ├─ GPU K-Means 在最终嵌入上聚类                    │
│              ├─ NMI, ARI: 与标签对比 (K≠#标签也能计算)          │
│              └─ ACC, F1: 需Munkres对齐, K≠#标签时返回0          │
│                                                                  │
│  [Phase 3] --tsne_viz True?                                     │
│    └── T-SNE可视化, 保存到 viz/{dataset}_TGC_K{K}.png           │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 K值变化的具体时间线（epoch=50为例）

```
epoch 0:  TGC训练开始, K=12 (Phase 1的结果)
...
epoch 10: [第一次K选择] warmup结束
          └─ select_k_quick (100 RL epochs, 新编码器)
          └─ best_K=10, K变为10
          └─ cooldown=2 (epoch 11,12 不检查)

epoch 15: [第二次K选择] (10-5=5, 5%5=0 → 触发)
          └─ select_k_quick (100 RL epochs, 热启动编码器)
          └─ best_K=10, K不变 (best_K stuck!)

epoch 20: [第三次K选择]
          └─ select_k_quick (热启动)
          └─ best_K=10, K不变

epoch 25, 30, 35, 40, 45: 同样卡在K=10
```

---

## 三、Loss 降不下去的根本原因

### 3.1 KL Loss 权重过大

```python
loss = info_loss + 10 * kl_loss
```

InfoNCE 对比损失负责学习好的表示，KL 损失负责聚类。权重 10 意味着编码器在训练的早期就过度关注拟合聚类结构，而表示还不够好。这是鸡生蛋问题：**没有好表示就看不清簇结构，看不清簇结构就被KL拉着乱跑**。

### 3.2 Activation 为 Identity

```python
if act == "ident":
    self.activate = lambda x: x
```

DualHeadEncoder 没有非线性激活函数。它本质上是两个线性层 + L2 归一化。无论多少层，线性变换的复合仍是线性的。编码器能力严重受限。

### 3.3 InfoNCE 子采样丢失信息

```python
max_infonce_nodes = 2500  # 默认
```

patent 有 12214 个节点，子采样到 2500 意味着 80% 的节点不参与对比学习。对于 100 个 RL epoch，每个 epoch 随机采样不同节点，导致 InfoNCE 的负样本不稳定。

### 3.4 Q-Network 每次重新初始化

```python
# select_k() 第220行
self.q_net = QNetwork(self.hidden_dim, self.num_actions).to(self.device)
```

每次调用 `select_k()` 都创建**全新的** Q-network。Fix 4 仅保留了编码器，Q-network 的 RL 知识全部丢弃。即使 Q-network 在某次K选择中学到了"某些K更好"，下次调用时这个知识就消失了。

### 3.5 Epsilon 线性衰减太快

```python
epsilon_decay = (0.5 - 0.1) / E_epochs  # 100 epochs → 0.004/epoch
```

前 25 个 epoch 内 ε 就从 0.5 降到 0.4，探索期过短。编码器还在学习表示，Q-network 却在基于噪声做决策。

### 3.6 Encoder 热启动的输入维度陷阱

```python
if pretrained_encoder is not None and pretrained_encoder.lin1.in_features == actual_input_dim:
    self.encoder = pretrained_encoder.to(self.device)
```

Phase 1 的编码器输入维度 = 128（Node2Vec 特征）。Phase 2 第一次 K 选择的输入维度 = 128（TGC 嵌入）。维度匹配 → 热启动成功。但 Phase 1 编码器是在**静态** Node2Vec 特征上训练的，直接迁移到**动态** TGC 嵌入上，表示的语义完全不同，热启动可能比随机初始化更差（局部最优陷阱）。

---

## 四、Best_K 锁死的根本原因

### 4.1 Reward 信号退化

训练日志显示：

| K选择时机 | best_K | reward | loss |
|-----------|--------|--------|------|
| Phase 1 (Node2Vec) | 12 | 0.219 | 7.84 |
| Epoch 10 (TGC) | 10 | 0.144 | 7.87 |
| Epoch 20 (TGC) | 10 | -0.118 | 7.71 |
| Epoch 30 (TGC) | 10 | -0.275 | 7.64 |

趋势：loss 缓慢下降（编码器在改善），但 reward 从正变负并持续恶化。

**原因**：TGC 嵌入仍在训练初期（epoch 10-30），嵌入空间尚未形成清晰的聚类结构。`center_dis < min_dis_mean` 导致 reward 为负。在这个信号下，所有候选 K 的 reward 区别很小，best_K 的自然选择是最多被选到的K（通常是中位值附近的）。

### 4.2 Best_K 更新机制缺陷

```python
best_k = self.candidate_ks[len(self.candidate_ks) // 2]  # 初始值 = 中位K
# ...
for k in self.candidate_ks:
    if len(reward_history[k]) >= 3:        # 只考虑被选≥3次的K
        avg_r = np.mean(reward_history[k][-10:])  # 最近10次平均
        if avg_r > best_reward_avg:
            best_k = k
```

**问题**：
- 初始 best_k = 中位K（candidate_ks=[2..12] → K=7），这意味着即使 K=6 的 reward 最高，只要 K=7 的 reward 曾经高过，best_k 就偏向中位值
- 只计算最近 10 次平均：如果前期探索阶段某个 K 恰好有高奖励（噪声），会长期占据 best_K
- 最终扫描只用 `reward_history[k][-10:]` 再次确认了同一个有噪声的候选

### 4.3 候选 K 访问不均匀

Epsilon-greedy 保证了探索，但：
- 前期（ε高）：各K访问概率均匀 → 好
- 后期（ε低）：Q-network 锁定某个K → 该K的reward_history持续增长，其他K不再被访问
- 最终扫描：只看已被访问过的K，未访问的K reward_history为空，自动排除

---

## 五、三层聚类的对比

| | TGC cluster_layer | KSelector GPU K-Means | Evaluation K-Means |
|---|---|---|---|
| **文件** | TGCtrain.py | rgc_k_selector.py | evaluation.py |
| **类型** | 可学习参数 (SGD) | 传统迭代算法 | 传统迭代算法 |
| **目标函数** | KL(q∥p) 软分配 | 欧氏距离最小化 | 欧氏距离最小化 |
| **初始化** | K-Means++ (GPU) | K-Means++ (GPU), init_time次重启 | K-Means++ (GPU), n_init=20 |
| **输入** | node_emb [N,128] | encoder(state) [N,500] | node_emb [N,128] |
| **K来源** | 当前 self.clusters | 每次RL选择的K | self.clusters (用于评估) |
| **输出** | 软聚类概率 q | 硬分配 labels | 硬分配 labels |

**关键矛盾**：TGC 的 cluster_layer 是在 128 维原始嵌入空间上聚类的，而 KSelector 是在 500 维编码器空间中聚类的。两者看到的"聚类结构"可能完全不同。KSelector 用 500 维编码器空间选出 K，但 TGC 实际在 128 维空间用这个 K 来训练。

---

## 六、总结：需要修改的点

| 序号 | 问题 | 严重程度 | 建议修改方向 |
|------|------|---------|-------------|
| 1 | Q-Network每次重新初始化 | **高** | 热启动Q-network或持久化到self.k_selector |
| 2 | DualHeadEncoder无非线性激活 | **高** | 添加ReLU或PReLU激活 |
| 3 | KL loss权重10过大 | **中** | 改为渐进权重(epoch前小后大)或降低到1-5 |
| 4 | Reward在欠训练嵌入上为负 | **中** | 添加baseline(如silhouette score)或延迟K选择时机 |
| 5 | Epsilon衰减过快 | **中** | 余弦衰减或阶梯衰减，保证前50% epoch ε>0.3 |
| 6 | InfoNCE子采样2500可能导致不稳定 | **中** | 自适应: min(N, max(2500, N//2)) |
| 7 | Best_K初始值偏向中位 | **低** | 改为最接近K_DICT值的候选K |
| 8 | Phase 1编码器热启动可能有害 | **低** | 添加reset选项，或强制Phase 2首次创建新编码器 |
| 9 | 128维与500维空间聚类不一致 | **低** | 考虑让TGC的cluster_layer也在编码器空间中优化 |
