import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from .kmeans_gpu import kmeans, kmeans_predict, setup_seed


class DualHeadEncoder(nn.Module):
    """Two-headed encoder for contrastive learning. L2-normalized outputs with PReLU."""

    def __init__(self, input_dim, hidden_dim, act="ident"):
        super(DualHeadEncoder, self).__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        nn.init.constant_(self.act1.weight, 0.25)
        nn.init.constant_(self.act2.weight, 0.25)

    def forward(self, x):
        out1 = self.act1(self.lin1(x))
        out2 = self.act2(self.lin2(x))
        out1 = F.normalize(out1, dim=1, p=2)
        out2 = F.normalize(out2, dim=1, p=2)
        return out1, out2


class QNetwork(nn.Module):
    """Q-Network with graph-level pooling for fixed-dim state representation."""

    def __init__(self, state_dim, num_actions, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.lin1 = nn.Linear(state_dim * 2, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, num_actions)
        self.reset_parameters()
        self.act = nn.ReLU()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, state_pooled, cluster_pooled):
        combined = torch.cat([state_pooled, cluster_pooled], dim=-1)
        x = self.act(self.lin1(combined))
        q_values = self.lin2(x)
        return q_values


def scatter_mean(src, index, num_classes):
    """Compute per-class mean."""
    if index.numel() == 0:
        return torch.zeros(num_classes, src.size(-1), device=src.device)
    index_onehot = F.one_hot(index.long(), num_classes=num_classes).float()
    sum_src = index_onehot.T @ src
    counts = index_onehot.sum(dim=0).clamp(min=1)
    return sum_src / counts.unsqueeze(1)


class RGCKSelector:
    def __init__(
        self,
        input_dim=128,
        hidden_dim=500,
        candidate_ks=None,
        k_min=2,
        k_max=10,
        k_step=1,
        E_epochs=200,
        Q_epochs=30,
        epsilon_start=0.5,
        epsilon_end=0.1,
        replay_buffer_size=50,
        lr_E=1e-3,
        lr_Q=1e-3,
        device=None,
        seed=42,
        pca_components=None,
        act="ident",
        max_infonce_nodes=2500,
        kl_chunk_size=2000,
        reward_k_penalty=0.3,
        q_discount=0.5,
        kmeans_chunk_size=2000,
        kmeans_init_time=20,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.E_epochs = E_epochs
        self.Q_epochs = Q_epochs
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.replay_buffer_size = replay_buffer_size
        self.lr_E = lr_E
        self.lr_Q = lr_Q
        self.seed = seed
        self.pca_components = pca_components
        self.act = act
        self.max_infonce_nodes = max_infonce_nodes
        self.kl_chunk_size = kl_chunk_size
        self.reward_k_penalty = reward_k_penalty
        self.q_discount = q_discount
        self.kmeans_chunk_size = kmeans_chunk_size
        self.kmeans_init_time = kmeans_init_time

        if candidate_ks is not None:
            self.candidate_ks = list(candidate_ks)
        else:
            self.candidate_ks = list(range(k_min, k_max + 1, k_step))

        self.num_actions = len(self.candidate_ks)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.encoder = None
        self.q_net = None
        self.optimizer_Q = None
        self.current_k = None

    def _build_mask(self, n):
        """Build InfoNCE mask without creating a full [2M,2M] tensor."""
        M2 = n * 2
        mask = torch.ones([M2, M2], device=self.device)
        mask.diagonal().fill_(0)
        return mask

    def _compute_reward(self, centers, dis, K):
        # Stage 3: division-based reward (like silhouette), always positive
        center_dis = torch.cdist(centers, centers, p=2).mean()
        min_dis_mean = dis.min(dim=1).values.mean()
        k_penalty = math.sqrt(K)
        reward = (center_dis / (min_dis_mean + 1e-6)) / k_penalty
        return reward

    def _select_action(self, state, cluster_state, epsilon):
        if np.random.random() < epsilon:
            action = np.random.randint(0, self.num_actions)
            return action, True
        else:
            with torch.no_grad():
                s_pooled = state.mean(dim=0, keepdim=True)
                c_pooled = cluster_state.mean(dim=0, keepdim=True)
                q_values = self.q_net(s_pooled, c_pooled)
                action = int(q_values.argmax(dim=-1).item())
            return action, False

    def _clustering_step(self, emb, K):
        predict_labels, centers, dis = kmeans(
            X=emb, num_clusters=K, distance="euclidean",
            init_mode="kmeans++", init_time=self.kmeans_init_time, device=self.device,
            chunk_size=self.kmeans_chunk_size,
        )
        return predict_labels, centers, dis

    def _train_q_network(self, replay_buffer):
        self.encoder.eval()
        self.q_net.train()

        s_pooled = torch.stack([item[0] for item in replay_buffer]).to(self.device)
        c_pooled = torch.stack([item[1] for item in replay_buffer]).to(self.device)
        actions = torch.tensor([item[2] for item in replay_buffer], device=self.device)
        s_new_p = torch.stack([item[3] for item in replay_buffer]).to(self.device)
        c_new_p = torch.stack([item[4] for item in replay_buffer]).to(self.device)
        rewards = torch.tensor([item[5] for item in replay_buffer], device=self.device)

        for _ in range(self.Q_epochs):
            self.optimizer_Q.zero_grad()
            q_values = self.q_net(s_pooled, c_pooled)
            q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q = self.q_net(s_new_p, c_new_p)
                y = rewards + self.q_discount * next_q.max(dim=1).values

            loss_Q = F.mse_loss(q_selected, y)
            loss_Q.backward()
            self.optimizer_Q.step()

    def _evaluate(self, state, K, labels):
        if labels is None:
            return {}
        predict_labels, _, _ = self._clustering_step(state, K)
        pred_np = predict_labels.numpy() if hasattr(predict_labels, 'numpy') else np.array(predict_labels)
        nmi = nmi_score(labels, pred_np, average_method='arithmetic')
        ari = ari_score(labels, pred_np)
        return {'nmi': 100 * nmi, 'ari': 100 * ari}

    def select_k(self, features, labels=None, pretrained_encoder=None):
        setup_seed(self.seed)

        if self.pca_components is not None and features.shape[1] > self.pca_components:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.pca_components, random_state=self.seed)
            features = pca.fit_transform(features)
            actual_input_dim = self.pca_components
        else:
            actual_input_dim = features.shape[1]

        X = torch.FloatTensor(features).to(self.device)
        N = X.size(0)

        # Fix 4: reuse pretrained encoder for warm-start fine-tuning
        if pretrained_encoder is not None and pretrained_encoder.lin1.in_features == actual_input_dim:
            self.encoder = pretrained_encoder.to(self.device)
        else:
            self.encoder = DualHeadEncoder(actual_input_dim, self.hidden_dim, act=self.act).to(self.device)

        # Stage 1: Q-network persists across select_k calls (no more amnesia)
        if self.q_net is None or self.q_net.lin2.out_features != self.num_actions:
            self.q_net = QNetwork(self.hidden_dim, self.num_actions).to(self.device)
            self.optimizer_Q = torch.optim.Adam(self.q_net.parameters(), lr=self.lr_Q)

        opt_E = torch.optim.Adam(self.encoder.parameters(), lr=self.lr_E)

        replay_buffer = []
        best_reward_avg = -float('inf')
        best_k = self.current_k if self.current_k is not None else self.candidate_ks[len(self.candidate_ks) // 2]
        best_metrics = {}
        reward_history = {k: [] for k in self.candidate_ks}
        min_visits = 5  # Stage 4b: statistical confidence threshold

        # Initial cluster state
        self.encoder.eval()
        with torch.no_grad():
            z1_init, z2_init = self.encoder(X)
            state_init = (z1_init + z2_init) / 2
        init_k = self.candidate_ks[len(self.candidate_ks) // 2]
        init_labels, _, _ = self._clustering_step(state_init, init_k)
        cluster_state = scatter_mean(
            state_init,
            init_labels.clone().detach().to(self.device),
            init_k
        )

        print(f'[KSelector] RL search: {self.num_actions} K values '
              f'[{self.candidate_ks[0]}..{self.candidate_ks[-1]}], '
              f'E={self.E_epochs}, ε={self.epsilon_start}→{self.epsilon_end}')

        for epoch in range(self.E_epochs):
            epsilon = max(self.epsilon_start * (0.98 ** epoch), self.epsilon_end)

            self.encoder.train()
            self.q_net.eval()
            opt_E.zero_grad()

            z1, z2 = self.encoder(X)
            state = (z1 + z2) / 2

            action, is_random = self._select_action(state, cluster_state, epsilon)
            K = self.candidate_ks[action]

            predict_labels, centers, dis = self._clustering_step(state.detach(), K)
            reward = self._compute_reward(centers, dis, K)

            cluster_state = scatter_mean(
                state.detach(),
                predict_labels.clone().detach().to(self.device),
                K
            )

            # InfoNCE (subsampled)
            if N > self.max_infonce_nodes:
                infonce_idx = torch.randperm(N, device=self.device)[:self.max_infonce_nodes]
                z1_sub = z1[infonce_idx]
                z2_sub = z2[infonce_idx]
                M = self.max_infonce_nodes
            else:
                z1_sub = z1
                z2_sub = z2
                M = N

            z1_z2 = torch.cat([z1_sub, z2_sub], dim=0)
            S = z1_z2 @ z1_z2.T
            mask_sub = self._build_mask(M)
            pos_neg = mask_sub * torch.exp(S)
            pos = torch.cat([torch.diag(S, M), torch.diag(S, -M)], dim=0)
            pos = torch.exp(pos)
            neg = (torch.sum(pos_neg, dim=1) - pos)
            info_loss = (-torch.log(pos / (pos + neg + 1e-8) + 1e-8)).sum() / (2 * M)

            # KL clustering loss (chunked)
            q_list = []
            for chunk_start in range(0, N, self.kl_chunk_size):
                chunk_end = min(chunk_start + self.kl_chunk_size, N)
                state_chunk = state[chunk_start:chunk_end]
                dist_chunk = torch.sum(
                    (state_chunk.unsqueeze(1) - centers.unsqueeze(0)).pow(2), 2
                )
                q_chunk = 1.0 / (1.0 + dist_chunk)
                q_list.append(q_chunk)
            q = torch.cat(q_list, dim=0)
            q = (q.t() / q.sum(1)).t()
            p = q.pow(2) / q.sum(0).unsqueeze(0)
            p = (p.t() / p.sum(1)).t()
            kl_loss = F.kl_div((q + 1e-8).log(), p, reduction='batchmean')

            # Dynamic KL annealing (Stage 2b): 0 for first 20% epochs, ramp to 5.0
            warmup_frac = 0.2
            kl_max = 5.0
            if epoch < self.E_epochs * warmup_frac:
                kl_weight = 0.0
            else:
                progress = (epoch - self.E_epochs * warmup_frac) / (self.E_epochs * (1.0 - warmup_frac))
                kl_weight = kl_max * progress
            loss = info_loss + kl_weight * kl_loss
            loss.backward()
            opt_E.step()

            # Next state
            self.encoder.eval()
            with torch.no_grad():
                z1_next, z2_next = self.encoder(X)
                next_state = (z1_next + z2_next) / 2
            next_cluster_state = scatter_mean(
                next_state,
                predict_labels.clone().detach().to(self.device),
                K
            )

            # Graph-level pooled states (fixed-size, not N-dependent)
            s_pooled = state.detach().mean(dim=0).cpu()
            c_pooled = cluster_state.detach().mean(dim=0).cpu()
            s_new_p = next_state.detach().mean(dim=0).cpu()
            c_new_p = next_cluster_state.detach().mean(dim=0).cpu()

            replay_buffer.append([s_pooled, c_pooled, action, s_new_p, c_new_p, reward.item()])
            reward_history[K].append(reward.item())

            if len(replay_buffer) >= self.replay_buffer_size:
                self._train_q_network(replay_buffer)
                replay_buffer = []

            if (epoch + 1) % max(1, self.E_epochs // 10) == 0:
                for k in self.candidate_ks:
                    if len(reward_history[k]) >= min_visits:
                        avg_r = np.mean(reward_history[k][-10:])
                        if avg_r > best_reward_avg:
                            best_reward_avg = avg_r
                            best_k = k
                mode_str = "R" if is_random else "Q"
                sys.stdout.write(
                    f'\r[KSelector] {epoch+1}/{self.E_epochs} | K={K}({mode_str}) '
                    f'| best_K={best_k} | loss={loss.item():.4f} | r={reward.item():.3f} | ε={epsilon:.3f}'
                )
                sys.stdout.flush()

        # Final sweep
        print(f'\n[KSelector] RL done. Evaluating candidates...')
        self.encoder.eval()
        with torch.no_grad():
            z1, z2 = self.encoder(X)
            final_state = (z1 + z2) / 2

        for k in self.candidate_ks:
            if len(reward_history[k]) >= min_visits:
                avg_r = np.mean(reward_history[k][-10:])
                if avg_r > best_reward_avg:
                    best_reward_avg = avg_r
                    best_k = k

        if labels is not None:
            best_metrics = self._evaluate(final_state, best_k, labels)

        self.current_k = best_k
        return best_k, best_metrics

    def select_k_quick(self, features, labels=None, pretrained_encoder=None):
        """
        Quick K-selection on in-training embeddings (fewer RL epochs, fewer inits).
        Used inside TGC training loop for periodic K re-evaluation.

        Runs with reduced E_epochs (1/2 of full mode) and kmeans_init_time (1/3).
        """
        saved_E = self.E_epochs
        saved_init = self.kmeans_init_time
        self.E_epochs = max(60, saved_E // 2)
        self.kmeans_init_time = max(5, saved_init // 3)

        best_k, metrics = self.select_k(features, labels, pretrained_encoder=pretrained_encoder)

        self.E_epochs = saved_E
        self.kmeans_init_time = saved_init
        return best_k, metrics


def select_k_for_dataset(dataset_name, feature_path, label_path, selector=None, **kwargs):
    """Convenience function to select K for a dataset. Supports .emb and .npy."""
    # Try .npy first
    npy_path = feature_path.replace('.emb', '.npy')
    ids_path = feature_path.replace('.emb', '_ids.npy')
    if os.path.exists(npy_path) and os.path.exists(ids_path):
        features_all = np.load(npy_path)
        ids_all = np.load(ids_path)
        node_emb = dict(zip(ids_all.astype(int), features_all))
    elif os.path.exists(feature_path):
        node_emb = dict()
        with open(feature_path, 'r') as reader:
            header = reader.readline()
            for line in reader:
                embeds = np.fromstring(line.strip(), dtype=float, sep=' ')
                node_id = int(embeds[0])
                node_emb[node_id] = embeds[1:]
    else:
        raise FileNotFoundError(f'Feature file not found: {feature_path} or {npy_path}')

    n2l = dict()
    with open(label_path, 'r') as reader:
        for line in reader:
            parts = line.strip().split()
            n_id, l_id = int(parts[0]), int(parts[1])
            n2l[n_id] = l_id

    features_list = []
    labels_list = []
    for n_id in sorted(node_emb.keys()):
        if n_id in n2l:
            features_list.append(node_emb[n_id])
            labels_list.append(n2l[n_id])

    features = np.array(features_list)
    labels = np.array(labels_list)

    print(f'[KSelector] Dataset: {dataset_name}, nodes={features.shape[0]}, dim={features.shape[1]}')

    if selector is None:
        selector = RGCKSelector(input_dim=features.shape[1], **kwargs)

    best_k, metrics = selector.select_k(features, labels)

    print(f'[KSelector] Selected K = {best_k}')
    if metrics:
        print(f'[KSelector] Metrics at K={best_k}: '
              f'NMI={metrics["nmi"]:.4f}, ARI={metrics["ari"]:.4f}')

    return best_k, metrics
