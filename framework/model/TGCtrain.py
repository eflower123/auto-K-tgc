import math

import torch
from torch.autograd import Variable
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from sklearn.cluster import KMeans
import numpy as np
import sys
import os
from model.DataSet import TGCDataSet
from model.evaluation import eva
from torch.nn import Linear
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

FType = torch.FloatTensor
LType = torch.LongTensor

DID = 0

EXP_MAX = 20.0
EXP_MIN = -20.0

# GPU K-Means lazy import
_gpu_kmeans_fn = None


def _get_gpu_kmeans():
    global _gpu_kmeans_fn
    if _gpu_kmeans_fn is None and torch.cuda.is_available():
        _k_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'k_selection')
        if _k_path not in sys.path:
            sys.path.insert(0, _k_path)
        from kmeans_gpu import kmeans as gk_fn
        _gpu_kmeans_fn = gk_fn
    return _gpu_kmeans_fn


class TGC:
    def __init__(self, args, num_workers=4):
        self.args = args
        self.the_data = args.dataset
        self.file_path = '../data/%s/%s.txt' % (self.the_data, self.the_data)
        self.emb_path = '../emb/%s/%s_TGC_%d.emb'
        self.feature_path = './pretrain/%s_feature.emb' % self.the_data
        self.label_path = '../data/%s/node2label.txt' % self.the_data
        self.labels = self.read_label()
        self.emb_size = args.emb_size
        self.neg_size = args.neg_size
        self.hist_len = args.hist_len
        self.batch = args.batch_size
        self.clusters = args.clusters
        self.save_step = args.save_step
        self.epochs = args.epoch
        self.num_workers = num_workers
        self.best_acc = 0
        self.best_nmi = 0
        self.best_ari = 0
        self.best_f1 = 0

        # Dynamic K-selection config (Fix 2: warmup default 10)
        self.dynamic_k = getattr(args, 'dynamic_k', False)
        self.k_warmup = getattr(args, 'k_warmup_epochs', 10)
        self.k_interval = getattr(args, 'k_interval', 10)
        self.k_selector = None
        # Fix 5: K change cooldown — prevent oscillation
        self.k_cooldown = 0
        self.k_cooldown_period = 2

        print(f'[TGC] Initialized: dataset={self.the_data}, K={self.clusters}, '
              f'emb_size={self.emb_size}, epochs={self.epochs}, num_workers={self.num_workers}, '
              f'dynamic_k={self.dynamic_k}')

        self.data = TGCDataSet(self.file_path, self.neg_size, self.hist_len, self.feature_path, args.directed)
        self.node_dim = self.data.get_node_dim()
        self.edge_num = self.data.get_edge_num()
        self.feature = self.data.get_feature()

        self.node_emb = Variable(torch.from_numpy(self.feature).type(FType).cuda(), requires_grad=True)
        self.pre_emb = Variable(torch.from_numpy(self.feature).type(FType).cuda(), requires_grad=False)
        self.delta = Variable((torch.zeros(self.node_dim) + 1.).type(FType).cuda(), requires_grad=True)

        self._init_cluster_layer()
        self.v = 1.0
        self.batch_weight = math.ceil(self.batch / self.edge_num)

        self.opt = SGD(lr=args.learning_rate, params=[self.node_emb, self.delta, self.cluster_layer])
        self.loss = torch.FloatTensor()
        self.scaler = GradScaler()

    def _init_cluster_layer(self):
        """Initialize cluster centers. Uses GPU K-Means when available (Fix 3)."""
        self.cluster_layer = Variable(
            (torch.zeros(self.clusters, self.emb_size) + 1.).type(FType).cuda(),
            requires_grad=True
        )
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        gk = _get_gpu_kmeans()
        if gk is not None:
            X = torch.FloatTensor(self.feature)
            _, centers, _ = gk(
                X, num_clusters=self.clusters, distance='euclidean',
                init_mode='kmeans++', init_time=20,
                device=torch.device('cuda'),
            )
            self.cluster_layer.data = centers
        else:
            kmeans = KMeans(n_clusters=self.clusters, n_init=20)
            _ = kmeans.fit_predict(self.feature)
            self.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).cuda()

    def _update_clusters(self, new_k, embeddings_np=None):
        """
        Hot-resize cluster_layer when K changes.
        Fix 1: Preserves optimizer momentum for node_emb and delta.
        Fix 3: Uses GPU K-Means.
        """
        old_k = self.clusters

        # Save old optimizer state for node_emb and delta
        old_opt_state = self.opt.state_dict() if hasattr(self.opt, 'state_dict') else None

        self.clusters = new_k

        new_cl = Variable(
            torch.zeros(new_k, self.emb_size).type(FType).cuda(),
            requires_grad=True
        )
        torch.nn.init.xavier_normal_(new_cl.data)

        # Use GPU K-Means when available
        if embeddings_np is not None:
            gk = _get_gpu_kmeans()
            if gk is not None:
                X = torch.FloatTensor(embeddings_np)
                _, centers, _ = gk(
                    X, num_clusters=new_k, distance='euclidean',
                    init_mode='kmeans++', init_time=10,
                    device=torch.device('cuda'),
                )
                new_cl.data = centers
            else:
                kmeans = KMeans(n_clusters=new_k, n_init=10)
                _ = kmeans.fit_predict(embeddings_np)
                new_cl.data = torch.tensor(kmeans.cluster_centers_).cuda()
        else:
            gk = _get_gpu_kmeans()
            if gk is not None:
                X = torch.FloatTensor(self.feature)
                _, centers, _ = gk(
                    X, num_clusters=new_k, distance='euclidean',
                    init_mode='kmeans++', init_time=10,
                    device=torch.device('cuda'),
                )
                new_cl.data = centers
            else:
                kmeans = KMeans(n_clusters=new_k, n_init=10)
                _ = kmeans.fit_predict(self.feature)
                new_cl.data = torch.tensor(kmeans.cluster_centers_).cuda()

        self.cluster_layer = new_cl

        # Rebuild optimizer preserving momentum for node_emb and delta
        self.opt = SGD(lr=self.args.learning_rate,
                       params=[self.node_emb, self.delta, self.cluster_layer])

        if old_opt_state is not None:
            new_opt_state = self.opt.state_dict()
            old_ids = old_opt_state['param_groups'][0]['params']
            new_ids = new_opt_state['param_groups'][0]['params']
            # Restore state for node_emb (index 0) and delta (index 1)
            for i in range(2):
                if old_ids[i] in old_opt_state['state']:
                    new_opt_state['state'][new_ids[i]] = old_opt_state['state'][old_ids[i]]
            self.opt.load_state_dict(new_opt_state)

        print(f'[TGC] K updated: {old_k} → {new_k}, cluster_layer resized (optimizer preserved)')

    def read_label(self):
        n2l = dict()
        labels = []
        with open(self.label_path, 'r') as reader:
            for line in reader:
                parts = line.strip().split()
                n_id, l_id = int(parts[0]), int(parts[1])
                n2l[n_id] = l_id
        for i in range(len(n2l)):
            labels.append(int(n2l[i]))
        return labels

    def kl_loss(self, z, p):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        the_kl_loss = F.kl_div((q.log()), p, reduction='batchmean')
        return the_kl_loss

    def target_dis(self, emb):
        q = 1.0 / (1.0 + torch.sum(torch.pow(emb.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        tmp_q = q.detach()
        weight = tmp_q ** 2 / tmp_q.sum(0)
        p = (weight.t() / weight.sum(1)).t()
        return p

    def forward(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        batch = s_nodes.size()[0]
        s_node_emb = self.node_emb.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)
        t_node_emb = self.node_emb.index_select(0, Variable(t_nodes.view(-1))).view(batch, -1)
        h_node_emb = self.node_emb.index_select(0, Variable(h_nodes.view(-1))).view(batch, self.hist_len, -1)
        n_node_emb = self.node_emb.index_select(0, Variable(n_nodes.view(-1))).view(batch, self.neg_size, -1)
        s_pre_emb = self.pre_emb.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)

        s_p = self.target_dis(s_pre_emb)
        s_kl_loss = self.kl_loss(s_node_emb, s_p)
        l_node = s_kl_loss

        new_st_adj = torch.cosine_similarity(s_node_emb, t_node_emb)
        res_st_loss = torch.norm(1 - new_st_adj, p=2, dim=0)
        new_sh_adj = torch.cosine_similarity(s_node_emb.unsqueeze(1), h_node_emb, dim=2)
        new_sh_adj = new_sh_adj * h_time_mask
        new_sn_adj = torch.cosine_similarity(s_node_emb.unsqueeze(1), n_node_emb, dim=2)
        res_sh_loss = torch.norm(1 - new_sh_adj, p=2, dim=0).sum(dim=0, keepdims=False)
        res_sn_loss = torch.norm(0 - new_sn_adj, p=2, dim=0).sum(dim=0, keepdims=False)
        l_batch = res_st_loss + res_sh_loss + res_sn_loss

        l_framework = l_node + l_batch

        att = softmax(((s_node_emb.unsqueeze(1) - h_node_emb) ** 2).sum(dim=2).neg(), dim=1)

        p_mu = ((s_node_emb - t_node_emb) ** 2).sum(dim=1).neg()
        p_alpha = ((h_node_emb - t_node_emb.unsqueeze(1)) ** 2).sum(dim=2).neg()

        delta = self.delta.index_select(0, Variable(s_nodes.view(-1))).unsqueeze(1)
        d_time = torch.abs(t_times.unsqueeze(1) - h_times)
        exp_arg = torch.clamp(delta * Variable(d_time), min=EXP_MIN, max=EXP_MAX)
        p_lambda = p_mu + (att * p_alpha * torch.exp(exp_arg) * Variable(h_time_mask)).sum(dim=1)

        n_mu = ((s_node_emb.unsqueeze(1) - n_node_emb) ** 2).sum(dim=2).neg()
        n_alpha = ((h_node_emb.unsqueeze(2) - n_node_emb.unsqueeze(1)) ** 2).sum(dim=3).neg()

        n_lambda = n_mu + (att.unsqueeze(2) * n_alpha * (torch.exp(exp_arg.unsqueeze(2))) * (
            Variable(h_time_mask).unsqueeze(2))).sum(dim=1)

        loss = -torch.log(p_lambda.sigmoid() + 1e-6) - torch.log(n_lambda.neg().sigmoid() + 1e-6).sum(dim=1)

        total_loss = loss.sum() + l_framework

        return total_loss

    def update(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        if torch.cuda.is_available():
            with torch.cuda.device(DID):
                self.opt.zero_grad()
                with autocast():
                    loss = self.forward(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask)
                self.loss += loss.detach()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
        else:
            self.opt.zero_grad()
            loss = self.forward(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask)
            self.loss += loss.detach()
            loss.backward()
            self.opt.step()

    def _k_selection_step(self, epoch):
        """Run quick K-selection on current TGC embeddings (Fix 4: warm-start encoder)."""
        if self.k_selector is None:
            from k_selection.rgc_k_selector import RGCKSelector
            # Fix 6: K range comes from args only (deduplicated)
            k_min = getattr(self.args, 'k_min', 2)
            k_max = getattr(self.args, 'k_max', 20)
            k_step = getattr(self.args, 'k_step', 1)

            if k_max - k_min > 30:
                k_step = max(k_step, (k_max - k_min) // 20)
            candidate_ks = list(range(k_min, k_max + 1, k_step))

            self.k_selector = RGCKSelector(
                input_dim=self.emb_size,
                hidden_dim=500,
                candidate_ks=candidate_ks,
                E_epochs=getattr(self.args, 'k_selection_epochs', 200),
                epsilon_start=getattr(self.args, 'k_selection_epsilon', 0.5),
                seed=getattr(self.args, 'k_selection_seed', 42),
                pca_components=getattr(self.args, 'k_selection_pca', None),
                reward_k_penalty=getattr(self.args, 'reward_k_penalty', 0.3),
                q_discount=getattr(self.args, 'q_discount', 0.5),
                kmeans_chunk_size=getattr(self.args, 'kmeans_chunk_size', 2000),
            )

        embeddings = self.node_emb.detach().cpu().numpy()
        print(f'\n[TGC-K] Epoch {epoch}: Quick K-selection on TGC embeddings '
              f'(nodes={embeddings.shape[0]}, dim={embeddings.shape[1]})')

        # Fix 4: pass previous encoder for warm-start fine-tuning
        new_k, _ = self.k_selector.select_k_quick(
            embeddings, self.labels,
            pretrained_encoder=self.k_selector.encoder
        )
        return new_k

    def train(self):
        print(f'[TGC] Training: nodes={self.node_dim}, edges={self.edge_num}, '
              f'clusters={self.clusters}')
        if self.dynamic_k:
            print(f'[TGC] Dynamic K enabled: warmup={self.k_warmup}, interval={self.k_interval}, '
                  f'cooldown={self.k_cooldown_period}')

        k_history = [self.clusters]

        # Windows multiprocessing: spawn requires pickling, fails on large arrays
        num_workers = self.num_workers if os.name != 'nt' else 0

        for epoch in range(self.epochs):
            self.loss = 0.0
            loader = DataLoader(self.data, batch_size=self.batch, shuffle=True,
                               num_workers=num_workers)

            for i_batch, sample_batched in enumerate(loader):
                if i_batch != 0:
                    sys.stdout.write('\r' + str(i_batch * self.batch) + '\tloss: ' + str(
                        self.loss.cpu().numpy() / (self.batch * i_batch)))
                    sys.stdout.flush()

                self.update(sample_batched['source_node'].type(LType).cuda(),
                            sample_batched['target_node'].type(LType).cuda(),
                            sample_batched['target_time'].type(FType).cuda(),
                            sample_batched['neg_nodes'].type(LType).cuda(),
                            sample_batched['history_nodes'].type(LType).cuda(),
                            sample_batched['history_times'].type(FType).cuda(),
                            sample_batched['history_masks'].type(FType).cuda())

            # Dynamic K-selection with cooldown (Fix 5)
            if self.dynamic_k and epoch >= self.k_warmup:
                if self.k_cooldown > 0:
                    self.k_cooldown -= 1
                elif epoch == self.k_warmup or (epoch - self.k_warmup) % self.k_interval == 0:
                    new_k = self._k_selection_step(epoch)
                    if new_k != self.clusters:
                        embeddings = self.node_emb.detach().cpu().numpy()
                        self._update_clusters(new_k, embeddings)
                        self.k_cooldown = self.k_cooldown_period
                    k_history.append(self.clusters)

            # Evaluation
            if self.the_data in ('arxivLarge', 'arxivPhy', 'arxivMath'):
                acc, nmi, ari, f1 = 0, 0, 0, 0
            else:
                with torch.no_grad():
                    acc, nmi, ari, f1 = eva(self.clusters, self.labels, self.node_emb)

            if nmi > self.best_nmi and epoch > 10:
                self.best_acc = acc
                self.best_nmi = nmi
                self.best_ari = ari
                self.best_f1 = f1
                self.save_node_embeddings(self.emb_path % (self.the_data, self.the_data, self.epochs))

            sys.stdout.write('\repoch %d: loss=%.4f  ' % (epoch, (self.loss.cpu().numpy() / len(self.data))))
            sys.stdout.write('ACC(%.4f) NMI(%.4f) ARI(%.4f) F1(%.4f)\n' % (acc, nmi, ari, f1))
            sys.stdout.flush()

        print('Best performance: ACC(%.4f) NMI(%.4f) ARI(%.4f) F1(%.4f)' %
              (self.best_acc, self.best_nmi, self.best_ari, self.best_f1))
        if self.dynamic_k and len(k_history) > 1:
            print(f'K evolution: {" → ".join(str(k) for k in k_history)}')

    def save_node_embeddings(self, path):
        if torch.cuda.is_available():
            embeddings = self.node_emb.cpu().detach().numpy()
        else:
            embeddings = self.node_emb.detach().numpy()

        npy_path = path.replace('.emb', '.npy')
        npy_ids_path = path.replace('.emb', '_ids.npy')
        np.save(npy_path, embeddings)
        np.save(npy_ids_path, np.arange(self.node_dim, dtype=np.int32))

        # Backward-compatible .emb text format
        writer = open(path, 'w')
        writer.write('%d %d\n' % (self.node_dim, self.emb_size))
        for n_idx in range(self.node_dim):
            writer.write(str(n_idx) + ' ' + ' '.join(str(d) for d in embeddings[n_idx]) + '\n')
        writer.close()
