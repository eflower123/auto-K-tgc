import sys
import math
import torch
import ctypes
import datetime
import numpy as np
import argparse
import time
import random
import os
from model import TGCtrain

sys.path.insert(0, os.path.dirname(__file__))
from k_selection import RGCKSelector, select_k_for_dataset, tsne_visualize

FType = torch.FloatTensor
LType = torch.LongTensor

K_RANGE_DICT = {
    'school':     (2, 16),
    'dblp':       (2, 20),
    'brain':      (2, 20),
    'patent':     (2, 12),
    'arxivAI':    (2, 12),
    'arxivCS':    (10, 60),
    'arxivPhy':   (20, 80),
    'arxivMath':  (10, 50),
    'arxivLarge': (100, 200),
}

K_DICT = {
    'arxivAI': 5, 'arxivCS': 40, 'arxivPhy': 53, 'arxivMath': 31,
    'arxivLarge': 172, 'school': 9, 'dblp': 10, 'brain': 10, 'patent': 6,
}


def run_standalone_k_selection(args):
    """Phase 1 (optional): Run RGC K-selection on Node2Vec embeddings as initial guess."""
    print(f'\n{"=" * 60}')
    print(f'Phase 1: Standalone RGC K-Selection on Node2Vec features [{args.dataset}]')
    print(f'{"=" * 60}')

    k_min = args.k_min
    k_max = args.k_max
    if k_min is None or k_max is None:
        default_range = K_RANGE_DICT.get(args.dataset, (2, 20))
        if k_min is None:
            k_min = default_range[0]
        if k_max is None:
            k_max = default_range[1]

    if k_max - k_min > 30:
        k_step = max(args.k_step, (k_max - k_min) // 20)
    else:
        k_step = args.k_step

    candidate_ks = list(range(k_min, k_max + 1, k_step))
    print(f'Searching K in [{k_min}, {k_max}] step={k_step} '
          f'({len(candidate_ks)} candidates)')

    feature_path = './pretrain/%s_feature.emb' % args.dataset
    label_path = '../data/%s/node2label.txt' % args.dataset

    selector = RGCKSelector(
        input_dim=args.emb_size,
        hidden_dim=500,
        candidate_ks=candidate_ks,
        E_epochs=args.k_selection_epochs,
        epsilon_start=args.k_selection_epsilon,
        seed=args.k_selection_seed,
        pca_components=args.k_selection_pca,
        reward_k_penalty=args.reward_k_penalty,
        q_discount=args.q_discount,
        kmeans_chunk_size=args.kmeans_chunk_size,
    )

    selected_k, k_metrics = select_k_for_dataset(
        args.dataset,
        feature_path=feature_path,
        label_path=label_path,
        selector=selector,
    )

    print(f'\n>>> Node2Vec-based initial K = {selected_k}')
    if k_metrics:
        print(f'>>> Metrics at K={selected_k}: '
              f'NMI={k_metrics["nmi"]:.4f}, ARI={k_metrics["ari"]:.4f}')
    print(f'{"=" * 60}\n')
    return selected_k


def run_visualization(args, the_train):
    """Phase 3: T-SNE Visualization."""
    print(f'\n{"=" * 60}')
    print(f'Phase 3: T-SNE Visualization')
    print(f'{"=" * 60}')

    try:
        emb_npy_path = '../emb/%s/%s_TGC_%d.npy' % (
            args.dataset, args.dataset, args.epoch
        )
        emb_path = '../emb/%s/%s_TGC_%d.emb' % (
            args.dataset, args.dataset, args.epoch
        )
        label_path = '../data/%s/node2label.txt' % args.dataset

        n2l = dict()
        with open(label_path, 'r') as reader:
            for line in reader:
                parts = line.strip().split()
                n_id, l_id = int(parts[0]), int(parts[1])
                n2l[n_id] = l_id

        if os.path.exists(emb_npy_path):
            embeddings_all = np.load(emb_npy_path)
            ids_all = np.load(emb_npy_path.replace('.npy', '_ids.npy'))
            node_emb = dict(zip(ids_all.astype(int), embeddings_all))
        elif os.path.exists(emb_path):
            node_emb = dict()
            with open(emb_path, 'r') as reader:
                reader.readline()
                for line in reader:
                    embeds = np.fromstring(line.strip(), dtype=float, sep=' ')
                    node_id = int(embeds[0])
                    if node_id in n2l:
                        node_emb[node_id] = embeds[1:]
        else:
            print(f'[Visualization] Embedding file not found, using in-memory embeddings...')
            embeddings = the_train.node_emb.cpu().data.numpy()
            labels = np.array(the_train.labels)
            _save_viz(args, embeddings, labels)
            return

        emb_list = []
        label_list = []
        for n_id in sorted(node_emb.keys()):
            if n_id in n2l:
                emb_list.append(node_emb[n_id])
                label_list.append(n2l[n_id])

        if len(emb_list) == 0:
            raise RuntimeError('No aligned embeddings found')

        embeddings = np.array(emb_list)
        labels = np.array(label_list)
        _save_viz(args, embeddings, labels)

    except Exception as e:
        print(f'[Visualization] Error: {e}')


def _save_viz(args, embeddings, labels):
    viz_dir = '../viz'
    os.makedirs(viz_dir, exist_ok=True)
    save_path = os.path.join(viz_dir, '%s_TGC_K%d' % (args.dataset, args.clusters))
    tsne_visualize(
        embeddings, labels, save_path,
        k=args.clusters,
        title='T-SNE: %s (K=%d, dynamic=%s)' % (
            args.dataset, args.clusters, str(args.dynamic_k)
        ),
        max_samples=5000,
    )


def main_train(args):
    start = datetime.datetime.now()

    # ============================================================
    # Validate: user must provide --clusters or enable --auto_k
    # ============================================================
    if args.clusters is None and not args.auto_k:
        print('错误：请指定初始K值（--clusters N）或启用自动K选择（--auto_k True）')
        sys.exit(1)

    # Auto-compute warmup and interval relative to total epochs
    if args.k_warmup_epochs is None:
        args.k_warmup_epochs = max(10, args.epoch // 5)
    if args.k_interval is None:
        args.k_interval = max(3, args.epoch // 10)

    # ============================================================
    # Phase 1 (Optional): Standalone K-guess on Node2Vec
    # ============================================================
    if args.auto_k:
        initial_k = run_standalone_k_selection(args)
        args.clusters = initial_k
    else:
        print(f'\nUsing initial K={args.clusters} from user config\n')

    # ============================================================
    # Phase 2: TGC Training (with optional dynamic K-selection)
    # ============================================================

    # Fix 6: Always populate k_min/k_max from shared K_RANGE_DICT before passing to TGC
    default_range = K_RANGE_DICT.get(args.dataset, (2, 20))
    if args.k_min is None:
        args.k_min = default_range[0]
    if args.k_max is None:
        args.k_max = default_range[1]

    print(f'{"=" * 60}')
    print(f'Phase 2: TGC Training')
    print(f'  Initial K = {args.clusters}')
    print(f'  Dynamic K = {args.dynamic_k}')
    if args.dynamic_k:
        print(f'  Warmup = {args.k_warmup_epochs}, Interval = {args.k_interval}')
        print(f'  K range = [{args.k_min}, {args.k_max}]')
    print(f'{"=" * 60}')

    the_train = TGCtrain.TGC(args, num_workers=args.num_workers)
    the_train.train()

    # After training, sync final K back to args for visualization
    args.clusters = the_train.clusters

    # ============================================================
    # Phase 3: T-SNE Visualization
    # ============================================================
    if args.tsne_viz:
        run_visualization(args, the_train)

    end = datetime.datetime.now()
    print(f'\n{"=" * 60}')
    print(f'Total time: {str(end - start)}')
    print(f'Final K: {args.clusters}')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    data = 'school'

    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--dataset', type=str, default=data)
    parser.add_argument('--clusters', type=int, default=None,
                        help='Initial K (required unless --auto_k True)')

    # TGC training
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--neg_size', type=int, default=5)
    parser.add_argument('--hist_len', type=int, default=3)
    parser.add_argument('--save_step', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--emb_size', type=int, default=128)
    parser.add_argument('--directed', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader worker processes (default: 4)')

    # Dynamic K-selection (in-training, on TGC embeddings)
    parser.add_argument('--dynamic_k', type=lambda x: x.lower() == 'true', default=True,
                        help='Enable dynamic K-selection inside TGC training (default: True)')
    parser.add_argument('--k_warmup_epochs', type=int, default=None,
                        help='Epochs before first in-training K-selection (default: epoch // 5)')
    parser.add_argument('--k_interval', type=int, default=None,
                        help='Epochs between K re-evaluations (default: max(epoch // 10, 3))')

    # Standalone K-selection (optional initial guess on Node2Vec, before TGC)
    parser.add_argument('--auto_k', type=lambda x: x.lower() == 'true', default=False,
                        help='Run standalone RGC K-selection on Node2Vec before TGC (default: False)')
    parser.add_argument('--k_min', type=int, default=None,
                        help='Minimum K to search')
    parser.add_argument('--k_max', type=int, default=None,
                        help='Maximum K to search')
    parser.add_argument('--k_step', type=int, default=1,
                        help='Step size between candidate K values')
    parser.add_argument('--k_selection_epochs', type=int, default=200,
                        help='Epochs for standalone RGC K-selection')
    parser.add_argument('--k_selection_pca', type=int, default=None,
                        help='PCA reduction dim for K-selection')
    parser.add_argument('--k_selection_epsilon', type=float, default=0.5,
                        help='Starting epsilon for epsilon-greedy')
    parser.add_argument('--k_selection_seed', type=int, default=42,
                        help='Random seed for K-selection reproducibility')
    parser.add_argument('--reward_k_penalty', type=float, default=0.3,
                        help='K penalty: reward = (sep-compact) / K^p')
    parser.add_argument('--q_discount', type=float, default=0.5,
                        help='Q-learning discount factor')
    parser.add_argument('--kmeans_chunk_size', type=int, default=2000,
                        help='Chunk size for GPU K-Means')

    # Visualization
    parser.add_argument('--tsne_viz', type=lambda x: x.lower() == 'true', default=True,
                        help='Generate T-SNE visualization after training (default: True)')

    args = parser.parse_args()

    main_train(args)
