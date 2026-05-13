import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def tsne_visualize(
    embeddings,
    labels,
    save_path,
    k=None,
    title=None,
    figsize=(10, 8),
    perplexity=30,
    random_state=42,
    max_samples=None,
):
    """
    Generate T-SNE visualization of node embeddings colored by ground-truth labels.

    Args:
        embeddings: np.ndarray [N, D] or torch.Tensor
        labels: np.ndarray [N] — ground truth cluster labels
        save_path: str — path to save figure (without extension)
        k: int | None — selected K to display in title
        title: str | None — custom title
        figsize: tuple — figure size
        perplexity: int — T-SNE perplexity
        random_state: int — random seed
        max_samples: int | None — max nodes to visualize (subsample if needed)

    Returns:
        tsne_embeddings: np.ndarray [N, 2] — 2D T-SNE coordinates
    """
    # Handle torch tensors
    if hasattr(embeddings, 'cpu'):
        embeddings = embeddings.cpu().data.numpy()

    # Subsample large datasets
    if max_samples is not None and embeddings.shape[0] > max_samples:
        idx = np.random.RandomState(random_state).choice(
            embeddings.shape[0], max_samples, replace=False
        )
        embeddings = embeddings[idx]
        labels = np.array(labels)[idx]

    n_nodes = embeddings.shape[0]

    # Adjust perplexity for small datasets
    effective_perplexity = min(perplexity, max(5, n_nodes // 3))

    # Run T-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=random_state,
        n_iter=1000,
        verbose=0,
    )
    tsne_emb = tsne.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    n_clusters = len(np.unique(labels))
    scatter = ax.scatter(
        tsne_emb[:, 0], tsne_emb[:, 1],
        c=labels, cmap='tab20', s=5, alpha=0.7, linewidths=0
    )

    # Title
    if title is None:
        title = 'T-SNE Visualization of Node Embeddings'
        if k is not None:
            title += f' (K={k})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('T-SNE Dimension 1')
    ax.set_ylabel('T-SNE Dimension 2')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(n_clusters))
    cbar.set_label('Cluster Label', fontsize=11)

    plt.tight_layout()

    # Save both PDF and PNG
    fig.savefig(save_path + '.pdf', dpi=150, bbox_inches='tight')
    fig.savefig(save_path + '.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'[Visualization] T-SNE plot saved to {save_path}.pdf / .png')
    print(f'[Visualization] Nodes={n_nodes}, Clusters={n_clusters}, Perplexity={effective_perplexity}')

    return tsne_emb


def tsne_visualize_dual(
    embeddings,
    labels,
    pred_labels,
    save_path,
    k=None,
    figsize=(18, 8),
    perplexity=30,
    random_state=42,
    max_samples=None,
):
    """
    Dual-panel T-SNE: left colored by ground truth, right colored by predicted clusters.

    Args:
        embeddings: np.ndarray [N, D] or torch.Tensor
        labels: np.ndarray [N] — ground truth labels
        pred_labels: np.ndarray [N] — predicted cluster assignments
        save_path: str — path to save figure (without extension)
        k: int | None — selected K
        figsize: tuple — figure size
        perplexity: int — T-SNE perplexity
        random_state: int — random seed
        max_samples: int | None — max nodes to visualize

    Returns:
        tsne_embeddings: np.ndarray [N, 2]
    """
    if hasattr(embeddings, 'cpu'):
        embeddings = embeddings.cpu().data.numpy()

    if max_samples is not None and embeddings.shape[0] > max_samples:
        idx = np.random.RandomState(random_state).choice(
            embeddings.shape[0], max_samples, replace=False
        )
        embeddings = embeddings[idx]
        labels = np.array(labels)[idx]
        pred_labels = np.array(pred_labels)[idx]

    n_nodes = embeddings.shape[0]
    effective_perplexity = min(perplexity, max(5, n_nodes // 3))

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=random_state,
        n_iter=1000,
        verbose=0,
    )
    tsne_emb = tsne.fit_transform(embeddings)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    n_true = len(np.unique(labels))
    n_pred = len(np.unique(pred_labels))

    ax1.scatter(
        tsne_emb[:, 0], tsne_emb[:, 1],
        c=labels, cmap='tab20', s=5, alpha=0.7, linewidths=0
    )
    title1 = f'Ground Truth ({n_true} classes)'
    if k is not None:
        title1 += f' [K={k}]'
    ax1.set_title(title1, fontsize=13, fontweight='bold')
    ax1.set_xlabel('T-SNE 1')
    ax1.set_ylabel('T-SNE 2')

    ax2.scatter(
        tsne_emb[:, 0], tsne_emb[:, 1],
        c=pred_labels, cmap='tab20', s=5, alpha=0.7, linewidths=0
    )
    ax2.set_title(f'Predicted Clusters ({n_pred} clusters)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('T-SNE 1')
    ax2.set_ylabel('T-SNE 2')

    plt.tight_layout()
    fig.savefig(save_path + '.pdf', dpi=150, bbox_inches='tight')
    fig.savefig(save_path + '.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'[Visualization] Dual T-SNE saved to {save_path}.pdf / .png')
    print(f'[Visualization] Nodes={n_nodes}, True classes={n_true}, Pred clusters={n_pred}')

    return tsne_emb
