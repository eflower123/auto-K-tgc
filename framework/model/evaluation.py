import numpy as np
from munkres import Munkres
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import torch

# Lazy import GPU K-Means (path adjusted for both model/ and experiments/ usage)
_gpu_kmeans = None


def _get_gpu_kmeans():
    global _gpu_kmeans
    if _gpu_kmeans is None and torch.cuda.is_available():
        import sys
        import os
        _sys_path = os.path.dirname(os.path.abspath(__file__))
        _kmeans_path = os.path.join(_sys_path, '..', 'k_selection')
        if _kmeans_path not in sys.path:
            sys.path.insert(0, _kmeans_path)
        try:
            from kmeans_gpu import kmeans as gpu_kmeans_fn, kmeans_predict as gpu_predict_fn
            _gpu_kmeans = (gpu_kmeans_fn, gpu_predict_fn)
        except ImportError:
            _gpu_kmeans = False
    return _gpu_kmeans


def evaluation(y_true, y_pred):
    """
    evaluate the clustering performance
    :param y_true: ground truth
    :param y_pred: prediction
    :returns acc, nmi, ari, f1:
    - accuracy
    - normalized mutual information
    - adjust rand index
    - f1 score
    """
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)

    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    if num_class1 != num_class2:
        # K mismatch: skip Munkres alignment, NMI/ARI still valid
        return
    cost = np.zeros((num_class1, num_class2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = accuracy_score(y_true, new_predict)
    f1 = f1_score(y_true, new_predict, average='macro')

    return acc, nmi, ari, f1


def eva(k, labels, emb, use_gpu=True):
    embeddings = emb.cpu().detach().numpy()
    gpu_kmeans = _get_gpu_kmeans() if use_gpu else None

    if gpu_kmeans:
        gpu_kmeans_fn, _ = gpu_kmeans
        X = torch.FloatTensor(embeddings)
        device = torch.device('cuda')
        cluster_id, _, _ = gpu_kmeans_fn(
            X, num_clusters=k, distance='euclidean',
            init_mode='kmeans++', init_time=20, device=device,
        )
        cluster_id = cluster_id.numpy()
    else:
        model = KMeans(n_clusters=k, n_init=20)
        cluster_id = model.fit_predict(embeddings)

    result = evaluation(labels, cluster_id)
    if result is None:
        nmi = nmi_score(labels, cluster_id, average_method='arithmetic')
        ari = ari_score(labels, cluster_id)
        return 0.0, nmi, ari, 0.0
    acc, nmi, ari, f1 = result
    return acc, nmi, ari, f1