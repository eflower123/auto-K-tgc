import torch
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def random_init(X, num_clusters):
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state


def pairwise_distance(data1, data2, device=torch.device('cuda')):
    """Compute pairwise squared Euclidean distance: output [N1, N2]."""
    data1, data2 = data1.to(device), data2.to(device)
    A = data1.unsqueeze(dim=1)
    B = data2.unsqueeze(dim=0)
    dis = (A - B) ** 2.0
    dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_distance_chunked(X, centers, chunk_size=2000, device=torch.device('cuda')):
    """
    Memory-efficient pairwise distance using chunked computation.
    Instead of [N, K, D] broadcast, processes X in chunks of [chunk, K, D].
    """
    X = X.to(device).float()
    centers = centers.to(device).float()
    N = X.size(0)
    dis_list = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = X[start:end]
        d = ((chunk.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(dim=-1)
        dis_list.append(d)
    return torch.cat(dis_list, dim=0)


def pairwise_cosine(data1, data2, device=torch.device('cuda')):
    data1, data2 = data1.to(device), data2.to(device)
    A = data1.unsqueeze(dim=1)
    B = data2.unsqueeze(dim=0)
    A_normalized = A / A.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    B_normalized = B / B.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cosine = A_normalized * B_normalized
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis


def kmeans_plusplus_init(X, num_clusters, device, chunk_size=2000):
    n_samples, n_features = X.shape
    centers = torch.empty((num_clusters, n_features), dtype=X.dtype).to(device)
    center_id = np.random.choice(n_samples, 1, replace=False)[0]
    n_local_trials = 2 + int(np.log(num_clusters))

    indices = np.full(num_clusters, -1, dtype=int)
    centers[0] = X[center_id]
    indices[0] = center_id

    # Use chunked if N is large
    use_chunked = n_samples > chunk_size

    if use_chunked:
        closest_dist_sq = pairwise_distance_chunked(X, centers[0, np.newaxis], chunk_size=chunk_size, device=device).squeeze(-1)
    else:
        closest_dist_sq = pairwise_distance(centers[0, np.newaxis], X, device=device)
    current_pot = closest_dist_sq.sum().item()

    for c in range(1, num_clusters):
        rand_vals = torch.rand(n_local_trials).to(device) * current_pot
        candidate_ids = torch.searchsorted(torch.cumsum(closest_dist_sq, dim=0), rand_vals)
        torch.clip(candidate_ids, None, len(closest_dist_sq) - 1, out=candidate_ids)

        # Compute distances from candidate points to all X (chunked for large N)
        if use_chunked:
            distance_to_candidates_list = []
            for start in range(0, n_samples, chunk_size):
                end = min(start + chunk_size, n_samples)
                chunk = X[start:end]
                d = pairwise_distance(X[candidate_ids], chunk, device=device)
                distance_to_candidates_list.append(d)
            distance_to_candidates = torch.cat(distance_to_candidates_list, dim=1)
        else:
            distance_to_candidates = pairwise_distance(X[candidate_ids], X, device=device)

        torch.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        best_candidate = torch.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate].item()
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        centers[c] = X[best_candidate]
        indices[c] = best_candidate
    return centers, indices


def kmeans(
        X,
        num_clusters,
        distance='euclidean',
        tol=1e-4,
        init_mode="kmeans++",
        init_time=20,
        device=torch.device('cuda'),
        chunk_size=2000,
        max_iter=300,
):
    if distance == 'euclidean':
        pairwise_fn = pairwise_distance
        pairwise_chunked_fn = pairwise_distance_chunked
    elif distance == 'cosine':
        pairwise_fn = pairwise_cosine
        pairwise_chunked_fn = None  # cosine not easily chunkable, fall back
    else:
        raise NotImplementedError

    X = X.float().to(device)
    N = X.size(0)

    # Use chunked distance for large N
    use_chunked = (pairwise_chunked_fn is not None and N > chunk_size and num_clusters > 5)

    dis_min = float('inf')
    initial_state_best = None
    for i in range(init_time):
        if init_mode == "kmeans++":
            initial_state, _ = kmeans_plusplus_init(X, num_clusters, device=device, chunk_size=chunk_size)
        elif init_mode == "random":
            initial_state = random_init(X, num_clusters)

        if use_chunked:
            dis = pairwise_chunked_fn(X, initial_state, chunk_size=chunk_size, device=device).sum()
        else:
            dis = pairwise_fn(X, initial_state, device=device).sum()
        if dis < dis_min:
            dis_min = dis
            initial_state_best = initial_state

    initial_state = initial_state_best
    iteration = 0
    while True:
        if use_chunked:
            dis = pairwise_chunked_fn(X, initial_state, chunk_size=chunk_size, device=device)
        else:
            dis = pairwise_fn(X, initial_state, device=device)
        choice_cluster = torch.argmin(dis, dim=1)
        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected_mask = (choice_cluster == index)
            selected_count = selected_mask.sum().item()
            if selected_count == 0:
                rand_idx = torch.randint(0, N, (1,)).item()
                initial_state[index] = X[rand_idx]
            else:
                initial_state[index] = X[selected_mask].mean(dim=0)

        center_shift = torch.sqrt(
            torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
        ).sum()

        iteration += 1

        if iteration > max_iter:
            break
        if center_shift ** 2 < tol:
            break
    return choice_cluster.cpu(), initial_state, dis


def kmeans_predict(
        X,
        cluster_centers,
        distance='euclidean',
        device=torch.device('cuda'),
        chunk_size=2000,
):
    if distance == 'euclidean':
        pairwise_fn = pairwise_distance
        pairwise_chunked_fn = pairwise_distance_chunked
    elif distance == 'cosine':
        pairwise_fn = pairwise_cosine
        pairwise_chunked_fn = None
    else:
        raise NotImplementedError

    X = X.float().to(device)
    N = X.size(0)
    use_chunked = (pairwise_chunked_fn is not None and N > chunk_size and cluster_centers.size(0) > 5)

    if use_chunked:
        dis = pairwise_chunked_fn(X, cluster_centers, chunk_size=chunk_size, device=device)
    else:
        dis = pairwise_fn(X, cluster_centers, device=device)
    choice_cluster = torch.argmin(dis, dim=1)

    return choice_cluster.cpu()
