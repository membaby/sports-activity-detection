import numpy as np
from sklearn.metrics import pairwise_distances

def DBSCAN(data, eps, min_pts):
    adj_mat = pairwise_distances(data, metric='euclidean')
    n = adj_mat.shape[0]
    clustering = np.zeros(n, dtype=np.int64)
    core = set()
    neighborhoods = []
    #Find core points and neighborhoods
    for i in range(n):
        clustering[i] = -1
        neighbors = np.where(adj_mat[i] <= eps)[0]
        neighbors = neighbors[neighbors != i]
        neighborhoods.append(neighbors)
        if len(neighborhoods[i]) >= min_pts:
            core.add(i)
    #Expand clusters
    k = 0
    for x_idx in core:
        if clustering[x_idx] != -1:
            continue
        clustering[i] = k
        #Add density connected points to cluster
        density_connected(x_idx, k, neighborhoods, clustering, core)
        k += 1
    #Clustering outliers
    outlier_count = 0
    for i in range(n):
        if clustering[i] == -1:
            outlier_count += 1
            nearest = -1
            for j in range(0, n):
                if clustering[j] != -1:
                    if nearest == -1 or adj_mat[i, j] < adj_mat[i, nearest]: nearest = j
            clustering[i] = clustering[nearest]
    print(f"DBSCAN: {outlier_count} outliers were clustered with 1NN.")
    return clustering, k, outlier_count #Clustered data, number of clusters  , number of outliers       


def density_connected(x_idx, k, neighborhoods, clustering, core):
    queue = [(x_idx, k)]
    while len(queue) > 0:
        x_idx, k = queue.pop(0)
        for j in neighborhoods[x_idx]:
                if clustering[j] == -1:
                    clustering[j] = k
                    if j in core:
                        queue.append((j, k))