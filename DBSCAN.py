import numpy as np


def euclid_distance_matrix(data):
    mat = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(i+1, data.shape[0]):
            mat[i, j] = np.linalg.norm(data[i] - data[j])
            mat[j, i] = mat[i, j]
    return mat



def DBSCAN(data, eps, min_pts):
    adj_mat = euclid_distance_matrix(adj_mat)
    n = adj_mat.shape[0]
    clustering = np.zeros(n)
    core = set()
    neighborhoods = []
    #Find core points
    for i in range(n):
        clustering[i] = -1
        neighborhoods.append(np.where(adj_mat[i] <= eps)[0])
        if len(neighborhoods[i]) >= min_pts:
            core.add(i)
            
    #Expand clusters
    k = 0
    for x_idx in core:
        if clustering[x_idx] != -1:
            continue
        clustering[i] = k
        k += 1
        #Add density connected points to cluster
        density_connected(x_idx, k, neighborhoods, clustering, core)
    
    #Clustering outliers
    outlier_count = 0
    for i in range(n):
        if clustering[i] == -1:
            outlier_count += 1
            nearest = 0
            for j in range(1, n):
                if clustering[j] != -1:
                    if adj_mat[i, j] < adj_mat[i, nearest]: nearest = j
            clustering[i] = clustering[nearest]
    print(f"DBSCAN: {outlier_count} outliers were clustered with 1NN.")
    
    return clustering, k #Clustered data, number of clusters         


def density_connected(x_idx, k, neighborhoods, clustering, core):
    for j in neighborhoods[x_idx]:
            if clustering[j] == -1:
                clustering[j] = k
                if j in core:
                    density_connected(j, k, neighborhoods, clustering, core)
    