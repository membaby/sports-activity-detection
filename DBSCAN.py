import numpy as np

test = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
arr = []
for i in range(test.shape[0]):
    condition = np.where(test[i] > 5)[0]
    arr.append(condition)
print(arr)
for i in range(len(arr)):
    print(test[i, arr[i]])



def euclid_distance_matrix(data):
    mat = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(i+1, data.shape[0]):
            mat[i, j] = np.linalg.norm(data[i] - data[j])
            mat[j, i] = mat[i, j]
    return mat


def rbf_matrix(data, gamma):
    return np.exp(-gamma * euclid_distance_matrix(data)**2)


def DBSCAN(data, eps, min_pts, distance_metric='euclidean', gamma=1.0):
    using_rbf = False
    if (distance_metric == 'euclidean'):
        adj_mat = euclid_distance_matrix(adj_mat)
    else:
        adj_mat = rbf_matrix(data, gamma)
        using_rbf = True
        #eps is the euclidean threshold for density connected points. Must be converted to RBF distance
        eps = np.exp(-gamma * eps**2)
    
    n = adj_mat.shape[0]
    clustering = np.zeros(n)
    core = set()
    neighborhoods = []
    #Find core points
    for i in range(n):
        clustering[i] = -1
        if not using_rbf:
            neighborhoods.append(np.where(adj_mat[i] <= eps)[0])
        else: neighborhoods.append(np.where(adj_mat[i] >= eps)[0])
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
                    if not using_rbf and adj_mat[i, j] < adj_mat[i, nearest]: nearest = j
                    elif using_rbf and adj_mat[i, j] > adj_mat[i, nearest]: nearest = j
            clustering[i] = clustering[nearest]
    print(f"DBSCAN: {outlier_count} outliers were clustered with 1NN.")
    
    return clustering            


def density_connected(x_idx, k, neighborhoods, clustering, core):
    for j in neighborhoods[x_idx]:
            if clustering[j] == -1:
                clustering[j] = k
                if j in core:
                    density_connected(j, k, neighborhoods, clustering, core)
    