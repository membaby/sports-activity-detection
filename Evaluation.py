import numpy as np
import math

def get_contingency_matrix(clustering, truth, n_classes, n_clusters):
    cont_mat = np.zeros((n_classes, n_clusters))
    for i in range(len(clustering)):
        xi = truth[i]
        yi = clustering[i]
        cont_mat[xi][yi] += 1
    return cont_mat

def conditional_entropy(clustering, truth, n_classes, n_clusters):
    cont_mat = get_contingency_matrix(clustering, truth, n_classes, n_clusters)
    entropy =  0
    for j in range(n_classes):
        for i in range(n_clusters):
            pij = cont_mat[j][i] / float(len(clustering))
            pci = sum(cont_mat[:, i]) / float(len(clustering))
            if pij == 0: continue
            entropy -= pij*math.log2(pij / pci)

    return entropy

def f_measure(clustering, truth, n_classes, n_clusters):
    cont_mat = get_contingency_matrix(clustering, truth, n_classes, n_clusters)
    fs = np.empty(n_clusters)
    for i in range(n_clusters):
        max_j = -1
        for j in range(n_classes):
            if (max_j == -1 or cont_mat[j,i] > cont_mat[max_j,i]): max_j = j
        prec = cont_mat[max_j,i] / float(sum(cont_mat[:, i]))
        recall = cont_mat[max_j,i] / float(sum(cont_mat[max_j]))
        fs[i] = 2*prec*recall / (prec + recall)
    return sum(fs) / float(n_clusters)       