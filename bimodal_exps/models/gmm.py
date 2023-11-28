from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.metrics import pairwise_distances_argmin
import torch
from sklearn.preprocessing import StandardScaler

"""
Compute centroids with a Gaussian Mixture Model initialized with K-Means.
Avoids convergence to a local minimum and improves clusters generated with K-Means.
"""

"""
Maybe run this function only once at the beginning of the first batch
"""
# Determine the optimal k and return the model initialized with it
def get_gmm_with_optimal_k(data, max_k):
    bic_scores = [0] * (max_k)
    bic_scores[0] = 1e10
    for k in range(2, max_k+1):
        gmm = GaussianMixture(n_components=k, tol=1e-4, init_params='kmeans')
        gmm.fit(data)
        bic_scores.append(gmm.bic(data))
    
    optimal_k = np.argmin(bic_scores) + 1
    # print(optimal_k)
    final_gmm = GaussianMixture(n_components=optimal_k, tol=1e-4, init_params='kmeans')
    final_gmm.fit(data)

    return final_gmm

def gmm_centroids(image_feat, text_feat, max_k=20):
    image_feat_std = StandardScaler().fit_transform(image_feat.cpu().numpy())
    text_feat_std = StandardScaler().fit_transform(text_feat.cpu().numpy())

    gmm_image = get_gmm_with_optimal_k(image_feat_std, max_k)
    gmm_text = get_gmm_with_optimal_k(text_feat_std, max_k)

    image_centroids = gmm_image.means_
    text_centroids = gmm_text.means_

    image_cluster_idxs = pairwise_distances_argmin(image_feat_std, image_centroids)
    text_cluster_idxs = pairwise_distances_argmin(text_feat_std, text_centroids)

    image_centroids = torch.from_numpy(image_centroids).to(image_feat.device)
    text_centroids = torch.from_numpy(text_centroids).to(text_feat.device)
    image_cluster_idxs = torch.from_numpy(image_cluster_idxs).to(image_feat.device)
    text_cluster_idxs = torch.from_numpy(text_cluster_idxs).to(text_feat.device)

    return image_centroids, image_cluster_idxs, text_centroids, text_cluster_idxs


if __name__ == '__main__':
    image_feats = np.random.rand(20, 2048)
    text_feats = np.random.rand(20, 768)
    image_centroids, image_cluster_idxs, text_centroids, text_cluster_idxs = gmm_centroids(image_feats, text_feats, 4)
    print(image_centroids)
    print(image_cluster_idxs)
    print(text_centroids)
    print(text_cluster_idxs)

