import numpy as np
import matplotlib.pyplot as plt

def k_mean(samples, num_clusters, stop_epsion= 1e-10, max_iter=100):
    """
    K-Mean Cluster Implementation
    :param samples: samples with dimension of (num_points, 2)
    :param num_clusters: number of clusters
    :param stop_epsion: stop condition
    :param max_iter: allowed max iteration
    :return cluster_loc: center of each cluster
    :return sample_cluster_index: cluster indices for each sample points 
    """

    # Cluster indices
    index_k = np.zeros(samples.shape[0], dtype=np.int)

    # Distance cache, dim: (num_clusters, num_samples)
    distance = np.zeros((num_clusters, samples.shape[0]), dtype=np.float32) 
    # Step 1: Random choose initial points as cluster center
    random_indices = np.arange(0, samples.shape[0], dtype=np.int)
    np.random.shuffle(random_indices)
    cluster_loc = samples[random_indices[:num_clusters], :]
    old_distance_var = -10000
    samples_k = np.stack([samples]*num_clusters,axis = 0)
    # Step 2:Iteration
    for itr in range(0, max_iter):
        # Instruction: Fill the following blanks with your implementation, you should finish the implementation with less than 25 lines of code. 
        
        # Compute the distance towards the cluster center, you can use 'np.linalg.norm' to compute L2 distance
        cluster_loc_k = np.stack([cluster_loc]*samples.shape[0],axis = 1)
        distance = np.linalg.norm(samples_k - cluster_loc_k, axis = 2)
        
        # For each sample point, set the cluster center with minimal distance, tip: use np.argmin to find the index that has minimal value 
        index_k = np.argmin(distance, axis = 0)
        
        # Re-compute the distance by average the cluster sampled points, and update the 'cluster_loc'
        avg_distance_var = 0
        distance_var = np.zeros((num_clusters,1),dtype=np.float32)
        for j in range(num_clusters):
            samples_j = samples[np.where(index_k == j)[0], : ]
            cluster_loc[j, :] = np.mean(samples_j,axis = 0) 
            distance_j = np.linalg.norm(samples_j - np.stack([cluster_loc[j,:]]*samples_j.shape[0]),axis = 1)
            distance_var[j] = np.var(distance_j)

        # Compute total avg. distance variance
        avg_distance_var = np.average(distance_var)
        

        # Check if the avg. distance variance has converged
        if np.abs(avg_distance_var - old_distance_var) < stop_epsion:
            break

        print("Itr %d, avg. distance variance: %f" % (itr, avg_distance_var))
        old_distance_var = avg_distance_var
    return cluster_loc, index_k 


# Load the sample points
points = np.load('k-mean_samples.npy')

# Do K-Mean cluster
cluster_loc, cluster_indices = k_mean(points, num_clusters=6)

# Draw the clusters
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'w']
for cluster_idx in range(0, cluster_loc.shape[0]):
    sub_sample_set = points[cluster_indices == cluster_idx]
    plt.scatter(sub_sample_set[:, 0], sub_sample_set[:, 1], c=colors[cluster_idx], label='group %d' % cluster_idx)

plt.scatter(cluster_loc[:, 0], cluster_loc[:, 1], c='k', label='center')
plt.legend()
plt.grid(True)
plt.title("K-Mean Cluster (%d centers)" % cluster_loc.shape[0])
plt.show()