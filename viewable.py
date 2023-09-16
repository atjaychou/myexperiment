import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Get the features of each cluster
cluster_features = []
for i in range(num_clusters):
    indices = np.where(cluster_labels == i)[0]
    features = train_features[indices]
    cluster_features.append(features)

# Concatenate the features of all clusters
all_features = np.concatenate(cluster_features, axis=0)

# Apply PCA to reduce the dimensionality of the features
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(all_features)

# Plot the reduced features, colored by cluster
color_list = plt.cm.Set1(np.linspace(0, 1, num_clusters))
start_index = 0
for i in range(num_clusters):
    num_samples = cluster_features[i].shape[0]
    end_index = start_index + num_samples
    plt.scatter(reduced_features[start_index:end_index, 0], reduced_features[start_index:end_index, 1], c=color_list[i], label='Cluster {}'.format(i))
    start_index = end_index

plt.legend()
plt.show()
