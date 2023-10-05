# CLUSTERING
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import warnings

# Suppress FutureWarnings from sklearn
warnings.simplefilter(action='ignore', category=FutureWarning)

def calculate_angle(a, b, c):
	ba = a - b  # vector from b to a
	bc = c - b  # vector from b to c
	cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
	angle = np.arccos(cosine_angle)
	return np.degrees(angle)

def elbow(embedding_df, random_seed=42, plot_results=False):
	matrix = np.vstack(embedding_df.embedding.values)
	distortions = []
	max_clusters = min(20, len(embedding_df) - 1)
	K_range = range(1, max_clusters + 1)
	for k in K_range:
		kmeanModel = KMeans(n_clusters=k, random_state=random_seed, n_init=10)  # Set seed and multiple initializations
		kmeanModel.fit(matrix)
		distortions.append(kmeanModel.inertia_)

	# Calculating the angles
	angles = []
	for i in range(1, len(distortions)-1):
		a = np.array([K_range[i-1], distortions[i-1]])
		b = np.array([K_range[i], distortions[i]])
		c = np.array([K_range[i+1], distortions[i+1]])
		angles.append(calculate_angle(a, b, c))

	# Finding the elbow
	optimal_k = angles.index(max(angles)) + 2  # +2 because index 0 corresponds to k=2

	# Plotting the elbow graph
	if plot_results:
		plt.figure(figsize=(10,5))
		plt.plot(K_range, distortions, 'bx-')
		plt.xlabel('Number of clusters')
		plt.ylabel('Distortion')
		plt.title('The Elbow Method showing the optimal number of clusters')
		plt.axvline(x=optimal_k, color='r', linestyle='--')
		plt.show()

	return optimal_k

def cluster(embedding_df, k, condition_embedding, plot_results=False):

	embedding_df = embedding_df.copy()
	# Stack them into a matrix for clustering
	matrix = np.vstack(embedding_df.embedding.values)

	# Cluster using K-means
	n_clusters = k
	kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
	kmeans.fit(matrix)
	embedding_df["Cluster"] = kmeans.labels_
	perplexity_value = min(15, len(embedding_df) - 1)  # -1 to ensure it's less than n_samples

	 # Calculate cluster centroids
	centroids = np.array([matrix[embedding_df["Cluster"] == i].mean(axis=0) for i in range(n_clusters)])

	# Find the closest cluster to the condition
	distances_to_condition = np.linalg.norm(centroids - condition_embedding, axis=1)
	closest_cluster = np.argmin(distances_to_condition)

	# Visualize using t-SNE
	if plot_results:
		tsne = TSNE(
			n_components=2, perplexity=perplexity_value, random_state=42, init="random", learning_rate=200
		)
		vis_dims2 = tsne.fit_transform(matrix)

		x = [x for x, y in vis_dims2]
		y = [y for x, y in vis_dims2]

		for category, color in enumerate(["purple", "green", "red", "blue", "yellow"]):
			xs = np.array(x)[embedding_df.Cluster == category]
			ys = np.array(y)[embedding_df.Cluster == category]
			plt.scatter(xs, ys, color=color, alpha=0.3)

			avg_x = xs.mean()
			avg_y = ys.mean()

			plt.scatter(avg_x, avg_y, marker="x", color=color, s=100)
		plt.title("Clusters identified visualized in language 2d using t-SNE")
		plt.show()

	print(f"The cluster closest to the condition is cluster {closest_cluster + 1}.")
  	# Extract embeddings and sentences for the closest cluster
  cluster_embeddings = matrix[embedding_df["Cluster"] == closest_cluster]
  cluster_sentences = embedding_df[embedding_df["Cluster"] == closest_cluster]["sentence"]

  # Calculate distances of each sentence in the cluster to the centroid
  centroid = centroids[closest_cluster]
  distances_to_centroid = np.linalg.norm(cluster_embeddings - centroid, axis=1)

  # Get indices of the 10 most similar sentences
  most_similar_idx = np.argsort(distances_to_centroid)[:10]

  # Extract and print the most similar sentences
  most_similar_sentences = cluster_sentences.iloc[most_similar_idx]
  for sentence in most_similar_sentences:
      print(f"- {sentence}")

  return most_similar_sentences
