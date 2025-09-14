import numpy as np
import cupy as cp

class Particle:
    def __init__(self, n_clusters, centroids_pos):
        self.n_clusters = n_clusters
        self.centroids_pos = cp.sort(centroids_pos)  # Keep centroids_pos as array



    def calculate_fitness(self, data: cp.ndarray):
        distances = self.get_distances(data)
        clusters = cp.argmin(distances, axis=0)
        self.pb_val = self.fitness_function(clusters=clusters, distances=distances)


    def calculate_cluster(self, data: cp.ndarray):
        distances = self.get_distances(data)
        clusters = cp.argmin(distances, axis=0)
        self.pb_val = self.fitness_function(clusters=clusters, distances=distances)
        self.pb_clustering = clusters.copy()

    def get_distances(self, data: cp.ndarray):
            distances = []
            for centroid in self.centroids_pos:
                d = cp.linalg.norm(data - centroid, axis=1)
                distances.append(d)
            distances = cp.array(distances)


            return distances
            """


    def get_distances(self, data: cp.ndarray):
        batch_size = 10000  # Adjusting this based on available memory
        num_batches = int(cp.ceil(len(data) / batch_size))
        distances = cp.empty((self.n_clusters, len(data)), dtype=cp.float32)  # Pre-allocation of memory

        for i in range(self.n_clusters):
            for j in range(num_batches):
                start = j * batch_size
                end = min((j + 1) * batch_size, len(data))
                batch_data = data[start:end]
                distances[i, start:end] = cp.linalg.norm(batch_data - self.centroids_pos[i], axis=1)

        return distances"""

    def fitness_function(self, clusters: cp.ndarray, distances: cp.ndarray) -> float:
        J = 0.0
        for i in range(self.n_clusters):
            p = cp.where(clusters == i)[0]
            if len(p):
                dist = cp.array(distances[i][p])
                d = cp.sum(dist)
                d /= len(p)
                J += d
        J /= self.n_clusters

        return J
