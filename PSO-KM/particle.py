import cupy as cp
import numpy as np

class Particle:
    def __init__(self, n_clusters, data, w=1.8, c1=1.5, c2=1.5):
        self.n_clusters = n_clusters
        self.centroids_pos = data[cp.random.choice(list(range(len(data))), self.n_clusters)]

        self.pb_val = cp.inf
        self.pb_pos = self.centroids_pos.copy()
        self.velocity = cp.zeros_like(self.centroids_pos) + cp.random.uniform(-1,1)
        self.pb_clustering = None
        self.w = w
        self.c1 = c1
        self.c2 = c2


    def update_pb(self, data: cp.ndarray):
        distances = self._get_distances(data=data)
        clusters = cp.argmin(distances, axis=0)

        new_val = self._fitness_function(clusters=clusters, distances=distances)
        if new_val < self.pb_val:
            self.pb_val = new_val
            self.pb_pos = self.centroids_pos.copy()
            #self.pb_clustering = clusters.copy()

    def clustering(self, data: cp.ndarray):
        distances = self._get_distances(data=data)
        clusters = cp.argmin(distances, axis=0)
        new_val = self._fitness_function(clusters=clusters, distances=distances)
        self.pb_val = new_val
        self.pb_pos = self.centroids_pos.copy()
        self.pb_clustering = clusters.copy()

    """def _get_distances(self, data: cp.ndarray):
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


    def _get_distances(self, data: cp.ndarray):
            distances = []
            for centroid in self.centroids_pos:
                d = cp.linalg.norm(data - centroid, axis=1)
                distances.append(d)
            distances = cp.array(distances)


            return distances

    def update_velocity(self, gb_pos: cp.ndarray):
        self.velocity = self.w * self.velocity + self.c1 * cp.random.random() * (self.pb_pos - self.centroids_pos) + self.c2 * cp.random.random() * (gb_pos - self.centroids_pos)
        #print(self.velocity)

    def move_centroids(self, gb_pos):
        self.update_velocity(gb_pos=gb_pos)
        new_pos = self.centroids_pos + self.velocity
        self.centroids_pos = new_pos.copy()

    def _fitness_function(self, clusters: cp.ndarray, distances: cp.ndarray) -> float:
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
