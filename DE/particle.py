import numpy as np

class Particle:
    def __init__(self, n_clusters, data, centroids_pos):
        self.n_clusters = n_clusters
        self.centroids_pos = centroids_pos
        distances = self.get_distances(data=data)
        clusters = np.argmin(distances, axis=0)
        self.pb_val = self.fitness_function(clusters=clusters, distances=distances)
        self.pb_clustering = clusters.copy()
        

    
    def get_distances(self, data: np.ndarray):
            distances = []
            for centroid in self.centroids_pos:
                d = np.linalg.norm(data - centroid, axis=1)
                distances.append(d)
            distances = np.array(distances)
            
                
            return distances
    

    def fitness_function(self, clusters: np.ndarray, distances: np.ndarray) -> float:
        J = 0.0
        for i in range(self.n_clusters):
            p = np.where(clusters == i)[0]
            if len(p):
                d = sum(distances[i][p])
                d /= len(p)
                J += d
        J /= self.n_clusters
        return J
    