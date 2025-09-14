import numpy as np
import cupy as cp
from particle import Particle


class PSOClustering:
  def __init__(self, n_clusters: int, n_particles: int, data, w=0.1, c1=1.5, c2=1.5):
    self.n_clusters = n_clusters
    self.n_particles = n_particles
    self.data = data
    self.particles = []
    self.gb_pos = None
    self.gb_val = cp.inf
    self.gb_clustering = None
    self.progress = []
    self._generate_particles(w, c1, c2)
    print("New particle..")

  def _generate_particles(self, w: float, c1: float, c2: float):
    for i in range(self.n_particles):
      particle = Particle(n_clusters=self.n_clusters, data=self.data, w=w, c1=c1, c2=c2)
      self.particles.append(particle)

  def update_gb(self, particle):
    if particle.pb_val < self.gb_val:
      self.gb_val = particle.pb_val
      self.gb_pos = particle.pb_pos.copy()
      #self.gb_clustering = particle.pb_clustering.copy()


  def start(self, iteration):
    for i in range(iteration):
      print("iteration no =", i)
      for particle in self.particles:
        particle.update_pb(data=self.data)
        self.update_gb(particle=particle)

      for particle in self.particles:
        particle.move_centroids(gb_pos=self.gb_pos)
      self.progress.append([self.gb_pos, self.gb_val])
      print(self.gb_val)

    bestp = Particle(n_clusters=self.n_clusters, data = self.data)
    bestp.centroids_pos = self.gb_pos.copy()
    bestp.clustering(data=self.data)
    self.gb_clustering = bestp.pb_clustering.copy()

    clusters = self.gb_clustering
    centers = self.gb_pos
    centers = cp.uint8(centers.get())
    res = centers[clusters.get().flatten()]
    #print(progress)



    print('Completed!\n')
    return res, self.gb_val

