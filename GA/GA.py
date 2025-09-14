import numpy as np
import cupy as cp
import random
from particle import Particle


class GAcluster:
  def __init__(self, n_clusters: int, n_particles: int, data):
      self.n_clusters = n_clusters
      self.n_particles = n_particles
      self.mu_rate = 0.2
      self.data = data
      self.particles = []
      self.gb_pos = None
      self.gb_val = cp.inf
      self.gb_clustering = None
      self.progress = []
      self.max, self.min = cp.max(data), cp.min(data)
      self._generate_particles()

  def _generate_particles(self):
      for _ in range(self.n_particles):
          centroids = self.data[cp.random.choice(list(range(len(self.data))), self.n_clusters)]
          particle = Particle(n_clusters=self.n_clusters, centroids_pos = centroids)
          particle.calculate_fitness(data=self.data)
          self.particles.append(particle)


  def update_gb(self, particle):
      if particle.pb_val < self.gb_val:
          self.gb_val = particle.pb_val
          self.gb_pos = particle.centroids_pos.copy()


  def crossover(self,n, p1, p2):
    x = cp.concatenate(((p1.centroids_pos[:n]), (p2.centroids_pos[n:])))
    y = cp.concatenate(((p2.centroids_pos[:n]), (p1.centroids_pos[n:])))
    x = self.mutation(x)
    y = self.mutation(y)
    c1 = Particle(n_clusters=self.n_clusters, centroids_pos = x)
    c1.calculate_fitness(data=self.data)
    c2 = Particle(n_clusters=self.n_clusters, centroids_pos = y)
    c2.calculate_fitness(data=self.data)
    return(c1, c2)

  def mutation(self, centroids):
    r = random.randint(0, self.n_clusters-1)
    if random.random() < self.mu_rate:
      centroids[r] = random.randint(self.min, self.max-1)
      centroids = cp.sort(centroids)
    return centroids


  def selection(self):
    select_population = []
    for i in range(self.n_particles):
      select_population.append(self.particles[i])
    select_population = sorted(select_population, key=lambda x: x.pb_val)
    return select_population[ : self.n_particles//2]


  def start(self, iteration):
    print("new gen fitness: ")
    for i in range(self.n_particles):
      print(self.particles[i].pb_val, end = "  ")
      self.update_gb(self.particles[i])
    self.progress.append([self.gb_pos, self.gb_clustering, self.gb_val])
    print(self.gb_val)


    for j in range(iteration):
      print("iteration no =", j)
      select_population = self.selection()
      Population = []
      for i in range(self.n_particles):
        r = random.randint(0, len(select_population)-1)
        Population.append(select_population[r])
        c = self.crossover(self.n_clusters//2, self.particles[i], select_population[r])
        Population.append(c[0])
        Population.append(c[1])

      Population = sorted(Population, key=lambda x: x.pb_val)
      z = [i for i in Population]
      new_generation = z[: self.n_particles]
      self.particles = new_generation.copy()

      print("new gen fitness: ")
      for i in range(self.n_particles):
        print(self.particles[i].pb_val, end = "  ")
        self.update_gb(self.particles[i])
      self.progress.append([self.gb_pos, self.gb_val])
      print("best of the population", self.gb_val)

    bestp = Particle(n_clusters=self.n_clusters, centroids_pos = self.gb_pos)
    bestp.calculate_cluster(data=self.data)
    self.gb_clustering = bestp.pb_clustering.copy()
    clusters = self.gb_clustering
    centers = self.gb_pos
    centers = cp.uint8(centers.get())
    res = centers[clusters.get().flatten()]
    print('Completed!\n')
    return res  
