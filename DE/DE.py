import numpy as np
import cupy as cp
import random
from particle import Particle



class DEcluster:
    def __init__(self, n_clusters: int, n_particles: int, data):
        self.n_clusters = n_clusters
        self.n_particles = n_particles
        self.mu_rate = 0.5
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
            self.update_gb(particle)


    def update_gb(self, particle):
        if particle.pb_val < self.gb_val:
            self.gb_val = particle.pb_val
            self.gb_pos = particle.centroids_pos.copy()


    def mutation(self):
        mutated_list = []
        l = len(self.particles)
        for i in range(l):
          idxs = [idx for idx in range(l) if idx != i]
          selected = np.random.choice(idxs, 3, replace=False)
          pbest = self.gb_pos
          p1 = self.particles[selected[0]].centroids_pos
          p2 = self.particles[selected[0]].centroids_pos
          p3 = self.particles[selected[1]].centroids_pos

          y = p1 + cp.rint(self.mu_rate * cp.abs(p2 - p3)) #+ cp.rint(self.mu_rate * cp.abs(pbest - p1))
          y = cp.clip(y, self.min, self.max)
          mutated_list.append(y)

        return mutated_list



    def start(self, iteration, crossover_rate = 0.9):
        progress = []
        for j in range(iteration):
            print("iteration no =", j)

            mutation_list = self.mutation()


            Population = []
            for i in range(self.n_particles):
                cross_points = cp.random.rand(self.n_clusters) < crossover_rate
                cross_points_reshaped = cross_points.reshape(-1, 1)
                trial = cp.where(cross_points_reshaped, mutation_list[i], self.particles[i].centroids_pos)
                mutated_particle = Particle(n_clusters=self.n_clusters, centroids_pos = trial)
                mutated_particle.calculate_fitness(data=self.data)

                if(mutated_particle.pb_val<self.particles[i].pb_val):
                  print("change detected! updating", self.particles[i].pb_val, mutated_particle.pb_val)
                  self.particles[i] = mutated_particle

                self.update_gb(self.particles[i])

            self.progress.append([self.gb_pos, self.gb_val])

            print("fitness global", self.gb_val)

        bestp = Particle(n_clusters=self.n_clusters, centroids_pos = self.gb_pos)
        bestp.calculate_cluster(data=self.data)
        self.gb_clustering = bestp.pb_clustering.copy()

        clusters = self.gb_clustering
        centers = self.gb_pos
        centers = cp.uint8(centers.get())
        res = centers[clusters.get().flatten()]

        print('Completed!\n')
        return res, self.gb_val
