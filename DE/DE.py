import numpy as np
import random
from particle import Particle


class DEcluster:
    def __init__(self, n_clusters: int, n_particles: int, data):
        self.n_clusters = n_clusters
        self.n_particles = n_particles
        self.data = data
        self.particles = []
        self.gb_pos = None
        self.gb_val = np.inf
        self.gb_clustering = None
        self._generate_particles()

    def _generate_particles(self):
        for i in range(self.n_particles):
            centroids = self.data[np.random.choice(list(range(len(self.data))), self.n_clusters)]
            particle = Particle(n_clusters=self.n_clusters, data=self.data, centroids_pos = centroids)
            self.particles.append(particle)

    
    def update_gb(self, particle):
        if particle.pb_val < self.gb_val:
            self.gb_val = particle.pb_val
            self.gb_pos = particle.centroids_pos.copy()
            self.gb_clustering = particle.pb_clustering.copy()
    
    def crossover(self,n, p1, p2):
      x = np.concatenate(((p1.centroids_pos[:n]), (p2.centroids_pos[n:])))
      y = np.concatenate(((p2.centroids_pos[:n]), (p1.centroids_pos[n:])))
      c1 = Particle(n_clusters=self.n_clusters, data=self.data, centroids_pos = x)
      c2 = Particle(n_clusters=self.n_clusters, data=self.data, centroids_pos = y)
      return(c1, c2)
    
    def mutation(self):
        mutated_particle_list = []
        l = len(self.particles)
        for i in range(l):
            x = self.particles[i].centroids_pos
            
            p1 = self.particles[random.randrange(l)]
            p2 = self.particles[random.randrange(l)] 
            while(p1 == p2):
                p2 = self.particles[random.randrange(l)]

            y = (x + np.rint(.5 * np.abs(p1.centroids_pos - p2.centroids_pos))) % 256
            
            mutated_particle = Particle(n_clusters=self.n_clusters, data=self.data, centroids_pos = y)
            mutated_particle_list.append(mutated_particle)
        
        return mutated_particle_list

      

    def start(self, iteration, crossover_rate = 0.5):
        progress = []
        for j in range(iteration):
            print("iteration no =", j)
            
            mutaded_list = self.mutation()
            random.shuffle(self.particles)
            random.shuffle(mutaded_list)
            n = len(self.particles[0].centroids_pos)//2
            Population = []
            for i in range(self.n_particles//2):
                p1 = self.particles[i] if(crossover_rate >random.random()) else mutaded_list[i]
                p2 = self.particles[i+1] if(crossover_rate >random.random()) else mutaded_list[i+1]
                Population.append((p1, p1.pb_val))
                Population.append((p2, p2.pb_val))
                c = self.crossover(n, self.particles[i], self.particles[i+1])
                Population.append((c[0], c[0].pb_val))
                Population.append((c[1], c[1].pb_val))

            Population = sorted(Population, key=lambda x: x[1])
            
            z = [i[0] for i in Population]
            new_generation = z[: self.n_particles]
            self.particles = new_generation.copy()
            for i in (self.particles):
                print(i.centroids_pos)
        
            self.update_gb(new_generation[0])
            
            progress.append([self.gb_pos, self.gb_clustering, self.gb_val])
            print(self.gb_val)
      
        clusters = self.gb_clustering
        centers = self.gb_pos
        centers = np.uint8(centers)
        res = centers[clusters.flatten()]

        print('Completed!\n')
        return res         