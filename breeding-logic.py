import numpy as np

def breed(self) -> np.ndarray:
    # Tournament selection for breeding
    offspring = np.zeros_like(self.population)

    for i in range(len(offspring)):
        # Select parents using tournament selection
        parent1_idx = self._tournament_selection(3)  # Tournament size of 3
        parent2_idx = self._tournament_selection(3)

        # Uniform crossover
        mask = np.random.rand(self.dim) < 0.5
        offspring[i] = np.where(mask, self.population[parent1_idx],
                              self.population[parent2_idx])

        # Mutation
        mutation_rate = 0.1 * (1 - self.fitness_scores[i])  # Adaptive mutation
        mutation_mask = np.random.rand(self.dim) < mutation_rate
        offspring[i] = np.where(mutation_mask, 1 - offspring[i], offspring[i])

        # Ensure minimum features
        if np.sum(offspring[i]) < 2:
            random_features = np.random.choice(self.dim, 2, replace=False)
            offspring[i][random_features] = 1

    return offspring

def _tournament_selection(self, tournament_size: int) -> int:
    candidates = np.random.choice(self.n_particles, tournament_size, replace=False)
    return candidates[np.argmax(self.fitness_scores[candidates])]
