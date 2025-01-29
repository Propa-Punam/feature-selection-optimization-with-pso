import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from typing import Tuple, List

class CompetitiveCoevolutionPSO:
    def __init__(self, n_particles: int, dim: int):
        self.n_particles = self._next_power_of_2(n_particles)
        self.dim = dim
        self.population = np.random.choice([0, 1], size=(self.n_particles, dim), p=[0.7, 0.3])
        self.velocities = np.random.uniform(-1, 1, (self.n_particles, dim))
        self.fitness_scores = np.zeros(self.n_particles)
        self.tournament_history: List[List[Tuple[int, int, int]]] = []  # [(winner, loser, round)]
        self.best_solution = None
        self.best_fitness = -np.inf

    def _next_power_of_2(self, x: int) -> int:
        return 1 if x == 0 else 2**(x - 1).bit_length()

    def optimize(self, X_train: np.ndarray, X_test: np.ndarray,
                y_train: np.ndarray, y_test: np.ndarray,
                max_generations: int = 40) -> Tuple[np.ndarray, float]:
        for generation in range(max_generations):
            # Competitive fitness assessment
            self.single_elimination_tournament(X_train, X_test, y_train, y_test)

            # Breeding and population update
            offspring = self.breed()

            # Generational replacement with elitism
            elite_size = self.n_particles // 4
            elite_indices = np.argsort(self.fitness_scores)[-elite_size:]

            new_population = np.zeros_like(self.population)
            new_population[:elite_size] = self.population[elite_indices]
            new_population[elite_size:] = offspring[:-elite_size]

            self.population = new_population

            print(f"Generation {generation}: Best fitness = {self.best_fitness:.4f}")

        return self.best_solution, self.best_fitness
