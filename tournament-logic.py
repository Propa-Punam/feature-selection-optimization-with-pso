import numpy as np
from typing import List, Tuple

def single_elimination_tournament(self, X_train: np.ndarray, X_test: np.ndarray,
                               y_train: np.ndarray, y_test: np.ndarray) -> None:
    # Reset tournament history for this generation
    self.tournament_history = []
    round_matches: List[Tuple[int, int, int]] = []

    # Initialize tournament brackets
    indices = np.arange(self.n_particles)
    np.random.shuffle(indices)
    current_round = 0

    while len(indices) > 1:
        next_round_indices = []
        round_matches = []

        for i in range(0, len(indices), 2):
            p1_idx, p2_idx = indices[i], indices[i + 1]

            # Evaluate both solutions
            p1_score = self.evaluate_solution(self.population[p1_idx],
                                           X_train, X_test, y_train, y_test)
            p2_score = self.evaluate_solution(self.population[p2_idx],
                                           X_train, X_test, y_train, y_test)

            # Determine winner and record match result
            if p1_score > p2_score:
                next_round_indices.append(p1_idx)
                round_matches.append((p1_idx, p2_idx, current_round))
            else:
                next_round_indices.append(p2_idx)
                round_matches.append((p2_idx, p1_idx, current_round))

        self.tournament_history.append(round_matches)
        indices = next_round_indices
        current_round += 1

    # Calculate relative fitness based on tournament performance
    self._calculate_tournament_fitness()

    # Update best solution
    winner_idx = indices[0]
    winner_score = self.evaluate_solution(self.population[winner_idx],
                                       X_train, X_test, y_train, y_test)

    if winner_score > self.best_fitness:
        self.best_solution = self.population[winner_idx].copy()
        self.best_fitness = winner_score

def _calculate_tournament_fitness(self) -> None:
    # Initialize base fitness values
    self.fitness_scores = np.zeros(self.n_particles)

    # Award points based on tournament performance
    for round_num, round_matches in enumerate(self.tournament_history):
        round_weight = 2 ** round_num  # Higher rounds worth more
        for winner, loser, _ in round_matches:
            self.fitness_scores[winner] += round_weight
            self.fitness_scores[loser] += round_weight * 0.5  # Consolation points

    # Normalize fitness scores
    if np.max(self.fitness_scores) > 0:
        self.fitness_scores = self.fitness_scores / np.max(self.fitness_scores)
