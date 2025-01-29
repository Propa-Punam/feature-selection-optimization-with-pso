import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def evaluate_solution(self, features: np.ndarray, X_train: np.ndarray,
                     X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> float:
    selected_features = features.astype(bool)
    n_selected = np.sum(selected_features)

    if n_selected < 2 or n_selected > X_train.shape[1] // 2:
        return 0.0

    try:
        selected_train = X_train[:, selected_features]
        selected_test = X_test[:, selected_features]

        model = LinearSVC(dual=False, max_iter=2000, random_state=42)
        model.fit(selected_train, y_train)
        predictions = model.predict(selected_test)
        accuracy = accuracy_score(y_test, predictions)

        # Balanced fitness function
        feature_ratio = n_selected / X_train.shape[1]
        complexity_penalty = 0.001 * feature_ratio
        diversity_bonus = 0.001 * (1 - self._solution_similarity_to_best(features))

        return accuracy - complexity_penalty + diversity_bonus

    except Exception as e:
        print(f"Error in evaluation: {e}")
        return 0.0

def _solution_similarity_to_best(self, solution: np.ndarray) -> float:
    if self.best_solution is None:
        return 0.0
    return np.mean(solution == self.best_solution)
