import numpy as np
from collections import Counter

def distiancia_euclidiana(x1, x2):
    return np.sqrt(np.sum(x1 - x2) ** 2)

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def ajustar(self, X, y):
        self.X_train = X
        self.y_train = y

    def prever(self, X):
        rotulos = [self._prever(x) for x in X]
        return np.array(rotulos)
    
    def _prever(self, x):
        # calcular as distancias
        distancias = [distiancia_euclidiana(x, x_train) for x_train in self.X_train]

        # pegar k amostras mais proximas
        indices_k = np.argsort(distancias)[:self.k]
        rotulos_proximos_k = [self.y_train[i] for i in indices_k]

        # classes mais proximas
        mais_comuns = Counter(rotulos_proximos_k).most_common(1)
        return mais_comuns[0][0]