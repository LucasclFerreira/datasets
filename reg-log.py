import numpy as np

class RegressaoLogistica:
    def __init__(self, taxa=0.001, iteracoes=1000):
        self.taxa = taxa
        self.iteracoes = iteracoes
        self.pesos = None
        self.vies = None
    
    def ajustar(self, X, y):
        num_instancias, num_features = X.shape
        self.pesos = np.zeros(num_features)
        self.vies = 0

        for _ in range(self.iteracoes):
            modelo_linear = np.dot(X, self.pesos) + self.vies
            y_prev = self._sigmoid(modelo_linear)
            derivada_pesos = (1 / num_instancias) * np.dot(X.T, y_prev - y)
            derivada_vies = (1 / num_instancias) * np.sum(y_prev - y)
            self.pesos -= self.taxa * derivada_pesos
            self.vies -= self.taxa * derivada_vies


    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))