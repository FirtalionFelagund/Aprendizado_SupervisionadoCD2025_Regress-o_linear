import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

class CustomLogisticRegression:
    def __init__(self, learning_rate=0.01, n_iter=1000, lambda_param=0.1, regularization='l2', tol=1e-4):
        """
        Inicializa o modelo de regressão logística customizada
        
        Parâmetros:
        -----------
        learning_rate: float (default=0.01)
            Taxa de aprendizado para o gradiente descendente
            
        n_iter: int (default=1000)
            Número máximo de iterações
            
        lambda_param: float (default=0.1)
            Parâmetro de regularização
            
        regularization: str (default='l2')
            Tipo de regularização ('l1' ou 'l2')
            
        tol: float (default=1e-4)
            Tolerância para critério de parada
        """
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.lambda_param = lambda_param
        self.regularization = regularization
        self.tol = tol
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):
        """Função sigmoide"""
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """Treina o modelo com gradiente descendente"""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iter):
            # Cálculo das predições
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            
            # Cálculo dos gradientes
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Adicionar regularização
            if self.regularization == 'l2':
                dw += (self.lambda_param/n_samples) * self.weights
            elif self.regularization == 'l1':
                dw += (self.lambda_param/n_samples) * np.sign(self.weights)
            
            # Atualização dos parâmetros
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Armazenar histórico de loss
            loss = self._compute_loss(X, y)
            self.loss_history.append(loss)
            
            # Critério de parada
            if i > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                break

    def _compute_loss(self, X, y):
        """Calcula a função de custo com regularização"""
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        
        # Log loss
        loss = -np.mean(y * np.log(y_pred + 1e-15) + (1-y) * np.log(1-y_pred + 1e-15))
        
        # Regularização
        if self.regularization == 'l2':
            loss += (self.lambda_param/(2*len(y))) * np.sum(self.weights**2)
        elif self.regularization == 'l1':
            loss += (self.lambda_param/len(y)) * np.sum(np.abs(self.weights))
            
        return loss

    def predict_proba(self, X):
        """Retorna probabilidades previstas"""
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """Retorna classes previstas"""
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        """Calcula a acurácia"""
        return accuracy_score(y, self.predict(X))
    
    def train_sklearn_model(X_train, y_train, params):
        """Treina um modelo LogisticRegression do scikit-learn"""
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        return model
    
def train_sklearn_model(X_train, y_train, params):
    """
    Treina um modelo LogisticRegression do scikit-learn
        
    Args:
        X_train: Dados de treino
        y_train: Labels de treino
        params: Dicionário de parâmetros para o modelo
            
    Returns:
        Modelo treinado
    """
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return model