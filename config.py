# Configurações de caminho e parâmetros
DATA_PATH = "data/pima_indians_diabetes_processed.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Parâmetros do modelo customizado
CUSTOM_PARAMS = {
    'learning_rate': 0.01,
    'n_iter': 1000,
    'lambda_param': 0.1,
    'regularization': 'l2',
    'tol': 1e-4
}

# Parâmetros do modelo Scikit-learn
SKLEARN_PARAMS = {
    'penalty': 'l2',
    'C': 1.0,
    'solver': 'lbfgs',
    'max_iter': 1000
}