import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from utils.preprocessing import load_data, preprocess_data
from utils.models import CustomLogisticRegression, train_sklearn_model
from utils.visualization import (
    plot_distributions,
    plot_correlation_matrix,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance
)
from config import DATA_PATH, RANDOM_STATE, TEST_SIZE, CUSTOM_PARAMS, SKLEARN_PARAMS

def evaluate_model(y_true, y_pred, y_proba, model_name):
    """Calcula e retorna métricas de avaliação"""
    return {
        'Modelo': model_name,
        'Acurácia': accuracy_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_proba),
        'Precisão': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred)
    }

def main():
    # 1. Carregar e explorar dados
    print("=== CARREGANDO DADOS ===")
    data = load_data(DATA_PATH)
    print(f"\nShape do dataset: {data.shape}")
    print("\nPrimeiras linhas:")
    print(data.head())
    print("\nEstatísticas descritivas:")
    print(data.describe())

    # Visualizações exploratórias
    plot_distributions(data, filename='distributions.png')
    plot_correlation_matrix(data, filename='correlation_matrix.png')

    # 2. Pré-processamento
    print("\n=== PRÉ-PROCESSAMENTO ===")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        data, TEST_SIZE, RANDOM_STATE
    )
    feature_names = data.drop("Outcome", axis=1).columns.tolist()

    # 3. Treinamento dos modelos
    print("\n=== TREINANDO MODELOS ===")
    
    # Modelo Customizado
    print("\nTreinando modelo customizado...")
    custom_model = CustomLogisticRegression(**CUSTOM_PARAMS)
    custom_model.fit(X_train, y_train)
    custom_proba = custom_model.predict_proba(X_test)
    custom_pred = custom_model.predict(X_test)

    # Modelo Scikit-learn
    print("Treinando modelo Scikit-learn...")
    sklearn_model = train_sklearn_model(X_train, y_train, SKLEARN_PARAMS)
    sklearn_proba = sklearn_model.predict_proba(X_test)[:, 1]
    sklearn_pred = sklearn_model.predict(X_test)

    # 4. Avaliação dos modelos
    print("\n=== AVALIANDO MODELOS ===")
    
    # Calcular métricas
    results = [
        evaluate_model(y_test, custom_pred, custom_proba, "Customizado"),
        evaluate_model(y_test, sklearn_pred, sklearn_proba, "Scikit-learn")
    ]
    
    # Exibir resultados
    results_df = pd.DataFrame(results)
    print("\nMétricas de desempenho:")
    print(results_df.round(4))

    # 5. Visualizações dos resultados
    print("\n=== GERANDO VISUALIZAÇÕES ===")
    
    # Matrizes de confusão
    plot_confusion_matrix(y_test, custom_pred, 
                         labels=["Não Diabético", "Diabético"], 
                         filename='confusion_custom.png')
    
    plot_confusion_matrix(y_test, sklearn_pred, 
                         labels=["Não Diabético", "Diabético"], 
                         filename='confusion_sklearn.png')

    # Curvas ROC
    plot_roc_curve(y_test, custom_proba, filename='roc_custom.png')
    plot_roc_curve(y_test, sklearn_proba, filename='roc_sklearn.png')

    # Importância das features (apenas para modelos que suportam)
    if hasattr(custom_model, 'weights'):
        plot_feature_importance(
            np.abs(custom_model.weights), 
            feature_names,
            filename='feature_importance_custom.png'
        )
    
    if hasattr(sklearn_model, 'coef_'):
        plot_feature_importance(
            np.abs(sklearn_model.coef_[0]), 
            feature_names,
            filename='feature_importance_sklearn.png'
        )

    print("\n=== PROCESSO CONCLUÍDO ===")
    print("Visualizações salvas nos arquivos PNG")

if __name__ == "__main__":
    main()
    # Importações adicionais necessárias (não incluídas no início para clareza)
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    main()