import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, auc

def is_interactive():
    """Verifica se o ambiente suporta exibição interativa de gráficos"""
    return matplotlib.get_backend().lower() not in ['agg', 'pdf', 'ps', 'svg']

def save_or_show(fig, filename=None, format='png', dpi=300):
    """
    Exibe o gráfico na tela ou salva em arquivo, conforme o ambiente
    
    Args:
        fig: Figura matplotlib
        filename: Nome do arquivo para salvar (None para automático)
        format: Formato do arquivo (png, jpg, pdf, etc)
        dpi: Resolução da imagem
    """
    if is_interactive():
        plt.show()
    else:
        if filename is None:
            filename = f"plot_{np.random.randint(1000)}.{format}"
        fig.savefig(filename, format=format, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Gráfico salvo como: {filename}")

def set_style():
    """Configura o estilo visual dos gráficos"""
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

def plot_distributions(data, cols=None, filename=None):
    """
    Plota distribuições das variáveis numéricas
    
    Args:
        data: DataFrame com os dados
        cols: Lista de colunas para plotar (None para todas numéricas)
        filename: Nome do arquivo para salvar
    """
    set_style()
    numeric_cols = cols if cols else data.select_dtypes(include=['number']).columns
    
    if len(numeric_cols) == 0:
        print("Nenhuma coluna numérica encontrada para plotar")
        return
    
    fig, axes = plt.subplots(nrows=(len(numeric_cols)+1)//2, ncols=2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        sns.histplot(data[col], kde=True, ax=axes[i], bins=20)
        axes[i].set_title(f'Distribuição de {col}', fontsize=12)
    
    plt.suptitle("Distribuição das Variáveis Numéricas", y=1.02)
    plt.tight_layout()
    save_or_show(fig, filename)

def plot_correlation_matrix(data, cols=None, filename=None):
    """
    Plota matriz de correlação entre variáveis
    
    Args:
        data: DataFrame com os dados
        cols: Lista de colunas para incluir (None para todas numéricas)
        filename: Nome do arquivo para salvar
    """
    set_style()
    numeric_cols = cols if cols else data.select_dtypes(include=['number']).columns
    
    if len(numeric_cols) < 2:
        print("Número insuficiente de colunas numéricas para matriz de correlação")
        return
    
    corr = data[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                center=0, square=True, linewidths=.5, ax=ax)
    ax.set_title("Matriz de Correlação", fontsize=14)
    save_or_show(fig, filename)

def plot_confusion_matrix(y_true, y_pred, labels=None, filename=None):
    """
    Plota matriz de confusão
    
    Args:
        y_true: Valores reais
        y_pred: Valores previstos
        labels: Nomes das classes
        filename: Nome do arquivo para salvar
    """
    set_style()
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, 
                yticklabels=labels, ax=ax)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão")
    save_or_show(fig, filename)

def plot_roc_curve(y_true, y_proba, filename=None):
    """
    Plota curva ROC
    
    Args:
        y_true: Valores reais
        y_proba: Probabilidades previstas da classe positiva
        filename: Nome do arquivo para salvar
    """
    set_style()
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taxa de Falsos Positivos')
    ax.set_ylabel('Taxa de Verdadeiros Positivos')
    ax.set_title('Curva ROC')
    ax.legend(loc="lower right")
    save_or_show(fig, filename)

def plot_feature_importance(importance, feature_names, filename=None):
    """
    Plota importância das features
    
    Args:
        importance: Valores de importância das features
        feature_names: Nomes das features
        filename: Nome do arquivo para salvar
    """
    set_style()
    indices = np.argsort(importance)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(importance)), importance[indices], align='center')
    ax.set_yticks(range(len(importance)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importância')
    ax.set_title('Importância das Features')
    save_or_show(fig, filename)