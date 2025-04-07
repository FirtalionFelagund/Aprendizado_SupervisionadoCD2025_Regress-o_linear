import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Carrega e retorna os dados."""
    col_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                 "Insulin", "BMI", "DiabetesPedigree", "Age", "Outcome"]
    return pd.read_csv(filepath, header=None ,names=col_names)

def preprocess_data(data, test_size, random_state):
    """Divide e normaliza os dados."""
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler