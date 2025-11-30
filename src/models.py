import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("ENTRENANDO MODELOS PREDICTIVOS")

class ChurnPredictor:
    def __init__(self):
        self.models = {}
        self.feature_names = []
        
    def prepare_data(self, features_path, labels_path):
        print("Preparando datos para modelado...")
        features_df = pd.read_csv(features_path)
        labels_df = pd.read_csv(labels_path)
        
        data = features_df.merge(labels_df[['customer_id', 'churn']], on='customer_id')
        
        # Eliminar customer_id y asegurar que todas las columnas sean numéricas
        X = data.drop(['customer_id', 'churn'], axis=1)
        
        # Convertir todas las columnas a numérico, forzando errores a NaN
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Eliminar filas con NaN resultantes de la conversión
        X = X.dropna()
        
        # Reindexar y para que coincida con X
        y = data.loc[X.index, 'churn']
        
        self.feature_names = X.columns.tolist()
        
        print(f"Características después de limpieza: {self.feature_names}")
        print(f"Tipos de datos: {X.dtypes.unique()}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test, data
    
    def train_models(self, X_train, X_test, y_train, y_test):
        print("Entrenando modelos...")
        
        # Regresión Logística
        print(" - Entrenando Regresión Logística...")
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train, y_train)
        self.models['logistic_regression'] = lr
        
        # Random Forest
        print(" - Entrenando Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf
        
        # XGBoost - asegurar que los datos son float32
        print(" - Entrenando XGBoost...")
        X_train_xgb = X_train.astype(np.float32)
        X_test_xgb = X_test.astype(np.float32)
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train_xgb, y_train)
        self.models['xgboost'] = xgb_model
        
        # Evaluar con datos convertidos
        self.evaluate_models(X_test_xgb, y_test)
        
        return self.models
    
    def evaluate_models(self, X_test, y_test):
        print("\nEvaluación de modelos:")
        print("=" * 50)
        
        results = []
        for name, model in self.models.items():
            # Para XGBoost usar datos convertidos, para otros usar normales
            if name == 'xgboost':
                X_test_eval = X_test.astype(np.float32)
            else:
                X_test_eval = X_test
                
            y_pred = model.predict(X_test_eval)
            y_pred_proba = model.predict_proba(X_test_eval)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            accuracy = accuracy_score(y_test, y_pred)
            
            results.append({'Modelo': name, 'AUC': auc, 'Accuracy': accuracy})
            print(f"{name:20} AUC: {auc:.3f} | Accuracy: {accuracy:.3f}")
            
            if name == 'xgboost':
                print(f"\nReporte de clasificación - {name}:")
                print(classification_report(y_test, y_pred))
        
        results_df = pd.DataFrame(results)
        best_model = results_df.loc[results_df['AUC'].idxmax()]
        print(f"\nMejor modelo: {best_model['Modelo']} (AUC: {best_model['AUC']:.3f})")
        
        return results_df
    
    def predict_churn(self, customer_data):
        if not self.models:
            raise ValueError("Primero entrena los modelos")
        
        model = self.models['xgboost']
        
        # Convertir a DataFrame y asegurar tipos numéricos
        if hasattr(customer_data, 'values'):
            # Es una Series de pandas
            customer_df = customer_data.to_frame().T
        else:
            # Es un array numpy
            customer_df = pd.DataFrame([customer_data], columns=self.feature_names)
        
        # Convertir todas las columnas a float32
        for col in customer_df.columns:
            customer_df[col] = pd.to_numeric(customer_df[col], errors='coerce')
        
        customer_df = customer_df.astype(np.float32)
        
        prediction = model.predict(customer_df)[0]
        probability = model.predict_proba(customer_df)[:, 1][0]
        
        return prediction, probability

def run_model_pipeline():
    print("Iniciando pipeline de modelado...")
    predictor = ChurnPredictor()
    
    X_train, X_test, y_train, y_test, full_data = predictor.prepare_data(
        'data/processed/features.csv',
        'data/synthetic/churn_labels.csv'
    )
    
    print(f"Datos para entrenamiento: {X_train.shape[0]} clientes")
    print(f"Datos para prueba: {X_test.shape[0]} clientes")
    print(f"Características: {X_train.shape[1]}")
    print(f"Tasa de churn: {y_train.mean():.1%}")
    
    models = predictor.train_models(X_train, X_test, y_train, y_test)
    
    print("Pipeline de modelado completado")
    return predictor, models

if __name__ == "__main__":
    run_model_pipeline()