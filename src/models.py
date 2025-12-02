import os
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


class ChurnPredictor:
    """Administra el ciclo completo de entrenamiento y evaluación de modelos."""

    def __init__(self):
        self.models = {}
        self.feature_names = []
        self.best_model_name: str | None = None
        
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
        """Entrena los modelos base y calcula sus métricas."""
        print("Entrenando modelos...")

        print(" - Entrenando Regresión Logística...")
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train, y_train)
        self.models['logistic_regression'] = lr

        print(" - Entrenando Random Forest...")
        rf = RandomForestClassifier(n_estimators=120, random_state=42)
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf

        print(" - Entrenando XGBoost...")
        X_train_xgb = X_train.astype(np.float32)
        X_test_xgb = X_test.astype(np.float32)

        xgb_model = xgb.XGBClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train_xgb, y_train)
        self.models['xgboost'] = xgb_model

        evaluation = self.evaluate_models(X_test_xgb, y_test)
        self.best_model_name = evaluation.loc[evaluation['AUC'].idxmax(), 'Modelo']
        self.evaluation_results = evaluation

        return evaluation

    def evaluate_models(self, X_test, y_test):
        """Calcula métricas, imprime un resumen y devuelve un DataFrame."""
        print("\nEvaluación de modelos:")
        print("=" * 50)

        results = []
        for name, model in self.models.items():
            X_test_eval = X_test.astype(np.float32) if name == 'xgboost' else X_test

            y_pred = model.predict(X_test_eval)
            y_pred_proba = model.predict_proba(X_test_eval)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            accuracy = accuracy_score(y_test, y_pred)

            results.append({'Modelo': name, 'AUC': auc, 'Accuracy': accuracy})
            print(f"{name:20} AUC: {auc:.3f} | Accuracy: {accuracy:.3f}")

            if name == 'xgboost':
                print(f"\nReporte de clasificación - {name}:")
                print(classification_report(y_test, y_pred))

        results_df = pd.DataFrame(results).sort_values(by='AUC', ascending=False)
        best_model = results_df.iloc[0]
        print(f"\nMejor modelo: {best_model['Modelo']} (AUC: {best_model['AUC']:.3f})")

        return results_df

    def save_evaluation_report(self, results_df, output_path="data/processed/model_report.csv"):
        """Guarda el resumen de evaluación para trazabilidad."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"Reporte de evaluación guardado en {output_path}")
    
    def predict_churn(self, customer_data):
        if not self.models:
            raise ValueError("Primero entrena los modelos")

        model_name = self.best_model_name or 'xgboost'
        model = self.models.get(model_name)

        if model is None:
            raise ValueError("No se encontró un modelo entrenado para predecir")
        
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
    
    evaluation = predictor.train_models(X_train, X_test, y_train, y_test)
    predictor.save_evaluation_report(evaluation)

    print("Pipeline de modelado completado")
    return predictor, predictor.models

if __name__ == "__main__":
    run_model_pipeline()
