import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


class DataGenerator:
    """Genera un dataset sintético coherente para pruebas de churn.

    La clase centraliza el control de aleatoriedad para garantizar
    reproducibilidad entre ejecuciones y crea automáticamente los
    directorios de salida para evitar errores de escritura.
    """

    def __init__(self, n_customers=1000, random_state: int | None = 42):
        self.n_customers = n_customers
        self.random_state = random_state
        self.customers_df = None
        self.events_df = None
        self.transactions_df = None
        self._set_random_seed()
        self._ensure_output_dirs()

    def _set_random_seed(self):
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)

    def _ensure_output_dirs(self):
        os.makedirs("data/synthetic", exist_ok=True)
        
    def generate_customer_profiles(self):
        print("Generando perfiles de clientes...")
        profiles = []
        today = datetime.now()
        locations = ['Bogota', 'Medellin', 'Cali', 'Barranquilla', 'Cartagena']
        devices = ['mobile', 'desktop', 'tablet']
        age_groups = ['18-25', '26-35', '36-45', '46+']

        for i in range(self.n_customers):
            signup_date = today - timedelta(days=random.randint(1, 365))
            profiles.append({
                'customer_id': f'CUST_{i:06d}',
                'signup_date': signup_date,
                'location': random.choice(locations),
                'device_type': random.choice(devices),
                'age_group': random.choice(age_groups)
            })

        self.customers_df = pd.DataFrame(profiles)
        return self.customers_df
    
    def generate_behavioral_data(self, days=180):
        print("Generando datos de comportamiento...")
        events = []
        event_types = ['page_view', 'add_to_cart', 'product_view', 'search', 'wishlist_add']

        for _, customer in self.customers_df.iterrows():
            n_sessions = max(1, int(np.random.poisson(8)))
            
            for session in range(n_sessions):
                session_id = f"SESS_{customer['customer_id']}_{session}"
                session_date = customer['signup_date'] + timedelta(
                    days=random.randint(0, (datetime.now() - customer['signup_date']).days)
                )
                
                n_events = random.randint(3, 20)
                for event_num in range(n_events):
                    event_time = session_date + timedelta(minutes=event_num * 2)
                    
                    events.append({
                        'customer_id': customer['customer_id'],
                        'session_id': session_id,
                        'timestamp': event_time,
                        'event_type': random.choice(event_types),
                        'page_url': random.choice(['/home', '/products', '/categories', '/deals']),
                        'duration_seconds': random.randint(10, 300)
                    })
        
        self.events_df = pd.DataFrame(events)
        return self.events_df
    
    def generate_transactions(self):
        print("Generando transacciones...")
        transactions = []

        buying_customers = random.sample(
            list(self.customers_df['customer_id']), int(self.n_customers * 0.6)
        )
        
        for customer_id in buying_customers:
            n_purchases = random.randint(1, 8)
            
            for purchase in range(n_purchases):
                purchase_date = datetime.now() - timedelta(days=random.randint(1, 180))
                
                transactions.append({
                    'transaction_id': f"TRX_{customer_id}_{purchase}",
                    'customer_id': customer_id,
                    'timestamp': purchase_date,
                    'amount': round(random.uniform(15000, 250000), -3),
                    'items_count': random.randint(1, 5),
                    'payment_method': random.choice(['credit_card', 'debit_card', 'pse', 'cash'])
                })
        
        self.transactions_df = pd.DataFrame(transactions)
        return self.transactions_df
    
    def generate_churn_labels(self, churn_threshold_days=60):
        print("Generando etiquetas de churn...")
        if self.events_df is None:
            raise ValueError("Primero genera los eventos")
            
        last_activity = self.events_df.groupby('customer_id')['timestamp'].max()
        analysis_date = self.events_df['timestamp'].max()
        
        churn_labels = []
        for customer_id, last_active in last_activity.items():
            days_inactive = (analysis_date - last_active).days
            is_churn = 1 if days_inactive > churn_threshold_days else 0
            
            churn_labels.append({
                'customer_id': customer_id,
                'churn': is_churn,
                'days_since_last_activity': days_inactive
            })
        
        return pd.DataFrame(churn_labels)
    
    def generate_all_data(self):
        print("Iniciando generacion de datos completos...")
        self.generate_customer_profiles()
        self.generate_behavioral_data()
        self.generate_transactions()
        
        churn_labels = self.generate_churn_labels()
        
        self.customers_df.to_csv('data/synthetic/customers.csv', index=False)
        self.events_df.to_csv('data/synthetic/events.csv', index=False)
        self.transactions_df.to_csv('data/synthetic/transactions.csv', index=False)
        churn_labels.to_csv('data/synthetic/churn_labels.csv', index=False)

        print(
            f"Datos generados: {len(self.customers_df)} clientes, {len(self.events_df)} eventos, {len(self.transactions_df)} transacciones"
        )
        print(f"Tasa de churn: {churn_labels['churn'].mean():.1%}")
        
        return self.customers_df, self.events_df, self.transactions_df, churn_labels

if __name__ == "__main__":
    generator = DataGenerator(n_customers=500)
    customers, events, transactions, churn = generator.generate_all_data()
    print("Muestra de datos:")
    print(customers.head())
