import pandas as pd
import numpy as np
from datetime import datetime

class FeatureEngineer:
    def __init__(self, customers_df, events_df, transactions_df):
        self.customers_df = customers_df
        self.events_df = events_df
        self.transactions_df = transactions_df
        
    def calculate_customer_features(self):
        print("Calculando caracteristicas por cliente...")
        features_list = []
        
        for customer_id in self.customers_df['customer_id']:
            customer_events = self.events_df[self.events_df['customer_id'] == customer_id]
            customer_transactions = self.transactions_df[
                self.transactions_df['customer_id'] == customer_id
            ]
            
            customer_feats = {
                'customer_id': customer_id,
                
                'total_sessions': customer_events['session_id'].nunique(),
                'total_events': len(customer_events),
                'avg_session_duration': self._avg_session_duration(customer_events),
                'pages_per_session': self._pages_per_session(customer_events),
                
                'cart_additions': len(customer_events[customer_events['event_type'] == 'add_to_cart']),
                'product_views': len(customer_events[customer_events['event_type'] == 'product_view']),
                'searches': len(customer_events[customer_events['event_type'] == 'search']),
                
                'total_purchases': len(customer_transactions),
                'total_spent': customer_transactions['amount'].sum() if len(customer_transactions) > 0 else 0,
                'avg_purchase_value': customer_transactions['amount'].mean() if len(customer_transactions) > 0 else 0,
                'avg_items_per_purchase': customer_transactions['items_count'].mean() if len(customer_transactions) > 0 else 0,
                
                'days_since_signup': self._days_since_signup(customer_id),
                'days_since_last_activity': self._days_since_last_activity(customer_events),
                'purchase_frequency': self._purchase_frequency(customer_transactions),
            }
            
            customer_feats['cart_addition_rate'] = customer_feats['cart_additions'] / max(1, customer_feats['total_events'])
            customer_feats['purchase_conversion_rate'] = customer_feats['total_purchases'] / max(1, customer_feats['total_sessions'])
            customer_feats['avg_days_between_purchases'] = self._avg_days_between_purchases(customer_transactions)
            
            features_list.append(customer_feats)
        
        return pd.DataFrame(features_list)
    
    def _avg_session_duration(self, customer_events):
        if customer_events.empty:
            return 0
            
        session_durations = []
        for session_id in customer_events['session_id'].unique():
            session_events = customer_events[customer_events['session_id'] == session_id]
            if len(session_events) > 1:
                duration = (session_events['timestamp'].max() - session_events['timestamp'].min()).total_seconds()
                session_durations.append(duration)
        
        return np.mean(session_durations) if session_durations else 0
    
    def _pages_per_session(self, customer_events):
        if customer_events.empty:
            return 0
            
        pages_per_session = customer_events.groupby('session_id').size()
        return pages_per_session.mean()
    
    def _days_since_signup(self, customer_id):
        customer = self.customers_df[self.customers_df['customer_id'] == customer_id].iloc[0]
        # Convertir a datetime si es string
        signup_date = customer['signup_date']
        if isinstance(signup_date, str):
            signup_date = pd.to_datetime(signup_date)
        return (datetime.now() - signup_date).days
    
    def _days_since_last_activity(self, customer_events):
        if customer_events.empty:
            return 999
        
        last_activity = customer_events['timestamp'].max()
        # Convertir a datetime si es string
        if isinstance(last_activity, str):
            last_activity = pd.to_datetime(last_activity)
        return (datetime.now() - last_activity).days
    
    def _purchase_frequency(self, customer_transactions):
        if len(customer_transactions) < 2:
            return 0
            
        # Convertir timestamps a datetime si son strings
        timestamps = customer_transactions['timestamp']
        if isinstance(timestamps.iloc[0], str):
            timestamps = pd.to_datetime(timestamps)
            
        days_span = (timestamps.max() - timestamps.min()).days
        if days_span == 0:
            return len(customer_transactions)
        
        return len(customer_transactions) / (days_span / 30.0)
    
    def _avg_days_between_purchases(self, customer_transactions):
        if len(customer_transactions) < 2:
            return 999
            
        # Convertir timestamps a datetime si son strings
        sorted_dates = customer_transactions['timestamp'].sort_values()
        if isinstance(sorted_dates.iloc[0], str):
            sorted_dates = pd.to_datetime(sorted_dates)
            
        days_between = [(sorted_dates.iloc[i] - sorted_dates.iloc[i-1]).days 
                       for i in range(1, len(sorted_dates))]
        
        return np.mean(days_between) if days_between else 999

def create_features(customers_path, events_path, transactions_path, output_path):
    print("Cargando datos para feature engineering...")
    
    # Cargar datos parseando las fechas correctamente
    customers = pd.read_csv(customers_path, parse_dates=['signup_date'])
    events = pd.read_csv(events_path, parse_dates=['timestamp'])
    transactions = pd.read_csv(transactions_path, parse_dates=['timestamp'])
    
    print("Calculando caracteristicas...")
    engineer = FeatureEngineer(customers, events, transactions)
    features_df = engineer.calculate_customer_features()
    
    print("Guardando caracteristicas...")
    features_df.to_csv(output_path, index=False)
    
    print(f"Caracteristicas creadas: {features_df.shape[1]} caracteristicas para {features_df.shape[0]} clientes")
    return features_df

if __name__ == "__main__":
    create_features(
        'data/synthetic/customers.csv',
        'data/synthetic/events.csv', 
        'data/synthetic/transactions.csv',
        'data/processed/features.csv'
    )