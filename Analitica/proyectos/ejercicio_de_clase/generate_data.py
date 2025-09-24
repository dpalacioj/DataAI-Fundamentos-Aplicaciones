"""
Este módulo permite generar datos sintéticos de usuarios
para orientar promociones.
"""

import json
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

class UserGenerator:
    def __init__(self, n_samples=1000, seed=42):
        self.n_samples = n_samples
        self.seed = seed

    def generate_synthetic_data(self):
        """
        Genera datos sintéticos de usuarios para promociones

        Args:
            n_samples (int): Número de muestras a generar.
            seed (int): Semilla para reproducibilidad.

        Returns:
            pd.DataFrame: DataFrame con los datos generados.
        """
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Definir categorías y valores posibles
        age_groups = ['18-25', '26-35', '36-45', '46-55', '56+']
        locations = ['Buenos Aires', 'Córdoba', 'Rosario', 'Mendoza', 'La Plata', 'Otros']
        device_types = ['Móvil', 'Desktop', 'Tablet']
        suscription_types = ['Free', 'Basic', 'Premium', 'Enterprise']

        # Generar fechas
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        # Generar datos sintéticos
        data = []

        for i in range(self.n_samples):
            # Fecha de registro
            registration_date = start_date + timedelta(
                days=random.randint(0, (end_date - start_date).days)
            )

            # Días desde el registro
            days_since_registration = (end_date - registration_date).days

            # Perfil del usuario
            age_group = random.choices(age_groups, weights=[0.2, 0.3, 0.25, 0.15, 0.1])[0]
            location = random.choices(locations, weights=[0.35, 0.2, 0.15, 0.1, 0.1, 0.1])[0]
            device_type = random.choices(device_types, weights=[0.6, 0.3, 0.1])[0]
            suscription_type = random.choices(suscription_types, weights=[0.4, 0.3, 0.25, 0.05])[0]

            # Comportamiento transaccional
            total_purchases = random.randint(0,50)
            avg_order_value = random.uniform(10, 500)
            last_purchase_days = random.randint(0,180) if total_purchases > 0 else 999

            # Métricas engagement
            session_last_30_days = random.randint(0,30)
            time_on_site_minutes = random.uniform(1,120)
            pages_per_session = random.uniform(1,20)

            # Métricas de conversión
            cart_abandonment_rate = random.uniform(0,1)
            purchase_frequency = total_purchases / max(days_since_registration / 30, 1)

            # Crear registros
            user_record = {
                'user_id': f'user_{i+1:06d}',
                'age_group': age_group,
                'location': location,
                'device_type': device_type,
                'suscription_type': suscription_type,
                'days_since_registration': days_since_registration,
                'total_purchases': total_purchases,
                'avg_order_value': round(avg_order_value, 2),
                'last_purchase_days': last_purchase_days,
                'session_last_30_days': session_last_30_days,
                'time_on_site_minutes': round(time_on_site_minutes, 1),
                'pages_per_session': round(pages_per_session, 1),
                'cart_abandonment_rate': round(cart_abandonment_rate, 3),
                'purchase_frequency': round(purchase_frequency, 2)
            }

            data.append(user_record)

        return pd.DataFrame(data)
    
    def add_missing_data(self, df):
        """
        Introduce valores faltantes aleatoriamente en el DataFrame.

        Args:
            df (pd.DataFrame): DataFrame al que se le agregarán valores faltantes.

        Returns:
            pd.DataFrame: DataFrame con valores faltantes introducidos.
        """

        null_config = {
                'age_group': 0.05,
                'location': 0.03,
                'device_type': 0.02,
                'suscription_type': 0.01,
                'last_purchase_days': 0.01,
                'time_on_site_minutes': 0.1,
                'cart_abandonment_rate': 0.2,
                'purchase_frequency': 0.2
        }

        df_with_nulls = df.copy()

        for column, null_prob in null_config.items():
            if column in df_with_nulls.columns:
                # Generar máscara de valores nulos
                null_mask = np.random.random(len(df_with_nulls)) < null_prob

                # Aplicar valores nulos
                df_with_nulls.loc[null_mask, column] = np.nan

                print(f"¡Agregados con éxito!")

        return df_with_nulls
        
    def create_dataset(self):
        """Función para crear y guardar el dataset sintético."""

        df = self.generate_synthetic_data()

        df = self.add_missing_data(df)

        # Crear variable objetivo
        df['dar_promocion'] = random.choices([0, 1], k=len(df))

        # Guardar los datos
        output_file = 'usuarios_promociones.csv'

        df.to_csv(output_file, index=False, encoding='utf-8')

        return df
    