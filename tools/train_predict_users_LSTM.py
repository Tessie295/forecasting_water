import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import glob
import warnings
import logging
import argparse

# Desactiva los warnings y ajusta el logging
warnings.filterwarnings("ignore")
logging.getLogger("NP").setLevel(logging.ERROR)

def cargar_datos(path_data):
    # Get a list of all CSV files in the directory
    csv_files = glob.glob(path_data) #'predictions_dia_avg/predictions_0.csv/*.csv'

    # Initialize an empty list to store individual DataFrames
    dfs = []

    # Loop through each CSV file and read it into a DataFrame
    for file in csv_files:
        df_temp = pd.read_csv(file)
        dfs.append(df_temp)

    # Concatenate all DataFrames into a single DataFrame
    df = pd.concat(dfs, ignore_index=True)

    # Optional: Reset the index of the final DataFrame
    df.reset_index(drop=True, inplace=True)

    df.head()

    df = df.rename(columns={'day': 'ds', 'avg_all_users': 'y'})

    # Asumiendo que tu DataFrame se llama 'df'

    # Primero, asegúrate de que 'ds' es de tipo datetime
    df['ds'] = pd.to_datetime(df['ds'])

    # Luego, elimina la información de zona horaria
    df['ds'] = df['ds'].dt.tz_localize(None)

    # Verifica el tipo de datos de la columna 'ds'
    print(df['ds'].dtype)

    df = df.sort_values('ds')

    df.head()

    return df

    # Función para crear ventanas deslizantes (sliding windows)
def create_sequences(data, input_steps, output_steps):
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps + 1):
        X.append(data[i:i + input_steps])
        y.append(data[i + input_steps:i + input_steps + output_steps])
    return np.array(X), np.array(y)

def pre_process_data_lstm_train(df, input_steps, output_steps):

    # Creamos ventanas del acumulado
    df['acumulado_movil_30'] = df['y'].rolling(window=30).sum()
    df = df.drop(columns=['y'])
    df = df.rename(columns={'acumulado_movil_30': 'y'})

    # # Quitamos datos de 2018-2019 ya que meten ruido
    # df = df[df["ds"] >= "2020-02-01 00:00:00"] 

    df = df[df["ds"] >= "2021-01-31 00:00:00"] # Quitamos datos que tienen la media acumulada Nula
    df_data_train = df[df["ds"] < "2023-12-01 00:00:00"] 
    df_data_test = df[df["ds"] >= "2023-12-01 00:00:00"] 

    data = df_data_train

    # Escalado de datos
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[['y']])
    data['y_scaled'] = data_scaled

    # Creación de secuencias
    X, y = create_sequences(data_scaled, input_steps, output_steps)

    # División en conjunto de entrenamiento y prueba
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return  X_train, X_test, y_train, y_test, X, y, scaler

def train_model(X, y, input_steps, output_steps):

    # Entrenamiento final con todos los datos de entrenamiento
    model_final = Sequential([
        LSTM(64, activation='tanh', input_shape=(input_steps, 1), return_sequences=False),
        Dense(output_steps)
    ])
    model_final.compile(optimizer='adam', loss='mse')

    # Entrenar el modelo final con los datos de entrenamiento completos
    model_final.fit(
        X, y.squeeze(),
        validation_split=0.2,  # Validación durante el entrenamiento para verificar sobreajuste
        epochs=50,             # Incrementamos los epochs para capturar patrones más complejos
        batch_size=32,
        verbose=1
)
    return model_final

def get_preds_lstm(X_test, y_test, model, scaler):
    # Predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)

    # Inversión del escalado para las métricas
    y_test_inverse = scaler.inverse_transform(y_test.squeeze())
    y_pred_inverse = scaler.inverse_transform(y_pred)

    # Cálculo de métricas de error
    mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
    rmse = np.sqrt(mean_squared_error(y_test_inverse, y_pred_inverse))
    mape = np.mean(np.abs((y_test_inverse - y_pred_inverse) / y_test_inverse)) * 100

    # Resultados
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%") #3.67%


def cargar_data_user(data_path):
    # Get a list of all CSV files in the directory
    csv_files = glob.glob(data_path) #'data_testing/cluster_0_sum.csv/*.csv'

    # Initialize an empty list to store individual DataFrames
    dfs = []

    # Loop through each CSV file and read it into a DataFrame
    for file in csv_files:
        df_temp = pd.read_csv(file)
        dfs.append(df_temp)

    # Concatenate all DataFrames into a single DataFrame
    df = pd.concat(dfs, ignore_index=True)

    # Optional: Reset the index of the final DataFrame
    df.reset_index(drop=True, inplace=True)

    df = df.rename(columns={'day': 'ds', 'sum_value': 'y', 'serial_number': 'user_id'})

    # Asumiendo que tu DataFrame se llama 'df'

    # Primero, asegúrate de que 'ds' es de tipo datetime
    df['ds'] = pd.to_datetime(df['ds'])

    # Luego, elimina la información de zona horaria
    df['ds'] = df['ds'].dt.tz_localize(None)

    # Verifica el tipo de datos de la columna 'ds'
    print(df['ds'].dtype)

    df_users = df.sort_values('ds')

    print(df_users.head())

    return df_users

def create_sequences_users(data, input_steps, output_steps):
    X, y = [], []
    for i in range(len(data) - input_steps):
        # Obtención de la secuencia de entrada
        X_seq = data[i:i + input_steps]
        
        # Obtención de la secuencia de salida, con relleno de ceros si es necesario
        y_seq = data[i + input_steps:i + input_steps + output_steps]
        
        # Rellenar con ceros si no hay suficientes datos en la secuencia de salida
        if len(y_seq) < output_steps:
            y_seq = np.pad(y_seq, (0, output_steps - len(y_seq)), mode='constant', constant_values=0)
        
        X.append(X_seq)
        y.append(y_seq)
    
    return np.array(X), np.array(y)

def preds_users(X_test, y_test, y_pred, scaler, user_id, cluster):

    # Desescalar las predicciones y los valores reales
    y_test_unscaled = scaler.inverse_transform(y_test)  # Desescalar los valores reales
    y_pred_unscaled = scaler.inverse_transform(y_pred)  # Desescalar las predicciones

    # Calcular métricas de evaluación
    mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_unscaled))
    mape = mean_absolute_percentage_error(y_test_unscaled, y_pred_unscaled) * 100

    # Imprimir métricas
    print(f"Conjunto de Prueba - MAE: {mae:.4f}")
    print(f"Conjunto de Prueba - RMSE: {rmse:.4f}")
    print(f"Conjunto de Prueba - MAPE: {mape:.2f}%")

    metrics_df = pd.DataFrame({"Cluster": [cluster],
                    "user_id": [user_id],
                    "MAE": [mae], 
                    "RMSE": [rmse],
                    "MAPE": [mape]})

    # Guardar resultados en CSV
    # metrics_df.to_csv('resultados_experimento_usuarios/metrics_LSTM_'+str(cluster)+'.csv', index=False)
    # print("Metricas modelo guardadas")

    return metrics_df


# Función principal
def main():
    parser = argparse.ArgumentParser(description="Script para procesar y filtrar predicciones de consumo de agua")
    parser.add_argument('--data_preds', required=True, help="Ruta del archivo de predicciones en formato parquet")
    parser.add_argument('--cluster', required=True, help="Cluster del que se quiere extraer datos")
    parser.add_argument('--data_user', required=True, help="Ruta del archivo de predicciones de usuarios en formato parquet")
    parser.add_argument('--output_path', required=True, help="Ruta de salida para guardar los datos finales en formato CSV")

    args = parser.parse_args()

    metrics_users = pd.DataFrame()

    # Configuración de ventanas
    input_steps = 60  # 3 meses
    output_steps = 30  # 1 mes

    # Cargar los datos
    predictions_df = cargar_datos(args.data_preds+'/*.csv') # args.cluster, args.data_gen
    X_train, X_test, y_train, y_test, X, y,scaler = pre_process_data_lstm_train(predictions_df, input_steps, output_steps)

    # model = train_model(X, y, input_steps, output_steps)
    # Entrenamiento final con todos los datos de entrenamiento
    model_final = Sequential([
        LSTM(64, activation='tanh', input_shape=(input_steps, 1), return_sequences=False),
        Dense(output_steps)
    ])
    model_final.compile(optimizer='adam', loss='mse')

    # Entrenar el modelo final con los datos de entrenamiento completos
    model_final.fit(
        X, y.squeeze(),
        validation_split=0.2,  # Validación durante el entrenamiento para verificar sobreajuste
        epochs=50,             # Incrementamos los epochs para capturar patrones más complejos
        batch_size=32,
        verbose=1
    )
    
    get_preds_lstm(X_test, y_test, model_final, scaler)

    df_users = cargar_data_user(args.data_user+'/*.csv')

    # Filtrar usuarios con datos en cualquier momento (sin restricciones de fecha)
    unique_users = df_users['user_id'].unique()

    # Iterar sobre cada usuario y filtrar los datos
    for user in unique_users:
    # Filtrar por usuario
        user_df = pd.DataFrame()
        input_steps = 7  
        output_steps = 30  # 30 días de salida

        print("Usuario:", user)
        user_df = df_users[df_users['user_id'] == user]
        user_df = user_df.sort_values('ds')
        # user_df['acumulado_movil_30'] = user_df['y'].rolling(window=30).sum()
        # user_df = user_df.drop(columns=['y'])
        # user_df = user_df.rename(columns={'acumulado_movil_30': 'y'})

        df_data_train_user = user_df[user_df["ds"] < "2023-12-01 00:00:00"]
        df_data_test_user = user_df[user_df["ds"] >= "2023-12-01 00:00:00"] 

        # Verificar si el DataFrame df_data_test_user está vacío
        if df_data_test_user.empty:
            print(f"El usuario {user} no tiene datos suficientes en el conjunto de entrenamiento. Se omite.")
            continue  # Saltar al siguiente usuario si está vacío
        
        # Proceder con el cálculo si el DataFrame no está vacío
        try:
            # Usamos solo la columna 'y' para las predicciones
            X_test_user, y_test_user = create_sequences_users(df_data_test_user['y'].values, input_steps, output_steps)
            X_test_user = X_test_user.reshape((X_test_user.shape[0], X_test_user.shape[1], 1))
            print(f"X_test: {X_test_user.shape}, y_test: {y_test_user.shape}")

            # Realizamos las predicciones con el modelo LSTM
            y_pred = model_final.predict(X_test_user)

            metrics_df = preds_users(X_test_user, y_test_user, y_pred, scaler, user, args.cluster)

            # print(metrics_users)
            metrics_users = pd.concat([metrics_users, metrics_df], ignore_index=True)
        except Exception as e:
            print(f"Error procesando el usuario {user}: {e}")
            continue

    print(metrics_users)
    metrics_users.to_csv(args.output_path+'/lstm_metrics_users_2.csv', index=False)


if __name__ == "__main__":
    main()

    #'predictions_dia_avg/predictions_0.csv/*.csv'

    # python train_predict_users_LSTM.py --data_preds predictions_dia_avg_v3/predictions_2.csv --data_user data_testing/cluster_2_sum_V3.csv --output_path res_lstm_cluster_users/ --cluster 2
