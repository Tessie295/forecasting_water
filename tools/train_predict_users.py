import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
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


def pre_process_data(df):

    # Creamos ventanas del acumulado
    df['acumulado_movil_30'] = df['y'].rolling(window=30).sum()
    df = df.drop(columns=['y'])
    df = df.rename(columns={'acumulado_movil_30': 'y'})

    # Quitamos datos de 2018-2019 ya que meten ruido
    df = df[df["ds"] >= "2020-02-01 00:00:00"] 

    # Obtenemos datos de train-test
    df_data_train = df[df["ds"] < "2023-06-01 00:00:00"] # Entrenamos con datos inferiores a Junio de 2023
    df_data_test = df[df["ds"] >= "2023-06-01 00:00:00"] 

    return df_data_train, df_data_test

def train_model(train):

    # Crear y entrenar el modelo NeuralProphet
    model = NeuralProphet(
        # n_lags=90,             # Captura dependencias de 30 días anteriores
        n_forecasts=30,        # Predicción de 30 días hacia adelante
        yearly_seasonality=True,  # Estacionalidad anual
        weekly_seasonality=True,  # Estacionalidad semanal
        daily_seasonality=True,   # Desactivar estacionalidad diaria si no es relevante
        ar_layers=[64, 64, 64, 64],
        learning_rate=0.09,
        growth="off"
        )

    model.add_country_holidays(country_name="ES")

    # Entrenar el modelo
    history = model.fit(train, freq="D")

    return model

def get_predictions_results(model, train, test, cluster, path_data):
    # Realizar predicciones en el conjunto de prueba
    future = model.make_future_dataframe(train, periods=len(test), n_historic_predictions=True)
    forecast = model.predict(future)

    # resum_forecast = forecast[['ds','y','yhat1']]

    # Extraer predicciones y valores reales para calcular métricas
    results_df = forecast[["ds", "yhat1"]].merge(train, on="ds", how="left")
    results_df.rename(columns={"yhat1": "y_pred", "y": "y_real"}, inplace=True)

    # Guardar resultados en CSV
    # results_df.to_csv(path_data+'results_'+str(cluster)+'.csv', index=False)

    return results_df


def model_metrics(results_df, path_data, cluster):

    results_df = results_df[results_df["ds"] < "2023-06-01 00:00:00"] 
    # Calcular métricas
    mae = mean_absolute_error(results_df["y_real"].dropna(), results_df["y_pred"].dropna())
    rmse = mean_squared_error(results_df["y_real"].dropna(), results_df["y_pred"].dropna(), squared=False)

    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    mape = mean_absolute_percentage_error(results_df["y_real"].dropna(), results_df["y_pred"].dropna())

    # Imprimir métricas
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")

    metrics_df = pd.DataFrame({"Cluster": [cluster], "MAE": [mae], "RMSE": [rmse], "MAPE": [mape]})

    # Guardar resultados en CSV
    metrics_df.to_csv(path_data+'/metrics_'+str(cluster)+'.csv', index=False)
    print("Metricas modelo guardadas")


    # Graficar resultados
    plt.figure(figsize=(14, 7))
    plt.plot(results_df["ds"], results_df["y_real"], label="Real")
    plt.plot(results_df["ds"], results_df["y_pred"], label="Predicción", linestyle="--")
    plt.xlabel("Fecha")
    plt.ylabel("Valores")
    plt.title(f"NeuralProphet: Comparación entre valores reales y predicciones\nMAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
    plt.legend()
    plt.grid()
    
    # Guardar el gráfico como imagen
    plt.savefig(path_data+'/grafica'+str(cluster)+'.png')

    print("Gráfica modelo guardadas")


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

    # df_users = df_users[(df_users['ds'] >= '2020-01-01') & (df_users['ds'] <= '2024-12-31')]
    df_users = df_users[df_users["ds"] >= "2020-02-01 00:00:00"]

    print(df_users.head())

    return df_users

def user_predictions(model, train_user):

    future = model.make_future_dataframe(train_user, periods=30, n_historic_predictions=True)  # 31 días para diciembre de 2023
    forecast_user = model.predict(future)
    # print(forecast_user[['ds', 'y', 'yhat1']])

    return forecast_user

def metrics_users_res(user_id, forecast_user, test, cluster):
    # Combinar valores reales y predichos en un solo DataFrame
    results = pd.merge(test, forecast_user[['ds', 'yhat1']], on='ds', how='inner')

    # Renombrar columnas para mayor claridad
    results = results.rename(columns={'y': 'y_real', 'yhat1': 'y_pred'})

    # Calcular el error absoluto y los porcentajes de error
    results['error'] = results['y_real'] - results['y_pred']
    results['abs_error'] = results['error'].abs()
    results['abs_pct_error'] = (results['abs_error'] / results['y_real']) * 100  # Error porcentual

    # Calcular las métricas
    # MAE y RMSE con scikit-learn
    y_real = results['y_real']
    y_pred = results['y_pred']

    mae = mean_absolute_error(y_real, y_pred)
    rmse = mean_squared_error(y_real, y_pred, squared=False)  # RMSE
    # MAPE (ya calculado previamente)
    mape = results['abs_pct_error'].mean()

    # 3. Mostrar resultados
    # print(f"MAE: {mae}")
    # print(f"RMSE: {rmse}")
    # print(f"MAPE: {mape}%")

        # Agregar las métricas al DataFrame de métricas

    metrics_df = pd.DataFrame({"Cluster": [cluster],
                      "user_id": [user_id],
                      "MAE": [mae], 
                      "RMSE": [rmse],
                      "MAPE": [mape]})

    # Guardar resultados en CSV
    # metrics_df.to_csv(path_data, index=False)
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

    # Cargar los datos
    predictions_df = cargar_datos(args.data_preds+'/*.csv') # args.cluster, args.data_gen
    train, test = pre_process_data (predictions_df)

    # Entrenamos modelo y guardamos resultado
    model = train_model(train)
    results_model = get_predictions_results(model, train, test, args.cluster, args.output_path)
    model_metrics(results_model, args.output_path, args.cluster)

    df_users = cargar_data_user(args.data_user+'/*.csv')

    # Filtrar usuarios con datos desde junio de 2023 en adelante
    users_after_june_2023 = df_users[df_users['ds'] >= '2023-06-01']['user_id'].unique()

    # Filtrar usuarios con datos en cualquier momento (sin restricciones de fecha)
    unique_users = df_users['user_id'].unique()

    # Intersección de usuarios
    valid_users = set(users_after_june_2023) & set(unique_users)

    print("Hay un total de: ", len(valid_users), 'usuarios con datos de junio de 2023 en adelante del total que son', len(unique_users))

    # Iterar sobre cada usuario y filtrar los datos
    for user in valid_users:
    # Filtrar por usuario
        user_df = pd.DataFrame()

        print("Usuario:", user)
        user_df = df_users[df_users['user_id'] == user]
        user_df = user_df.sort_values('ds')
        user_df['acumulado_movil_30'] = user_df['y'].rolling(window=30).sum()
        user_df = user_df.drop(columns=['y'])
        user_df = user_df.rename(columns={'acumulado_movil_30': 'y'})

        df_data_train_user = user_df[user_df["ds"] < "2023-06-01 00:00:00"]
        df_data_test_user = user_df[user_df["ds"] >= "2023-06-01 00:00:00"] 

        df_data_train_user = df_data_train_user[['ds','y']]
        # Verificar si el DataFrame df_data_train_user está vacío
        if df_data_train_user.empty:
            print(f"El usuario {user} no tiene datos suficientes en el conjunto de entrenamiento. Se omite.")
            continue  # Saltar al siguiente usuario si está vacío
        
        # Proceder con el cálculo si el DataFrame no está vacío
        try:
            forecast_user = user_predictions(model, df_data_train_user)
            results_user = metrics_users_res(user, forecast_user, df_data_test_user, args.cluster)
            metrics_users = pd.concat([metrics_users, results_user], ignore_index=True)
            # print(metrics_users)
        except Exception as e:
            print(f"Error procesando el usuario {user}: {e}")
            continue

    print(metrics_users)
    metrics_users.to_csv(args.output_path+'metrics_users_7.csv', index=False)


if __name__ == "__main__":
    main()

    #'predictions_dia_avg/predictions_0.csv/*.csv'

    # python train_predict_users.py --data_preds predictions_dia_avg/predictions_1.csv --data_user data_testing/cluster_1_sum.csv --output_path resultados_experimento_usuarios/ --cluster 0
