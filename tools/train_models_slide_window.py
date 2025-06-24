import pandas as pd
import numpy as np
import glob
import argparse

from neuralprophet import NeuralProphet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
import logging

# Desactiva los warnings y ajusta el logging
warnings.filterwarnings("ignore")
logging.getLogger("NP").setLevel(logging.ERROR)

def cargar_datos(data):

    # Get a list of all CSV files in the directory
    csv_files = glob.glob(data)

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


    df = df.rename(columns={'day': 'ds', 'avg_all_users': 'y'})

    # Aseguramos que 'ds' es de tipo datetime
    df['ds'] = pd.to_datetime(df['ds'])

    # Eliminamos la información de zona horaria
    df['ds'] = df['ds'].dt.tz_localize(None)

    # Verifica el tipo de datos de la columna 'ds'
    print(df['ds'].dtype)
    
    df = df[df["ds"] >= "2020-01-01 00:00:00"] # Quitamos datos de 2018-2019 ya que meten ruido
    df = df[df["ds"] < "2024-01-01 00:00:00"] # Quitamos datos de 2024

    print(df.head())

    return df

def NeuralProphet_train(df, train_size, test_size):
    
    # Lista para guardar los resultados de cada ventana
    results = []

    # Realizar el proceso de la ventana deslizante
    for start in range(0, len(df) - train_size - test_size + 1, test_size):
        # Define el conjunto de entrenamiento y prueba en cada iteración
        train_df = df.iloc[start:start + train_size]
        test_df = df.iloc[start + train_size:start + train_size + test_size]
        
        model = NeuralProphet(
        # n_lags=30,             # Captura dependencias de 30 días anteriores
        n_forecasts=30,        # Predicción de 30 días hacia adelante
        yearly_seasonality=True,  # Estacionalidad anual
        weekly_seasonality=True,  # Estacionalidad semanal
        daily_seasonality=True,   # Desactivar estacionalidad diaria si no es relevante
        ar_layers=[64, 64, 64, 64],
        learning_rate=0.09,
        growth="off",
    )
        
        # Añadimos festividades nacionales
        model.add_country_holidays(country_name="ES")
        # model = model.add_events(["fiestas_magdalena", "san_juan", "fiestas_patronales"])
        # Entrenar el modelo incluyendo los eventos
        # model.fit(df, freq="D", events_df=df_events)

        # Ajusta el modelo usando la ventana de entrenamiento
        model.fit(train_df, freq="D")
        
        # Realiza la predicción sobre el conjunto de prueba
        future = model.make_future_dataframe(train_df, periods=test_size, n_historic_predictions=len(train_df))
        forecast = model.predict(future)
        
        # Extrae solo las predicciones para el período de prueba
        forecast_test = forecast.iloc[-test_size:][["ds", "yhat1"]]
        forecast_test["y_true"] = test_df["y"].values  # Añade los valores reales de prueba
        
        # Calcular MAE, RMSE y MAPE para esta ventana y almacenar los resultados
        mae = mean_absolute_error(forecast_test["y_true"], forecast_test["yhat1"])
        rmse = np.sqrt(mean_squared_error(forecast_test["y_true"], forecast_test["yhat1"]))
        mape = mean_absolute_percentage_error(forecast_test["y_true"], forecast_test["yhat1"]) * 100
        
        # Almacena los resultados en la lista
        results.append({
            "start_date": train_df["ds"].iloc[0], 
            "end_date": test_df["ds"].iloc[-1], 
            "MAE": mae, 
            "RMSE": rmse,
            "MAPE": mape
        })

    return results

def HoltWinters(df, train_size, test_size):
    # Lista para almacenar los resultados de cada ventana
    results = []

    # Crear ventana deslizante
    for start in range(0, len(df) - train_size - test_size + 1):
        # Separar datos de entrenamiento y prueba
        train = df.iloc[start:start + train_size]
        test = df.iloc[start + train_size:start + train_size + test_size]
        
        model = ExponentialSmoothing(
            train["y"],
            trend="add",
            seasonal="add",
            seasonal_periods=12  # Ajusta según la estacionalidad de tus datos
        ).fit()
        
        # Hacer predicciones sobre el conjunto de prueba
        predictions = model.forecast(test_size)
        
        # Preparar datos de prueba y predicciones para las métricas
        forecast_test = pd.DataFrame({
            "y_true": test["y"],
            "yhat1": predictions
        })
        
        # Calcular MAE, RMSE y MAPE para esta ventana y almacenar los resultados
        mae = mean_absolute_error(forecast_test["y_true"], forecast_test["yhat1"])
        rmse = np.sqrt(mean_squared_error(forecast_test["y_true"], forecast_test["yhat1"]))
        mape = mean_absolute_percentage_error(forecast_test["y_true"], forecast_test["yhat1"]) * 100
        
        # Almacena los resultados en la lista
        results.append({
            "start_date": train.index[0], 
            "end_date": test.index[-1], 
            "MAE": mae, 
            "RMSE": rmse,
            "MAPE": mape
        })

    return results

def NeuralNetworkLSTM(df, train_size, test_size):

    print('Entrenando modelo NeuralNetworkLSTM: ')
    input_window_size = 7  # Número de días pasados usados como entrada para predecir el siguiente

    # Escalamos los datos
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['y'] = scaler.fit_transform(df[['y']])

    # Lista para almacenar los resultados de cada ventana
    results = []

    # Crear ventana deslizante
    for start in range(0, len(df) - train_size - test_size + 1):
        # Separar datos de entrenamiento y prueba
        train = df.iloc[start:start + train_size]
        test = df.iloc[start + train_size:start + train_size + test_size]
        
        # Preparar los datos para la LSTM
        X_train, y_train = [], []
        for i in range(len(train) - input_window_size):
            X_train.append(train["y"].iloc[i:i + input_window_size].values)
            y_train.append(train["y"].iloc[i + input_window_size])
        
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # reshape para LSTM

        # Crear y entrenar el modelo LSTM
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(input_window_size, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=20, verbose=0)
        
        # Preparar los datos de prueba para predicciones
        X_test, y_test = [], []
        for i in range(len(test) - input_window_size):
            X_test.append(test["y"].iloc[i:i + input_window_size].values)
            y_test.append(test["y"].iloc[i + input_window_size])
        
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  # reshape para LSTM
        y_test = np.array(y_test)
        
        # Hacer predicciones
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions).flatten()  # Desescalar las predicciones
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()  # Desescalar los valores reales
        
        # Calcular MAE, RMSE y MAPE para esta ventana y almacenar los resultados
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mape = mean_absolute_percentage_error(y_test, predictions) * 100
        
        # Almacena los resultados en la lista
        results.append({
            "start_date": train.index[0], 
            "end_date": test.index[-1], 
            "MAE": mae, 
            "RMSE": rmse,
            "MAPE": mape
        })

def mean_results(results, model_name, cluster):

    # Convierte los resultados en un DataFrame para analizarlos
    results_df = pd.DataFrame(results)

    # Calcula el promedio de cada métrica
    mean_mae = results_df["MAE"].mean()
    mean_rmse = results_df["RMSE"].mean()
    mean_mape = results_df["MAPE"].mean()

    mean_metrics = pd.DataFrame({
        "Model": [model_name],
        "Cluster": [cluster],
        "MAE": [mean_mae],
        "RMSE": [mean_rmse],
        "MAPE (%)": [mean_mape]
    })

    print(mean_metrics)
    
    return mean_metrics


# Función principal
def main():
    parser = argparse.ArgumentParser(description="Script para procesar y filtrar predicciones de consumo de agua")
    parser.add_argument('--data_preds', required=True, help="Ruta del archivo de predicciones en formato parquet")
    parser.add_argument('--model', required=True, help="Elige modelo para entrenar: NeuralProphet, HoltWinters o NeuralNetworkLSTM")
    parser.add_argument('--train_size', default=365, help="Tamaño de ventana de entrenamiento")
    parser.add_argument('--test_size', default=30, help="Tamaño de ventana de test")

    parser.add_argument('--output_path', required=True, help="Ruta de salida para guardar los datos finales en formato CSV")


    args = parser.parse_args()

    metrics_train = pd.DataFrame()

    # Get a list of all CSV files in the directory
    # csv_files = sorted(glob.glob(args.data_preds+'/*csv'))
    csv_files = sorted(glob.glob(args.data_preds+'/predictions_[0-10].csv')) 
    # csv_files = sorted(glob.glob('predictions_dia_avg/predictions_sub_1_*.csv'))
    print(csv_files)
    for i, file in enumerate(csv_files):
        if i >= 6:
            cluster = i+1
        else:
            cluster = i
        cluster = i
        print(file, 'cluster:', cluster)                 
        # Cargar los datos
        predictions_df = cargar_datos(file+'/*.csv')

        if args.model == 'NeuralProphet':
            results = NeuralProphet_train(predictions_df, args.train_size, args.test_size)
        if args.model == 'HoltWinters':
            results = HoltWinters(predictions_df, args.train_size, args.test_size)
        if args.model == 'NeuralNetworkLSTM':
            results = NeuralNetworkLSTM(predictions_df, args.train_size, args.test_size)

        mean_metrics = mean_results(results, args.model, cluster)
        mean_metrics.to_csv(args.output_path+'_cluster_'+str(cluster)+'.csv', index=False)
        print("saved: ", args.output_path+'_cluster_'+str(cluster)+'.csv')

        metrics_train = pd.concat([metrics_train, mean_metrics], ignore_index=True)
    
    print(metrics_train)
    metrics_train.to_csv(args.output_path+'_means.csv', index=False)

if __name__ == "__main__":
    main()


# python train_models_slide_window.py --data_preds predictions_dia_avg/ --output_path results_models2/neuralProphet --model NeuralNetworkLSTM
# python train_models_slide_window.py --data_preds predictions_dia_avg/ --output_path results_sub_models/HoltWinters --model HoltWinters
# python train_models_slide_window.py --data_preds predictions_dia_avg/ --output_path resultados_cut_data/neuralProphet --model NeuralProphet

