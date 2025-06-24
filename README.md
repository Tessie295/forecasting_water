# Forecasting

En esta guía se presentan los pasos para reproducir y generar nuevos modelos de forecasting.


## Preparación de los datos

### Para el entrenamiento general del modelo LSTM con todos los usuarios:

Asumiendo que tenemos los datos clasificados mediante un clustering realizado anteriormente, debemos prepararlos para poder realizar las predicciones de consumo generando un modelo por cluster. Necesitaremos también todos los datos antes de esa clasificación para obtener toda la información que nos hace falta de los mismos (en nuestro caso utilizábamos este archivo `output/data/consumption.parquet` de base).

El objetivo es obtener las medias de consumo diario por cluster.

Para ello podemos hacer uso de las funciones que se encuentran en el archivo `tools/extraccion.py`. Para generar estos datos de forma más rápida se utilizó el script en bash `extraccion_clusters.sh`, al cual debemos pasarle la ruta/nombre de los nuevos archivos que vamos a generar, por ejemplo:

```
    output_path="predictions_data/predictions_subcluster_7_${cluster}_filtrado.csv" 
```
En dicha ruta se almacenarán (en este caso de ejemplo) por cada cluster, los datos del subcluster 7. Recordemos que teníamos los datos clasificados en diferentes clusters y subclusters.

En el parámetro `--data-preds` añadiremos el archivo de predicciones que hemos generado con el modelo de clustering, del cual queremos extraer los datos con las medias de consumo diario por cluster.


### Para la predicción de usuarios concretos:

Una vez tuviéramos un modelo de predicción entrenado para cada cluster en general, podríamos predecir el consumo de un usuario concreto. Para ello, debemos seguir los siguientes pasos que se encuentran documentados en `tools/extraccion_notebook.ipynb `. De nuevo aquí en este notebook estamos utilizando las funciones que se encuentran en `tools/extraccion.py`, solo que para comprobar que la extracción se hacía correctamente sobre el cluster que nos interesaba, no se automatizó como en el caso de la extracción de los datos generales.

Otra diferencia es que esta vez se calcula el acumulado del consumo de díario por usuario (código que se encuentra en la última celda):

```
df = df_filtrado.withColumn("day", F.to_date("date"))

# Agrupar por 'day' y 'serial_number' y calcular la media de 'value' solo con valores no nulos
df_sum_day = df.groupBy("day", "serial_number") \
                .agg(F.sum("value").alias("sum_value")) \
                .orderBy("day", "serial_number")
df_sum_day.show()
df_sum_day.write.csv('data_testing/cluster_sub1_3_sum_V3.csv', header=True, mode='overwrite')
```
## Entrenamiento del modelo de predicción

En el notebook `NN_LSTM_v2.ipynb` se encuentra el código y los pasos para generar el modelo de predicción. En dicho notebook, en primer lugar se cargan los datos extraídos anteriormente y posteriormente como parte del pre-procesamiento de datos, se calcula el acumulado móvil de los datos con una ventana de 30 días. 

```
# Creamos una columna 'acumulado_movil_30' que suma los valores de 'y' en ventanas móviles de 30 días
df['acumulado_movil_30'] = df['y'].rolling(window=30).sum()
```

Seguidamentes se filtran los datos según las fechas con las que nos interese trabajar y se presentan diferentes gráficos con los datos. Se dividen los datos en train y test y se procede a realizar las pruebas y entrenamientos para obtener un modelo. Un paso importante es la creación de secuencias de datos para que el modelo pueda trabajar con ellos. Para ello se utiliza la función `create_sequences`:

```
# Leemos los datos desde el dataframe de entrenamiento
data = df_data_train

# Escalamos los datos entre 0 y 1 usando MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data[['y']])
data['y_scaled'] = data_scaled

# Función para crear secuencias de entrada y salida usando ventanas deslizantes
def create_sequences(data, input_steps, output_steps):
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps + 1):
        # La secuencia de entrada son los input_steps anteriores
        X.append(data[i:i + input_steps])
        # La secuencia de salida son los output_steps siguientes
        y.append(data[i + input_steps:i + input_steps + output_steps]) 
    return np.array(X), np.array(y)

# Definimos el tamaño de las ventanas de entrada y salida
input_steps = 60  # Ventana de entrada de 60 días (2 meses)
output_steps = 30  # Ventana de predicción de 30 días (1 mes)

# Generamos las secuencias de entrada X y salida y
X, y = create_sequences(data_scaled, input_steps, output_steps)

# Dividimos los datos en conjuntos de entrenamiento (80%) y prueba (20%) de forma aleatoria
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```



La celda que más nos interesa tras las pruebas de entrenamiento y comprobaciones de métricas, es la que contiene este código con el que finalmente obtenemos el modelo entrenado:

```
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

# Guardar el modelo en formato .h5
model_final.save('output/models_forecasting/model_final_cluster_1.h5')
```


## Predicción para los datos de un usuario concreto

En la segunda parte del el notebook `NN_LSTM_v2.ipynb` se encuentra ejemplificado este proceso, cargararíamos los datos anteriomente extraídos (con el acumulado del consumo diario), filtrariamos los datos con un usuario concreto:

```
serial_to_predict = "J18YA004141"  # Serial que quieres predecir 
df_filtered = df_users[df_users['serial_number'] == serial_to_predict]
df_filtered = df_filtered.sort_values('ds')
df_filtered.head()
```

Filtrariamos por las fechas que nos interesan, creación de las secuencias como se ha mostrado anteriormente, división de train/test para la evaluación posterior... y finalmente obtendríamos una predicción con el modelo entrenado con todos los datos para el cluster elegido:

```
# Realizamos las predicciones con el modelo LSTM
y_pred = model_final.predict(X_test)

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
```

