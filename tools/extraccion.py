from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pyspark.sql.functions import udf, collect_list
from pyspark.sql.types import ArrayType, FloatType
from pyspark.sql.functions import year, col
import numpy as np
import argparse

# Crear una sesión de Spark
# Configurar el tamaño de la memoria del driver y de los ejecutores
spark = SparkSession.builder \
    .appName("EDA SmartWater ") \
    .config("spark.driver.memory", "120g") \
    .config("spark.executor.memory", "120g") \
    .config("spark.driver.maxResultSize", "120g") \
    .getOrCreate()

def carga_datos(data_preds, cluster, data_gen='output/data/consumption.parquet'):
    # Carga de las predicciones generales
    df_data = spark.read.parquet(data_preds) #'output/data/predictions_ALL_filtered.parquet'

    # Cluster a filtrar
    df_filtered = df_data[df_data['prediction'] == cluster]
    df = df_filtered.drop('features').drop('prediction')

    # Carga de todos los datos
    df_all = spark.read.parquet(data_gen)
    grouped_df = df_all.withColumn("year", year(col("date"))).orderBy("date")
    # grouped_df = df_all.orderBy("date").groupBy("serial_number").agg(collect_list("value").alias("value_list"))

    predictions_df = grouped_df.join(
        df_filtered.select("serial_number", "year").distinct(),
        on=["serial_number", "year"],
        how="inner"
    )

    print("Número total de predicciones: ", predictions_df.count(), "para el cluster: ", cluster)

    return predictions_df

# Filtrado de datos
def filtrado_outliers(predictions_df):
    # Calcular percentiles y eliminar valores extremos
    percentile_05 = predictions_df.approxQuantile("value", [0.05], 0.01)[0]
    percentile_95 = predictions_df.approxQuantile("value", [0.95], 0.01)[0]

    df_filtered = predictions_df.filter(predictions_df["value"] >= percentile_05).filter(predictions_df["value"] <= percentile_95)

    print("Outliers filtrados")

    return df_filtered

# Normalización
def normalizacion_datos(df):
    # Primero, convertir la columna 'valor' en un vector para usarla con el MinMaxScaler
    assembler = VectorAssembler(inputCols=["value"], outputCol="value_vector")
    df_vector = assembler.transform(df)

    # Configurar el MinMaxScaler
    scaler = MinMaxScaler(inputCol="value_vector", outputCol="value_norm")

    # Ajustar y transformar los datos
    scaler_model = scaler.fit(df_vector)
    df_normalizado = scaler_model.transform(df_vector)

    # Mostrar los resultados normalizados
    # df_normalizado.select("value", "value_norm").show()

    firstelement=F.udf(lambda v:float(v[0]),FloatType())
    df_normalizado = df_normalizado.withColumn("value_norm_num", firstelement("value_norm"))

    df_transformed = df_normalizado.drop("value", "value_vector", "value_norm")
    df_transformed = df_transformed.withColumnRenamed("value_norm_num", "value")

    print("Datos normalizados")

    return df_transformed

def agrupar_dia_hora(df, path_preds):
    # Extraer la fecha y la hora de la columna 'date'
    df = df.withColumn("day", F.to_date("date"))
    df = df.withColumn("hour", F.hour("date"))

    # Agrupar por 'day' y 'hour' y calcular la media de 'value' para cada serial_number
    df_avg_hour = df.groupBy("day", "hour", "serial_number") \
                    .agg(F.avg("value").alias("avg_value")) \
                    .orderBy("day", "hour", "serial_number")

    # Crear la columna combinada 'day_hour' para cada grupo de día y hora
    df_avg_hour = df_avg_hour.withColumn("day_hour", 
                                         F.concat(F.col("day"), F.lit(" "), F.format_string("%02d", F.col("hour"))))

    # Calcular la media total de todos los 'serial_number' para cada combinación de 'day_hour'
    df_avg_all_users = df_avg_hour.groupBy("day_hour") \
                                  .agg(F.avg("avg_value").alias("avg_all_users")) \
                                  .orderBy("day_hour")

    # Mostrar y guardar el resultado
    df_avg_all_users.show()
    df_avg_all_users.write.csv(path_preds, header=True, mode='overwrite')
    print("Predicciones guardadas en: ", path_preds)

def agrupar_dia_avg(df, path_preds):
    # Extraer solo la fecha (día) de la columna 'date'
    df = df.withColumn("day", F.to_date("date"))

    # Agrupar por 'day' y 'serial_number' y calcular la media de 'value' solo con valores no nulos
    df_avg_day = df.groupBy("day", "serial_number") \
                   .agg(F.avg("value").alias("avg_value")) \
                   .orderBy("day", "serial_number")

    # Calcular la media total de todos los 'serial_number' para cada día
    # La función F.avg("value") ignora los valores nulos de value automáticamente,
    df_avg_all_users = df_avg_day.groupBy("day") \
                                 .agg(F.avg("avg_value").alias("avg_all_users")) \
                                 .orderBy("day")

    # Mostrar y guardar el resultado
    df_avg_all_users.show()
    df_avg_all_users.write.csv(path_preds, header=True, mode='overwrite')
    print("Predicciones guardadas en: ", path_preds)

def agrupar_semana_avg(df, path_preds):
    # Extraer la semana (como año-semana) de la columna 'date'
    df = df.withColumn("week", F.date_format("date", "YYYY-ww"))

    # Agrupar por 'week' y 'serial_number' y calcular la media de 'value' solo con valores no nulos
    df_avg_week = df.groupBy("week", "serial_number") \
                    .agg(F.avg("value").alias("avg_value")) \
                    .orderBy("week", "serial_number")

    # Calcular la media total de todos los 'serial_number' para cada semana
    # La función F.avg("value") ignora los valores nulos de value automáticamente.
    df_avg_all_users = df_avg_week.groupBy("week") \
                                  .agg(F.avg("avg_value").alias("avg_all_users")) \
                                  .orderBy("week")

    # Mostrar y guardar el resultado
    df_avg_all_users.show()
    df_avg_all_users.write.csv(path_preds, header=True, mode='overwrite')
    print("Predicciones semanales totales guardadas en: ", path_preds)

def agrupar_dia_suma(df, path_preds):
    # Extraer solo la fecha (día) de la columna 'date'
    df = df.withColumn("day", F.to_date("date"))

    # Agrupar por 'day' y 'serial_number', y calcular la suma de 'value' para cada serial_number
    df_sum_day = df.groupBy("day", "serial_number") \
                   .agg(F.sum("value").alias("sum_value")) \
                   .orderBy("day", "serial_number")

    # Calcular la suma total de todos los 'serial_number' para cada día
    df_sum_all_users = df_sum_day.groupBy("day") \
                                 .agg(F.sum("sum_value").alias("sum_all_users")) \
                                 .orderBy("day")

    # Mostrar y guardar el resultado
    df_sum_all_users.show()
    df_sum_all_users.write.csv(path_preds, header=True, mode='overwrite')
    print("Predicciones con suma de valores totales guardadas en:", path_preds)


 
# Función principal
def main():
    parser = argparse.ArgumentParser(description="Script para procesar y filtrar predicciones de consumo de agua")
    parser.add_argument('--data_preds', required=True, help="Ruta del archivo de predicciones en formato parquet")
    parser.add_argument('--cluster', required=True, help="Cluster del que se quiere extraer datos")
    parser.add_argument('--data_gen', default='output/data/consumption.parquet', help="Ruta del archivo de datos generales en formato parquet")
    parser.add_argument('--output_path', required=True, help="Ruta de salida para guardar los datos finales en formato CSV")

    # Flags opcionales para filtrar y normalizar
    parser.add_argument('--filtrar_outliers', action='store_true', help="Incluir para aplicar el filtrado de outliers")
    parser.add_argument('--normalizar', action='store_true', help="Incluir para aplicar la normalización de los datos")


    args = parser.parse_args()

    # Cargar los datos
    predictions_df = carga_datos(args.data_preds, args.cluster, args.data_gen)

    if args.filtrar_outliers:
        predictions_df = filtrado_outliers(predictions_df)

    if args.normalizar:
        predictions_df = normalizacion_datos(predictions_df)

    # Agrupar por día y media valores
    agrupar_dia_avg(predictions_df, args.output_path)


if __name__ == "__main__":
    main()

# Ejemplos de uso:
# python extraccion.py --data_preds output/data/predictions_ALL_filtered.parquet --output_path predictions_data/predictions_name.csv --cluster 3
# python extraccion.py --data_preds output/data/predictions_ALL_filtered.parquet --output_path predictions_data/predictions_name.csv --filtrar_outliers --cluster 3
