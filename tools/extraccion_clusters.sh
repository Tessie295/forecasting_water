#!/bin/bash

# Definir el rango de clústeres
for cluster in {1..10}; do
    # Crear el nombre del archivo de salida usando el número de clúster
    output_path="predictions_data/predictions_subcluster_7_${cluster}_filtrado.csv"

    # Ejecutar el comando de Python pasando los argumentos correspondientes
    python3 extraccion.py --data_preds output/data/predictions_ALL_7_filtered.parquet \
                          --output_path "$output_path" \
                          --filtrar_outliers \
                          --cluster "$cluster"
done
