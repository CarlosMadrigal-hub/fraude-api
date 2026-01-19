import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from collections import Counter
import zipfile  # <--- Librería necesaria

def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def perform_kmeans_analysis(n_clusters, file_path):
    # 1. Cargar datos desde ZIP
    try:
        # Abrimos el ZIP en modo lectura
        with zipfile.ZipFile(file_path, 'r') as z:
            # Buscamos el primer archivo que termine en .csv dentro del zip
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            
            if not csv_files:
                return {"error": "El archivo ZIP no contiene ningún CSV."}
            
            # Leemos el primer CSV encontrado
            with z.open(csv_files[0]) as f:
                df = pd.read_csv(f)
                
    except FileNotFoundError:
        return {"error": "No se encontró el archivo creditcard.zip en el servidor."}
    except zipfile.BadZipFile:
        return {"error": "El archivo está corrupto o no es un ZIP válido."}
    except Exception as e:
        return {"error": f"Error leyendo el archivo: {str(e)}"}

    # 2. Selección de características (Igual que antes)
    top_features = ['V17', 'V14', 'V16', 'V12', 'V10', 'V11', 'V18']
    
    # Verificamos que las columnas existan (por seguridad)
    if not all(col in df.columns for col in top_features):
        return {"error": "El CSV no tiene las columnas necesarias (V17, V14...)"}

    X = df[top_features].copy()
    y = df["Class"].copy()

    # 3. Ejecutar KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    # 4. Análisis por Cluster
    counter = Counter(clusters)
    bad_counter = Counter(clusters[y == 1])
    
    cluster_details = []
    for i in range(n_clusters):
        total_samples = counter[i]
        fraud_samples = bad_counter[i]
        fraud_percentage = (fraud_samples / total_samples * 100) if total_samples > 0 else 0
        
        cluster_details.append({
            "cluster_id": i,
            "total_samples": total_samples,
            "fraud_samples": fraud_samples,
            "fraud_percentage": round(fraud_percentage, 4),
            "is_high_risk": fraud_percentage > 50
        })

    # 5. Métricas
    metrics_result = {
        "purity_score": round(purity_score(y, clusters), 4),
        "silhouette_score": round(metrics.silhouette_score(X, clusters, sample_size=10000), 4),
        "calinski_harabasz": round(metrics.calinski_harabasz_score(X, clusters), 2)
    }

    return {
        "n_clusters_used": n_clusters,
        "metrics": metrics_result,
        "clusters_analysis": cluster_details
    }