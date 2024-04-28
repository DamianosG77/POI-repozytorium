from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import csv
import pyransac3d as pyrsc
import warnings


def load_points(file_path):
    points = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        next(reader)
        for row in reader:
            points.append([float(row[0]), float(row[1]), float(row[2])])
    return np.array(points)


#załadowanie plików
file_paths = ["Podpunkt_A.xyz", "Podpunkt_B.xyz", "Podpunkt_C.xyz"]
cluster_labels = ["płaszczyzna pionowa", "płaszczyzna pozioma", "płaszczyzna cylindryczna"]

warnings.filterwarnings("ignore", category=UserWarning)

for file_path, label in zip(file_paths, cluster_labels):
    print(f"Analiza dla pliku: {file_path}")
    points = load_points(file_path)


    scaler = StandardScaler()
    points_normalized = scaler.fit_transform(points)

    # DBSCAN klasteryzacja
    dbscan = DBSCAN(eps=0.1, min_samples=10)
    labels = dbscan.fit_predict(points_normalized)
    num_clusters = len(np.unique(labels)) - 1  # subtract 1 to account for noise points labeled as -1
    cluster_sizes = [np.sum(labels == i) for i in range(num_clusters)]
    largest_clusters_indices = np.argsort(cluster_sizes)[::-1][:3]

    clusters = []
    for i in largest_clusters_indices:
        cluster_points = points[labels == i]
        clusters.append(cluster_points)

    # dopasowanie Ransac
    for i, cluster_points in enumerate(clusters):
        plane = pyrsc.Plane()
        try:
            best_eq, _ = plane.fit(cluster_points, 0.01)
        except Exception as e:
            print("Błąd podczas dopasowania RANSAC:", e)
            continue

        normal_vector = best_eq[:3]
        print(f"Wektor normalny dla {label} - klaster {i + 1}:", normal_vector)
        distance_from_origin = -best_eq[3]

        if abs(normal_vector[2]) < 0.1:
            orientation = "pozioma"
        elif abs(normal_vector[0]) < 0.01 and abs(normal_vector[1]) < 0.01:
            orientation = "pionowa"
        else:
            orientation = "Nie można określić płaszczyzny"

        print(f"Płaszczyzna dla {label} - klaster {i + 1}: {orientation}.")
    print("\n")