import os
from sklearn.cluster import KMeans
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

os.environ["OMP_NUM_THREADS"] = "6"

# Load points from three different files
file_paths = ["Podpunkt_A.xyz", "Podpunkt_B.xyz", "Podpunkt_C.xyz"]
cluster_labels = ["płaszczyzna pionowa", "płaszczyzna pozioma", "płaszczyzna cylindryczna"]

warnings.filterwarnings("ignore", category=UserWarning)

for file_path, label in zip(file_paths, cluster_labels):
    print(f"Analiza dla pliku: {file_path}")
    points = load_points(file_path)

    # K-Means clustering
    kmeans = KMeans(n_clusters=3, n_init=10)
    labels = kmeans.fit_predict(points)

    # Assign points to clusters
    clusters = []
    for i in range(3):
        cluster_points = points[labels == i]
        clusters.append(cluster_points)

    # RANSAC fitting
    for i, cluster_points in enumerate(clusters):
        plane = pyrsc.Plane()
        best_eq, _ = plane.fit(cluster_points, 0.01)

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