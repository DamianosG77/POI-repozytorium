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

points = load_points("Podpunkt_B.xyz") #wczytanie pliku z chmurą punktów

warnings.filterwarnings("ignore", category=UserWarning)

# Znalezienie rozłącznych chmur punktów za pomocą k-średnich (k=3)
kmeans = KMeans(n_clusters=3, n_init=10)
labels = kmeans.fit_predict(points)

# Przypisanie punktów do odpowiednich chmur
clusters = []
for i in range(3):
    cluster_points = points[labels == i]
    clusters.append(cluster_points)

# Dopsaowanie Ransac
for i, cluster_points in enumerate(clusters):
    plane = pyrsc.Plane()
    best_eq, _ = plane.fit(cluster_points, 0.01)

    normal_vector = best_eq[:3]
    print(f"Wektor normalny dla chmury punktów {i + 1}:", normal_vector)
    distance_from_origin = -best_eq[3]

    if abs(normal_vector[2]) < 0.1:
        orientation = "pozioma"
    elif abs(normal_vector[0]) < 0.1 and abs(normal_vector[1]) < 0.1:
        orientation = "pionowa"
    else:
        orientation = "inna"

    print(f"Płaszczyzna dla chmury punktów {i + 1} jest {orientation}.")