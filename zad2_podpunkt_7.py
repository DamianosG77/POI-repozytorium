import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
import open3d as o3d

# Wczytanie chmury punktów z pliku conferenceRoom_1.txt
cloud = o3d.io.read_point_cloud("conferenceRoom_1.txt")

if 'cloud_planes' in locals():
    o3d.visualization.draw_geometries([cloud_planes])
else:
    print("Chmura punktów nie została wczytana poprawnie, więc nie można jej wyświetlić.")
# Definicja funkcji do dopasowania płaszczyzn
def fit_planes(cloud, num_iterations):
    for i in range(num_iterations):
        print("Iteracja:", i)
        print("Liczba punktów przed dopasowaniem:", len(cloud.points))

        # Obliczenie wektorów normalnych płaszczyzn za pomocą PCA
        pca = PCA(n_components=3)
        pca.fit(np.asarray(cloud.points))
        normals = pca.components_

        # Wykrywanie punktów płaszczyznowych
        kmeans = KMeans(n_clusters=1)
        kmeans.fit(normals)
        center = kmeans.cluster_centers_[0]
        idx = pairwise_distances_argmin_min(center.reshape(1, -1), normals)[0]

        # Usunięcie punktów płaszczyznowych z chmury
        cloud = cloud.select_by_index(np.asarray(idx))

        print("Liczba punktów po dopasowaniu:", len(cloud.points))

    return cloud


# Ustalenie liczby iteracji
K = 6

# Przeprowadzenie procedury iteracyjnego dopasowania płaszczyzn
try:
    cloud_planes = fit_planes(cloud, K)
except Exception as e:
    print("Wystąpił błąd:", e)

