import numpy as np

def generate_points_cylindrical_surface(radius, height, num_points):
    angle = np.random.uniform(low=0, high=2*np.pi, size=num_points)
    z = np.random.uniform(low=0, high=height, size=num_points)
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    return np.column_stack((x, y, z))

radius = 5
height = 100
num_points = 10000
points = generate_points_cylindrical_surface(radius, height, num_points)
print(points)

np.savetxt("Podpunkt_C.xyz", points, fmt='%.6f', delimiter=' ', header='x y z', comments='')
print("Wyniki zostały zapisane do pliku Podpunkt_C.xyz")