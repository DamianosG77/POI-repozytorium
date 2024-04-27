import numpy as np
def generate_points_flat_vertical(width, height, num_points):
    x = np.random.uniform(low=-width/2, high=width/2, size=num_points)
    y = np.zeros(num_points)
    z = np.random.uniform(low=0, high=height, size=num_points)
    return np.column_stack((x, y, z))

width = 50
height = 10
num_points = 1500
points = generate_points_flat_vertical(width, height, num_points)
print(points)

np.savetxt("Podpunkt_B.xyz", points, fmt='%.6f', delimiter=' ', header='x y z', comments='')

print("Wyniki zostały zapisane do pliku Podpunkt_B.xyz")