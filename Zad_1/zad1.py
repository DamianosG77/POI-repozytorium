import numpy as np

def generate_points_flat_horizontal(width, length, num_points):
    x = np.random.uniform(low=-width/2, high=width/2, size=num_points)
    y = np.random.uniform(low=-length/2, high=length/2, size=num_points)
    z = np.zeros(num_points)
    return np.column_stack((x, y, z))


width = 500
length = 450
num_points = 1500
points = generate_points_flat_horizontal(width, length, num_points)
print(points)

np.savetxt("Podpunkt_A.xyz", points, fmt='%.6f', delimiter=' ', header='x y z', comments='')
print("Wyniki zostały zapisane do pliku Podpunkt_A.xyz")