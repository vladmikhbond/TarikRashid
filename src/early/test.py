import numpy as np

# Початкова матриця
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Видалення останнього стовбця
new_matrix = np.delete(matrix, -1, axis=1)

print(type(new_matrix[0]))