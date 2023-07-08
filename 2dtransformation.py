import pandas as pd
import numpy as np

# Read Excel
picnum = input("Which Picture??[1,4]:")
picnum = int(picnum) -1

df = pd.read_excel("Input\points.xlsx", sheet_name=picnum)

points1 = list(df[['x', 'y']].itertuples(index=False, name=None))
points2 = list(df[['x_d', 'y_d']].itertuples(index=False, name=None))

def conformal_transform(params, x, y):
    a, b, c, d = params
    x_t = a * x + b * y + c
    y_t = -b * x + a * y + d
    return x_t, y_t

def calculate_parameters(points1, points2):
    # Create matrix A
    A = np.zeros((2 * len(points1), 4))
    for i in range(len(points1)):
        A[2 * i] = [points1[i][0], points1[i][1], 1, 0]
        A[2 * i + 1] = [points1[i][1], -points1[i][0], 0, 1]

    # Create matrix B
    B = np.array(points2).reshape(2 * len(points2))

    # Solve for the parameters
    params, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

    return params

params = calculate_parameters(points1, points2)
landa = np.sqrt(params[0] ** 2 + params[1] ** 2)
kappa = np.degrees(np.arctan(params[1] / params[0]))
print("------------------- Parameters --------------------")
print("a = ", params[0])
print("b = ", params[1])
print("c = ", params[2])
print("d = ", params[3])
print("\u03BB = ", landa)
print("\u03BA (Degrees) = ", kappa)

# Apply the conformal transformation to the points
transformed_points = [conformal_transform(params, x, y) for x, y in points1]

print("------------------- Transformed Points --------------------")

# Print the transformed points
for i, point in enumerate(transformed_points):
    print(f"Transformed point {i+1}: {point}")

print("------------------- RMSE --------------------")

# Calculate the RMSE
squared_errors = [(x2 - x1)**2 + (y2 - y1)**2 for ((x1, y1), (x2, y2)) in zip(points2, transformed_points)]
mean_squared_error = sum(squared_errors) / len(squared_errors)
rmse = np.sqrt(mean_squared_error)
print("RMSE = ", rmse)