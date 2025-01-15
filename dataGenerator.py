import numpy as np
import os
N = 160 
t = 300
for i in range (N):
    for j in range(t):
        points = np.random.rand(2500, 4)
        folder_path = f"data/sequence_{i}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.savetxt(f"{folder_path}/t_{j}.pts", points, fmt="%.6f")

