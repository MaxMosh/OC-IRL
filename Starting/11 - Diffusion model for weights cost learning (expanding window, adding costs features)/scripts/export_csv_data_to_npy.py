import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paths
# EXPERIMENTAL_DATA_PATH = "../8 - Diffusion model with conditionning (Berret data)/data/S01/Trial45.csv"
EXPERIMENTAL_DATA_PATH = "../8 - Diffusion model with conditionning (Berret data)/data/S01/Trial55.csv"

berret_df = pd.read_csv(EXPERIMENTAL_DATA_PATH, header=None)

berret_array_rad = np.array(berret_df).T

print(berret_array_rad.shape)

print(min(berret_array_rad[:,0]))
print(max(berret_array_rad[:,0]))

plt.plot(berret_array_rad[:,0])
plt.show()

print(min(berret_array_rad[:,1]))
print(max(berret_array_rad[:,1]))

plt.plot(berret_array_rad[:,1])
plt.show()


berret_array_deg = np.rad2deg(berret_array_rad)

print(min(berret_array_deg[:,0]))
print(max(berret_array_deg[:,0]))

print(min(berret_array_deg[:,1]))
print(max(berret_array_deg[:,1]))


np.save("data/berret_array_rad_2.npy", berret_array_rad)
np.save("data/berret_array_deg_2.npy", berret_array_deg)