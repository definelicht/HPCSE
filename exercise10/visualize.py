import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) < 2:
  print("Usage: <path to input file>")
  sys.exit(1)

data = np.loadtxt(sys.argv[1], delimiter=",")
time = data[:, 0]
data = data[:, 1:]
nSnapshots = data.shape[0]
nParticles = data.shape[1]
y = np.zeros(nParticles)
fig, axs = plt.subplots(nSnapshots, 1)
xMin = float('inf')
xMax = float('-inf')
for i in range(nSnapshots):
  axs[i].plot(data[i, :], y, "o", markersize=30, alpha=0.25)
  xMin = min([xMin, axs[i].get_xlim()[0]])
  xMax = max([xMax, axs[i].get_xlim()[1]])
for ax in axs:
  ax.set_xlim([xMin, xMax])
plt.show()
