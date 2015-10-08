#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) < 2 or len(sys.argv) > 3:
  print("Usage: <path to input file> [<path to output image file>]")
  sys.exit(1)

data = np.loadtxt(sys.argv[1], delimiter=",")
dim = data.shape[1];
snapshots = []
for i in range(int(data.shape[0] / dim)):
  snapshots.append(data[i*dim:(i+1)*dim, :])

fig, axs = plt.subplots(1, len(snapshots))
for i in range(len(axs)):
  axs[i].imshow(snapshots[i])

if len(sys.argv) == 3:
  plt.savefig(sys.argv[2], bbox_inches="tight")
else:
  plt.show()
