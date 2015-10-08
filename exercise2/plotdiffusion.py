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
if len(snapshots) != 4:
  print("Expected 4 snapshots, but received {}.".format(len(snapshots)))
  sys.exit(1)

plt.rcParams.update({'font.size': 8})
fig, axs = plt.subplots(1, 4)
for i in range(4):
  axs[i].imshow(snapshots[i])
  axs[i].set_xlabel("x")
axs[0].set_ylabel("y")
axs[0].set_title("t=0")
axs[1].set_title("t=0.5")
axs[2].set_title("t=1")
axs[3].set_title("t=2")

if len(sys.argv) == 3:
  plt.savefig(sys.argv[2], bbox_inches="tight")
else:
  plt.show()
