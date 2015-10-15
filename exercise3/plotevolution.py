#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) < 2:
  print("Enter input file.")
  sys.exit(1)

data = np.loadtxt(sys.argv[1], dtype=np.float32, delimiter=",")

ds = np.unique(data[:, 0])
t = np.unique(data[:, 1])
n = [data[data[:, 0] == d, 2] for d in ds]
muSquared = [data[data[:, 0] == d, 3] for d in ds]

figN, axN = plt.subplots()
figMu, axMu = plt.subplots()
for i, d in enumerate(ds):
  axN.plot(t, n[i], "+-", label="D = {}".format(int(d)), markersize=10,
           markeredgewidth=2)
  axMu.plot(t, muSquared[i], "+-", label="D = {}".format(int(d)), markersize=10,
            markeredgewidth=2)
axN.set_yscale("log")
axN.set_xlabel("t")
axN.set_ylabel("N")
axN.legend(loc=1)
axMu.set_yscale("log")
axMu.set_xlabel("t")
axMu.set_ylabel("mu^2")
axMu.legend(loc=1)
plt.show()
