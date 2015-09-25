#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import re, sys

if len(sys.argv) < 2 or len(sys.argv) > 3:
  print("Usage: <path to benchmark file> [<path to output image file>]")
  sys.exit(1)

results = {"cores": [], "time": []}
with open(sys.argv[1]) as inFile:
  nThreads = None
  for line in inFile:
    if not nThreads:
      match = re.search("Running on ([0-9]+) threads", line)
      if match:
        nThreads = int(match.group(1))
    else:
      match = re.search("Finished in ([0-9]+\.[0-9]+)", line)
      if match:
        results["cores"].append(nThreads)
        results["time"].append(float(match.group(1)))
        nThreads = None

cores = np.array(results["cores"], dtype=np.int)
time = np.array(results["time"], dtype=np.float)

iSorted = np.argsort(cores)
cores = cores[iSorted]
time = time[iSorted]

if cores[0] != 1:
  print("Missing measurement for 1 core to compute speedup")
  sys.exit(1)

seqTime = np.mean(time[cores == 1])

fig, ax = plt.subplots()
ax.plot(cores, cores, "--k", linewidth=2, label="Linear speedup")
ax.plot(cores, seqTime/time, "-o", color="Firebrick", linewidth=2, markersize=6,
        markeredgewidth=1, label="Brutus speedup")
ax.set_xlabel("Number of threads")
ax.set_ylabel("Speedup")
ax.set_xticks(cores)
ax.legend(loc=2)

if len(sys.argv) == 3:
  plt.savefig(sys.argv[2], bbox_inches="tight")
else:
  plt.show()

