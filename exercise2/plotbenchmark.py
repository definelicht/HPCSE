#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import re, sys

if len(sys.argv) < 2 or len(sys.argv) > 3:
  print("Usage: <path to benchmark file> [<path to output image file>]")
  sys.exit(1)

results = {}
with open(sys.argv[1]) as inFile:
  nThreads = None
  for line in inFile:
    if not nThreads:
      match = re.search(
          "Running on ([0-9]+) core\(s\) for ([0-9]+)x[0-9]+ grid with " +
          "timestep 1e-[0-9]+ for ((1e\+)?[0-9]+)", line)
      if match:
        nThreads = int(match.group(1))
        gridSize = match.group(2)
        iterations = float(match.group(3))
    else:
      match = re.search("Finished in ([0-9]+\.[0-9]+) seconds.", line)
      if match:
        if not gridSize in results:
          results[gridSize] = {"cores": [], "time": []}
        results[gridSize]["cores"].append(nThreads)
        results[gridSize]["time"].append(float(match.group(1))/iterations)
        nThreads = None

for _, gs in results.items():
  cores = np.array(gs["cores"], dtype=np.int)
  time = np.array(gs["time"], dtype=np.float)

  gs["coresUnique"] = np.unique(cores)
  gs["timeMean"] = np.empty(gs["coresUnique"].shape)
  gs["timeStd"] = np.empty(gs["coresUnique"].shape)
  gs["speedupMean"] = np.empty(gs["coresUnique"].shape)
  gs["speedupStd"] = np.empty(gs["coresUnique"].shape)
  for (i, c) in enumerate(gs["coresUnique"]):
    indices = cores == c
    gs["timeMean"][i] = np.mean(time[indices])
    gs["timeStd"][i] = np.std(time[indices])
    gs["speedupMean"][i] = np.mean(gs["timeMean"][0] / time[indices])
    gs["speedupStd"][i] = np.std(gs["timeMean"][0] / time[indices])

  if gs["coresUnique"][0] != 1:
    print("Missing measurement for 1 core to compute speedup")
    sys.exit(1)

plt.rcParams.update({"font.size": 15})
fig, [time, speedup] = plt.subplots(1, 2, figsize=(12, 5))
for label, gs in results.items():
  time.plot(gs["coresUnique"], gs["timeMean"], "--D",
            label="{}x{}".format(label, label))
  speedup.plot(gs["coresUnique"], gs["speedupMean"], "--D",
               label="{}x{}".format(label, label))
time.set_xlabel("Number of threads")
time.set_ylabel("Time per iteration [s/1]")
time.legend()
speedup.set_xlabel("Number of threads")
speedup.set_ylabel("Speedup")
speedup.legend(loc=2)

if len(sys.argv) == 3:
  plt.savefig(sys.argv[2], bbox_inches="tight")
else:
  plt.show()

