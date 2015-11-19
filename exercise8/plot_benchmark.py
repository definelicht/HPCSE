#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import re, sys

if len(sys.argv) < 3 or len(sys.argv) > 4:
  print("Usage: <path to benchmark file> [<path to output image file>]")
  sys.exit(1)

def parseFile(path):
  results = {}
  with open(path) as inFile:
    nThreads = None
    for line in inFile:
      if not nThreads:
        match = re.search(
            "Running on ([0-9]+) thread\(s\) for ([0-9]+)x[0-9]+ grid with " +
            "timestep 1e-[0-9]+ for ((1e\+)?[0-9]+)", line)
        if match:
          nThreads = int(match.group(1))
          gridSize = match.group(2)
          iterations = float(match.group(3))
      else:
        match = re.search("Finished in ([0-9]+\.[0-9]+) (\([0-9]+\.[0-9]+\) )?"
                          + "seconds.", line)
        if match:
          time = float(match.group(1))
          if time < 5:
            print("Time for {}x{} on {} cores too low. Discarding.".format(
                gridSize, gridSize, nThreads))
          else:
            if not gridSize in results:
              print("Found grid size {}...".format(gridSize))
              results[gridSize] = {"threads": [], "timePerIteration": []}
            results[gridSize]["threads"].append(nThreads)
            results[gridSize]["timePerIteration"].append(time/iterations)
          nThreads = None
  for label, gs in results.items():
    threads = np.array(gs["threads"], dtype=np.int)
    time = np.array(gs["timePerIteration"], dtype=np.float)

    gs["threadsUnique"] = np.sort(np.unique(threads))
    gs["timeMean"] = np.empty(gs["threadsUnique"].shape)
    gs["timeStd"] = np.empty(gs["threadsUnique"].shape)
    for (i, c) in enumerate(gs["threadsUnique"]):
      indices = threads == c
      gs["timeMean"][i] = np.mean(time[indices])
      gs["timeStd"][i] = np.std(time[indices])

    if gs["threadsUnique"][0] != 1:
      print("Missing measurement for 1 core to compute speedup")
      sys.exit(1)
  return results

def strongscaling(sequential, grid):
  plt.rcParams.update({"font.size": 15})
  fig, ax = plt.subplots()
  highest = 0
  max_cores = 0
  for size in sorted(grid, key=float):
    speedup = grid[size]["timeMean"][0]/sequential[size]["timeMean"]
    highestThis = np.max(speedup)
    max_cores = max(max_cores, np.max(grid[size]["threadsUnique"]))
    highest = highestThis if highestThis > highest else highest
    ax.plot(grid[size]["threadsUnique"],
            speedup, "-D", label="{}x{}".format(size, size))
  ax.plot(np.array([6, highest]),
          np.array([6, highest]), "--k", linewidth=2,
          label="Linear speedup")
  ax.set_title("MPI Grid Layout Strong Scaling")
  ax.set_xlabel("Number of threads")
  ax.set_ylabel("Speedup")
  ax.set_ylim(np.array([-.5, 1.17*highest]))
  ax.set_xticks(np.array([6, 12, 18, 24, 30, 36, 42, 48]))
  ax.set_yticks(np.linspace(0, 20, 11))
  ax.legend(bbox_to_anchor=(0., 0.881, 1, 0.1), loc=3, ncol=3, mode="expand",
            borderaxespad=0., fontsize=13)
  return fig, ax

def comparison(sequential, rows, grid):
  plt.rcParams.update({"font.size": 18})
  nThreads = grid["4096"]["threadsUnique"]
  baseline = sequential["4096"]["timeMean"][0]
  rows = rows["4096"]
  grid = grid["4096"]
  speedup_rows = baseline/rows["timeMean"]
  speedup_grid = baseline/grid["timeMean"]
  fig, ax = plt.subplots()
  ax.plot(nThreads, speedup_rows, "--s", color="navy", linewidth=2,
          markersize=10, label="Split by rows")
  ax.plot(nThreads, speedup_grid, "--D", color="firebrick", linewidth=2,
          markersize=10, label="Split by grid")
  ax.legend(loc=2)
  ax.set_xlabel("Number of threads")
  ax.set_ylabel("Speedup")
  ax.set_xticks([1, 6, 12, 18, 24, 30, 36, 42, 48])
  ax.set_title("MPI Split Strategy Comparison")
  return fig, ax

print("Parsing sequential benchmarks...")
sequential = parseFile(sys.argv[1])
print("Parsing C++ thread benchmarks...")
rows = parseFile(sys.argv[2])
print("Parsing MPI benchmarks...")
grid = parseFile(sys.argv[3])

figStrong, axStrong = strongscaling(sequential, grid)
figComp, axComp = comparison(sequential, rows, grid)

if len(sys.argv) == 4:
  figStrong.savefig(sys.argv[3] + "/grid_strong.pdf", bbox_inches="tight")
  figComp.savefig(sys.argv[3] + "/grid_comp.pdf", bbox_inches="tight")
else:
  figStrong.show()
  figComp.show()
  input("Press return to exit...")
