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

def strongscaling(manual, mpi):
  plt.rcParams.update({"font.size": 15})
  fig, ax = plt.subplots()
  highest = 0
  max_cores = 0
  for size in sorted(mpi, key=float):
    speedup = manual[size]["timeMean"][0]/mpi[size]["timeMean"]
    highestThis = np.max(speedup)
    max_cores = max(max_cores, np.max(mpi[size]["threadsUnique"]))
    highest = highestThis if highestThis > highest else highest
    ax.plot(mpi[size]["threadsUnique"],
            speedup, "-D", label="{}x{}".format(size, size))
  ax.plot(np.array([1, highest]),
          np.array([1, highest]), "--k", linewidth=2,
          label="Linear speedup")
  ax.set_title("MPI Strong Scaling")
  ax.set_xlabel("Number of threads")
  ax.set_ylabel("Speedup w.r.t. sequential implementation")
  ax.set_ylim(np.array([-.5, 1.17*highest]))
  ax.set_xticks(np.array([1, 6, 12, 18, 24, 30, 36, 42, 48]))
  ax.set_yticks(np.linspace(0, 20, 11))
  ax.legend(bbox_to_anchor=(0., 0.881, 1, 0.1), loc=3, ncol=3, mode="expand",
            borderaxespad=0., fontsize=13)
  return fig, ax

def weakscaling(manual, mpi):
  plt.rcParams.update({"font.size": 18})
  fig, ax = plt.subplots()
  for nThreads in [6, 12, 18, 24, 30, 36, 42, 48]:
    sizes = []
    speedup = []
    for size in mpi:
      sizes.append(int(size))
      speedup.append(manual[size]["timeMean"][0]
                     /mpi[size]["timeMean"][mpi[size]["threadsUnique"] ==
                                            nThreads][0])
    sizes = np.array(sizes)
    speedup = np.array(speedup)
    i_sorted = np.argsort(sizes)
    sizes = np.array(sizes[i_sorted])
    speedup = speedup[i_sorted]
    ax.plot(sizes, speedup, "--D", markersize=10,
            linewidth=2, label="{} MPI Processes".format(nThreads))
  ax.set_xscale("log", basex=2)
  ax.set_xlim([1.5*64, 1.5*4096])
  # ax.set_yticks(np.linspace(0, 1.1*np.max(speedup), 10, dtype=np.int))
  # ax.set_ylim([-0.5, 1.1*np.max(speedup)])
  ax.set_ylabel("Speedup")
  ax.set_xlabel("Grid size")
  ax.set_title("MPI Weak Scaling")
  ax.legend(loc=2, fontsize=16)
  return fig, ax

def comparison(manual, mpi):
  plt.rcParams.update({"font.size": 18})
  nThreads = manual["4096"]["threadsUnique"]
  baseline = manual["4096"]["timeMean"][0]
  manual = manual["4096"]
  mpi = mpi["4096"]
  speedup_manual = baseline/manual["timeMean"]
  speedup_mpi = baseline/mpi["timeMean"][mpi["threadsUnique"] <= 24]
  fig, ax = plt.subplots()
  ax.plot(nThreads, speedup_manual, "--s", color="navy", linewidth=2,
          markersize=10, label="C++ Threads")
  ax.plot(nThreads, speedup_mpi, "--D", color="firebrick", linewidth=2,
          markersize=10, label="MPI")
  ax.legend(loc=2)
  ax.set_xlabel("Number of threads")
  ax.set_ylabel("Speedup")
  ax.set_xticks([1, 6, 12, 18, 24])
  ax.set_title("Implementation Comparison")
  return fig, ax

print("Parsing C++ thread benchmarks...")
manual = parseFile(sys.argv[1])
print("Parsing MPI benchmarks...")
mpi = parseFile(sys.argv[2])

figStrong, axStrong = strongscaling(manual, mpi)
figWeak, axWeak = weakscaling(manual, mpi)
figComp, axComp = comparison(manual, mpi)

print(("Highest single node performance: {} seconds per iteration per grid size "
      + "squared.").format(
          mpi["4096"]["timeMean"][mpi["4096"]["threadsUnique"] == 24][0]
          /4096**2))

if len(sys.argv) == 4:
  figWeak.savefig(sys.argv[3] + "/mpi_weak.pdf", bbox_inches="tight")
  figStrong.savefig(sys.argv[3] + "/mpi_strong.pdf", bbox_inches="tight")
  figComp.savefig(sys.argv[3] + "/mpi_comp.pdf", bbox_inches="tight")
else:
  figWeak.show()
  figStrong.show()
  figComp.show()
  input("Press return to exit...")
