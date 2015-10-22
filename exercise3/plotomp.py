#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import re, sys

for arg in sys.argv:
  with open(arg) as dataFile:
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

