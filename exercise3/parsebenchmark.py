#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import re, sys

with open(sys.argv[1]) as inFile:
  with open(sys.argv[2]) as outFile:
    ompThreads = None if len(sys.argv) < 4 else int(sys.argv[3])
    nThreads = None
    for line in inFile:
      if not nThreads:
        match = re.search(
            "Running on ([0-9]+) core\(s\) for ([0-9]+)x[0-9]+ grid with " +
            "timestep 1e-[0-9]+ for ((1e\+)?[0-9]+)", line)
        if match:
          nThreads = int(match.group(1))
          gridSize = int(match.group(2))
          iterations = float(match.group(3))
      else:
        match = re.search("Finished in ([0-9]+\.[0-9]+) seconds.", line)
        if match:
          timePerIteration = float(match.group(1))/float(iterations)
          if ompThreads:
            outFile.write("1,{},{},{}".format(
                ompThreads, gridSize, timePerIteration))
          else:
            outFile.write("0,{},{},{}".format(
                nThreads, gridSize, timePerIteration))
          nThreads = None
