#!/usr/bin/env python3
import os, sys

grids = [(128, 1e-5, 250), (256, 1e-6, 6.25), (1024, 1e-8, 0.002)]
if len(sys.argv) < 2:
  cores = [1, 6, 12, 18, 24, 30, 36, 42, 48]
else:
  cores = [int(x) for x in sys.argv[1:]]

for (dim, dt, end) in grids:
  for c in cores:
    outputStr = str(dim) + "_" + str(c) + ".txt";
    command = "mpirun -n {} exercise6/RunDiffusionMPI 1 {} {} {} {}".format(
        c, dim, dt, outputStr, end)
    # Warmup round
    os.system(command + " >> /dev/null")
    for i in range(3):
      os.system(command)
