#!/usr/bin/env python3
import os, sys

grids = [(128, 1e-5, 10), (256, 1e-6, 0.5), (1024, 1e-8, 0.0002)]
cores = [1, 6, 12]

for (dim, dt, end) in grids:
  for c in cores:
      outputStr = str(dim) + "_" + str(c) + ".txt";
      command = "./RunDiffusion {} {} {} {} {}".format(
          c, dim, dt, outputStr, end)
      # Warmup round
      os.system(command + " >> /dev/null")
      for i in range(3):
        os.system(command)
