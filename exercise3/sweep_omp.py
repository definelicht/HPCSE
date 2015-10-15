#!/usr/bin/env python3
import os, sys

grids = [(128, 1e-5, 10), (256, 1e-5, 10), (512, 1e-6, 0.5), (1024, 1e-7, 0.01), (2048, 1e-7, 0.003), (4096, 1e-8, 0.0001)]

for (dim, dt, end) in grids:
  outputStr = str(dim) + "_omp.txt";
  command = "./RunDiffusion 1 1 {} {} {} {}".format(
      dim, dt, outputStr, end)
  # Warmup round
  os.system(command + " >> /dev/null")
  for i in range(3):
    os.system(command)
