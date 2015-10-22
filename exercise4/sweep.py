#!/usr/bin/env python3
import os, sys

nCores = [1, 2, 4, 8]
# iterations = [2**x for x in range(1, 28)]
iterations = [2**24]

for c in nCores:
  for i in iterations:
    command = "./RunRandomWalk {} {}".format(c, i)
    # Warmup round
    os.system(command + " >> /dev/null")
    for i in range(3):
      os.system(command)
