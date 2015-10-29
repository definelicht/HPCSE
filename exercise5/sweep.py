#!/usr/bin/env python3
import os, sys

nCores = [24]
iterations = [i*100 for i in range(10)]

for c in nCores:
  for i in iterations:
    command = "./RunRigidDisks {} 100 115 1 0.5 100 {} 512 histogram.txt".format(c, i)
    for i in range(3):
      os.system(command)
