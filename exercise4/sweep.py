#!/usr/bin/env python3
import os, sys

nCores = 8
iterations = [2**x for x in range(1, 28)]

for i in iterations:
  command = "./RunRandomWalk {} {}".format(nCores, i)
  # Warmup round
  os.system(command + " >> /dev/null")
  for i in range(3):
    os.system(command)
