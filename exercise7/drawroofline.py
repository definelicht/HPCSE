#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import sys

nSamples = 10000
peakFlops = 576
peakMem = 59.7
diffusionFlop = 6/24
diffusionPeak = min(peakFlops, diffusionFlop*peakMem)
peakAchieved = 44.4
print("Diffusion peak performance: {} GFLOP/s".format(diffusionPeak))
flopsPerByte = np.logspace(-4, 8, nSamples, base=2)
peak = np.minimum(np.full(nSamples, peakFlops), flopsPerByte*peakMem)
peakNoSimd = np.minimum(np.full(nSamples, peakFlops/8), flopsPerByte*peakMem)
plt.rcParams.update({"font.size": 18})
fig, ax = plt.subplots()
ax.plot(flopsPerByte, peak, "-r", linewidth=2, label="Peak performance")
ax.plot(flopsPerByte, peakNoSimd, "-", linewidth=2, label="No SIMD",
        color="cyan")
ax.plot(np.array([diffusionFlop, diffusionFlop]),
        np.array([ax.get_ylim()[0], diffusionPeak]),
        "--k", linewidth=2, label="Diffusion theoretical peak")
ax.plot(np.array([flopsPerByte[0], flopsPerByte[-1]]),
        np.array([peakAchieved, peakAchieved]),
        "--g", linewidth=2, label="Peak throughput achieved")
ax.set_xscale("log", basex=2)
ax.set_yscale("log")
ax.set_xlabel("FLOP/B")
ax.set_ylabel("GFLOP/s")
ax.set_title("Roofline model for 24-core Ivy Bridge Euler node", fontsize=17)
ax.legend(loc=4, fontsize=17)
if len(sys.argv) > 1:
  fig.savefig(sys.argv[1], bbox_inches="tight")
else:
  fig.show()
  input("Press enter to exit...")
