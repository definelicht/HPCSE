#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) < 3:
  print("Usage: <weak scaling file> <strong scaling file>")
  sys.exit(1)

def integralerror(data):
  fig, ax = plt.subplots()
  cores = data[0, 0]
  samples = np.unique(data[:, 1])
  result = np.array([np.mean(data[data[:, 1] == n, 2]) for n in samples])
  errors = np.array([np.mean(np.sqrt(data[data[:, 1] == n, 3])) for n in samples])
  ax.plot(samples, 0.3*np.ones(samples.shape), "--k", linewidth=2,
          label="Analytical value")
  ax.plot(samples, result, "x", color="firebrick", markeredgewidth=2,
          markersize=10)
  ax.errorbar(samples, result, yerr=errors, color="firebrick", linestyle="none",
              linewidth=2, capsize=6, elinewidth=2, markeredgewidth=2,
              label="Estimated values")
  ax.set_xscale("log", basex=2)
  ax.set_title("Integration")
  ax.set_xlabel("Number of samples")
  ax.set_ylabel("Value")
  ax.legend(loc=1, fontsize=13)
  return fig, ax

def weakscaling(data):
  fig, ax = plt.subplots()
  cores = np.unique(data[:, 0])
  time = np.array([np.mean(data[data[:, 0] == c, 4]) for c in cores])
  speedup = time[0] / time
  ideal = np.array([1, 2, 4, 4])
  ax.plot(cores, speedup, "+-", linewidth=2, markersize=20, markeredgewidth=3,
          label="Measured speedup", color="navy")
  ax.plot(cores, ideal, "--k", linewidth=2,
          label="Linear speedup with number of cores")
  ax.set_xlim([0.8, 8.2])
  ax.set_xticks([1, 2, 4, 8])
  ax.set_ylim([0.8, 6])
  ax.set_title("Hyperthreading speedup")
  ax.set_xlabel("Number of threads")
  ax.set_ylabel("Speedup")
  ax.legend(loc=2, fontsize=16)
  return fig, ax

def strongscaling(data):
  fig, axs = plt.subplots(1, 2, figsize=(16, 6))
  cores = data[0, 0]
  samples = np.unique(data[:, 1])
  time = np.array([np.mean(data[data[:, 1] == n, 4]) for n in samples])
  print(time)
  timePerSample = time/samples
  timePerSamplePerAverage = timePerSample[19:]/np.mean(timePerSample[19:])
  axs[0].plot(samples, timePerSample, "+-", color="firebrick", linewidth=2,
              markersize=20, markeredgewidth=3)
  axs[0].plot([samples[19], samples[19]], axs[0].get_ylim(), "--k", linewidth=2)
  axs[0].set_xscale("log", basex=2)
  axs[0].set_yscale("log", basey=2)
  axs[0].set_title("Complete sweep")
  axs[0].set_xlabel("Number of samples")
  axs[0].set_ylabel("Time per sample")
  axs[1].plot(samples[19:], timePerSamplePerAverage, "+-", linewidth=2,
              markersize=19, markeredgewidth=3, color="firebrick");
  axs[1].set_xscale("log", basex=2)
  axs[1].set_title("Computationally bound region")
  axs[1].set_xlabel("Number of samples")
  axs[1].set_ylabel("Time per sample/average time per sample")
  return fig, axs

plt.rcParams.update({'font.size': 18})
weakdata = np.loadtxt(sys.argv[1], delimiter=",")
strongdata = np.loadtxt(sys.argv[2], delimiter=",")

figVal, axVal = integralerror(strongdata)
figWeak, axWeak = weakscaling(weakdata)
figStrong, axStrong = strongscaling(strongdata)
if len(sys.argv) < 4:
  figVal.show()
  figWeak.show()
  figStrong.show()
  input("Press any key to terminate...")
else:
  figVal.savefig(sys.argv[3], bbox_inches="tight")
  if len(sys.argv) >= 5:
    figWeak.savefig(sys.argv[4], bbox_inches="tight")
  if len(sys.argv) >= 6:
    figStrong.savefig(sys.argv[5], bbox_inches="tight")

