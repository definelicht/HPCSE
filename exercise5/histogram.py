import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) < 3:
  print("Please specify input file and d0.")
  sys.exit(1)

values = np.loadtxt(sys.argv[1], delimiter=",")
d0 = float(sys.argv[2])
binSize = (2 - d0*d0)/values.size + d0*d0
bins = np.array([i*binSize*1.5 for i in range(values.size)])
fig, ax = plt.subplots()
ax.bar(bins, values)
plt.show()
