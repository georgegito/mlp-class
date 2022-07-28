import numpy as np

def data_summary(arr):
  shape = np.shape(arr)
  min = np.amin(arr)
  max = np.amax(arr)
  range = np.ptp(arr)
  variance = np.var(arr)
  sd = np.std(arr)
  print("Shape =", shape)
  print("Minimum =", min)
  print("Maximum =", max)
  print("Range =", range)
  print("Variance =", variance)
  print("Standard Deviation =", sd)
  print()
