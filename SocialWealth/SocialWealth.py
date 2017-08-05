#! /usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
print 'where the money going'

x = [0, 100]
y = np.random.randint(1, 100, 100)
plt.figure()
plt.title("round =")
plt.xlabel("human number")
plt.ylabel("human wealth")
plt.hist(y, 100)
plt.show()
