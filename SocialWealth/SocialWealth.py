#! /usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
import math
day = range(0, 365*(65-22))
people = range(0, 100)
money = [100.0 for i in people]
LIABILITY = True  # False True
GDPCPI_EN = True
INCREASE_WEALTH = 'individual'  # average individual none
GDP = 0.02
CPI = 0.05


def wealth_switch(day):
    global INCREASE_WEALTH
    if GDPCPI_EN:
        return math.pow((CPI+1), day//365)
    else:
        INCREASE_WEALTH = 'none'
        return 1


for i in day:
    index = np.random.randint(0, 100, 100)
    for j in people:
        change = wealth_switch(i)
        if LIABILITY:
            money[j] -= change
            money[index[j]] += change
        else:
            if money[j] > 0:
                money[j] -= change
                money[index[j]] += change
        if (i % 365 == 0):
            if INCREASE_WEALTH == 'average':
                temp = [abs(i) for i in money]
                money[j] = money[j] + (sum(temp)*GDP)/100
            elif INCREASE_WEALTH == 'individual':
                money[j] = money[j]*(1+GDP)

#temp = [abs(i) for i in money]
#print(sum(temp))
# print(len(temp))
money.sort()
plt.figure()
plt.grid(True)
plt.title("total wealth = %d" % sum(money))
plt.xlabel("player")
plt.ylabel("wealth")
plt.bar(people, money, width=1, facecolor='g', alpha=0.75)
# plt.plot(people, money)
plt.savefig('money.jpg')
plt.show()
