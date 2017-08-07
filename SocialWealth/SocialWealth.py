#! /usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
round = range(0, 365*(65-22))
people = range(0, 100)
money = [100 for i in people]
LIABILITY = False # False True
INCREASE_WEALTH = 'absolute' #absolute precent none
#SUB_ALLOWANCE = True

'''
for i in people:
    test_rand = np.random.randint(1, 100, 100)
    print(test_rand)
'''
for i in round:
    index = np.random.randint(0, 100, 100)
    for j in people:
        if LIABILITY:
            money[j] -= 1
            money[index[j]] += 1
        else:
            if money[j] > 0:
                money[j] -= 1
                money[index[j]] += 1
        if INCREASE_WEALTH == 'absolute':
            if i % 365 == 0:
                money[j] += sum(money)*0.02/100
        elif INCREASE_WEALTH == 'precent':
            if i % 365 == 0:
                money[j] += money[j]*(0.02)

money.sort()
plt.figure()
plt.grid(True)
#plt.title("round = %d" % round)
plt.xlabel("player")
plt.ylabel("wealth")
plt.bar(people, money, width=1, facecolor='b', alpha=0.75)
#print(money)
plt.savefig('money.jpg')
plt.show()
#'''
