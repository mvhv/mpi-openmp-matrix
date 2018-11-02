import random

n = 1000
density = 0.01

for i in range(1, n+1):
    for j in range(1, n+1):
        if (random.random() < density):
            print("{} {} {}".format(i, j, random.uniform(-10, 10)))