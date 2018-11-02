import random

n = 1000
density = 0.01
paths = ["1000_001_A", "1000_001_B"]

for path in paths:
    with open(path, "w") as f:
        for i in range(1, n+1):
            for j in range(1, n+1):
                if (random.random() < density):
                    f.write("{} {} {}\n".format(i, j, random.uniform(-10, 10)))
