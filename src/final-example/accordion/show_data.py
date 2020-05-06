import pandas as pd
import matplotlib.pyplot as plt

with open('SteadyStateGA_CIFAR10_1.txt', 'r') as f:
    text = f.readlines()
    cols = text[0].strip().split('\t')
    data = []
    for row in text[1:]:
        data.append(list(map(float, row.strip().split('\t'))))

df = pd.DataFrame(data, columns=cols)
print(df.describe())
