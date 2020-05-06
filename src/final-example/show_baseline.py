import pandas as pd
import matplotlib.pyplot as plt

with open('./GDCNN_CIFAR10.txt', 'r') as f:
    text = f.readlines()
    cols = text[0].strip().split('\t')
    data = []
    for row in text[1:]:
        data.append(list(map(float, row.strip().split('\t'))))

df = pd.DataFrame(data, columns=cols)

plt.figure(figsize=(20, 10))
plt.title("CIFAR10 with ADAM optimizer")

plt.subplot(121)
plt.plot(df[['accuracy']])
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.suptitle('Accuracy')

plt.subplot(122)
plt.plot(df[['loss']])
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.suptitle('Loss')

plt.show()
