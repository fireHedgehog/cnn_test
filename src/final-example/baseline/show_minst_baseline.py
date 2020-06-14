import pandas as pd
import matplotlib.pyplot as plt

with open('GDCNN_MNIST.txt', 'r') as f:
    text = f.readlines()
    cols = text[0].strip().split('\t')
    data = []
    for row in text[1:]:
        data.append(list(map(float, row.strip().split('\t'))))

df = pd.DataFrame(data, columns=cols)

print(df.describe())

fig, axs = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle('SGD Optimizer for MINST Dataset', y=1, fontsize=16)

axs[0].plot(df[['accuracy']], linewidth=4)
axs[0].set_title('SGD Accuracy')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Accuracy')

axs[1].plot(df[['loss']], linewidth=4)
axs[1].set_title('SGD Loss')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Loss')

plt.show()
