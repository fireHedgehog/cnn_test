import pandas as pd
import matplotlib.pyplot as plt

with open('adam-cifar-10.txt', 'r') as f:
    text = f.readlines()
    cols = text[0].strip().split('\t')
    data = []
    for row in text[1:]:
        data.append(list(map(float, row.strip().split('\t'))))

df = pd.DataFrame(data, columns=cols)

with open('SGDCNN_CIFAR10.txt', 'r') as f:
    text = f.readlines()
    cols = text[0].strip().split('\t')
    data = []
    for row in text[1:]:
        data.append(list(map(float, row.strip().split('\t'))))

df2 = pd.DataFrame(data, columns=cols)

fig, axs = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle('SGD and ADAM Optimizer for CIFAR-10 ', y=1, fontsize=16)

axs[0].plot(df2[['accuracy']], linewidth=4)
axs[0].set_title('SGD Accuracy')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Accuracy')

axs[1].plot(df2[['loss']], linewidth=4)
axs[1].set_title('SGD Loss')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Loss')

axs[2].plot(df[['accuracy']], linewidth=4)
axs[2].set_title('ADAM Accuracy')
axs[2].set_xlabel('Epochs')
axs[2].set_ylabel('Accuracy')

axs[3].plot(df[['loss']], linewidth=4)
axs[3].set_title('ADAM Loss')
axs[3].set_xlabel('Epochs')
axs[3].set_ylabel('Loss')

plt.show()
