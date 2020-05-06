import pandas as pd
import matplotlib.pyplot as plt

with open('GDCNN_CIFAR10.txt', 'r') as f:
    text = f.readlines()
    cols = text[0].strip().split('\t')
    data = []
    for row in text[1:]:
        data.append(list(map(float, row.strip().split('\t'))))

df = pd.DataFrame(data, columns=cols)

with open('GDCNN_MNIST.txt', 'r') as f:
    text = f.readlines()
    cols = text[0].strip().split('\t')
    data = []
    for row in text[1:]:
        data.append(list(map(float, row.strip().split('\t'))))

df2 = pd.DataFrame(data, columns=cols)

fig, axs = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle('ADAM Optimizer for CIFAR-10 and MINST ', y=1, fontsize=16)

axs[0].plot(df[['accuracy']], linewidth=4)
axs[0].set_title('Accuracy Plot')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Accuracy')

axs[1].plot(df[['loss']], linewidth=4)
axs[1].set_title('Loss Value Plot')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Loss')

axs[2].plot(df2[['accuracy']], linewidth=4)
axs[2].set_title('Accuracy Plot')
axs[2].set_xlabel('Epochs')
axs[2].set_ylabel('Accuracy')

axs[3].plot(df2[['loss']], linewidth=4)
axs[3].set_title('Loss Value Plot')
axs[3].set_xlabel('Epochs')
axs[3].set_ylabel('Loss')

plt.show()
