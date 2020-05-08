import pandas as pd
import matplotlib.pyplot as plt


def get_data_by_file_name(name):
    with open(name, 'r') as f:
        text = f.readlines()
        cols = text[0].strip().split('\t')
        data = []
        for row in text[1:]:
            data.append(list(map(float, row.strip().split('\t'))))

    print(data, cols)
    return pd.DataFrame(data, columns=cols)


df = get_data_by_file_name('ElitismGA_CIFAR10.txt')
df1 = get_data_by_file_name('../population-data/01-07-100-300.txt')
df2 = get_data_by_file_name('GenerationalGA_CIFAR10.txt')

print(df.describe())
print(df1.describe())
print(df2.describe())

fig, axs = plt.subplots(1, 2, figsize=(20, 5))
fig.suptitle('Selection Function Comparision for Accordion Encoding ( CIFAR-10 )', y=1, fontsize=16)

axs[0].plot(df[['best_acc']], marker='_', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=3)
axs[0].plot(df1[['accuracy']], marker='.', markerfacecolor='red', markersize=12, color='orange', linewidth=3)
axs[0].plot(df2[['best_acc']], marker='+', markerfacecolor='black', markersize=12, color='green', linewidth=3)

axs[0].legend(['Elitism', 'Steady-state', 'Generational'], loc='upper left')
axs[0].set_title('Accuracy')
axs[0].set_xlabel('Generations')
axs[0].set_ylabel('Accuracy')

axs[1].plot(df[['best_loss']], marker='_', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=3)
axs[1].plot(df1[['loss']], marker='.', markerfacecolor='red', markersize=12, color='orange', linewidth=3)
axs[1].plot(df2[['best_loss']], marker='+', markerfacecolor='black', markersize=12, color='green', linewidth=3)
axs[1].legend(['Elitism', 'Steady-state', 'Generational'], loc='upper left')
axs[1].set_title('Loss')
axs[1].set_xlabel('Generations')
axs[1].set_ylabel('Loss')

# axs[2].plot(df[['accuracy']], linewidth=4)
# axs[2].set_title('ADAM Accuracy')
# axs[2].set_xlabel('Epochs')
# axs[2].set_ylabel('Accuracy')
#
# axs[3].plot(df[['loss']], linewidth=4)
# axs[3].set_title('ADAM Loss')
# axs[3].set_xlabel('Epochs')
# axs[3].set_ylabel('Loss')

plt.show()
