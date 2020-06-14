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


df = get_data_by_file_name('Accuracy_03_08_20.txt')
print(df.describe())

plt.plot(df[['Accuracy']], marker='+', markerfacecolor='skyblue', markersize=5, linewidth=2)
plt.legend(['Mutation Rate=0.3, crossover=0.8, population=10'], loc='upper left')
plt.title('Accuracy')
plt.xlabel('Iterations (Networks)')
plt.ylabel('Accuracy')

plt.show()
