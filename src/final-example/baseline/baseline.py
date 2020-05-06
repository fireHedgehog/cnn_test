from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

model = Sequential()
model.add(Conv2D(6, (1, 1), activation='relu', use_bias=False, input_shape=(32, 32, 3)))
model.add(AveragePooling2D((2, 2)))
model.add(Conv2D(16, (1, 1), activation='relu', use_bias=False))
model.add(AveragePooling2D((2, 2)))
model.add(Conv2D(120, (1, 1), activation='relu', use_bias=False))
model.add(AveragePooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(120, activation='relu', use_bias=False))
model.add(Dense(84, activation='relu', use_bias=False))
model.add(Dense(10, activation='softmax', use_bias=False))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=600, validation_data=(X_test, y_test))

with open('SGDCNN_CIFAR10.txt', 'a+') as f:
    info = history.history
    cols = list(info.keys())
    text = '\t'.join(cols) + '\n'
    for i in range(len(info['loss'])):
        text += ('\t'.join(['{:.4f}'.format(info[c][i]) for c in cols]) + '\n')
    f.write(text)
