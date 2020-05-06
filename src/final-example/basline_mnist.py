from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
X_train /= 255.0
X_test /= 255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

model = Sequential()
model.add(Conv2D(40, (1, 1), activation='relu', use_bias=False, input_shape=(28, 28, 1)))
model.add(AveragePooling2D((2, 2)))
model.add(Conv2D(40, (1, 1), activation='relu', use_bias=False))
model.add(AveragePooling2D((2, 2)))
model.add(Conv2D(5, (1, 1), activation='relu', use_bias=False))
model.add(AveragePooling2D((2, 2)))
model.add(Conv2D(1, (1, 1), activation='relu', use_bias=False))
model.add(Flatten())
model.add(Dense(40, activation='relu', use_bias=False))
model.add(Dense(10, activation='softmax', use_bias=False))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=600, validation_data=(X_test, y_test))

with open('GDCNN_MNIST.txt', 'a+') as f:
    info = history.history
    cols = list(info.keys())
    text = '\t'.join(cols) + '\n'
    for i in range(len(info['loss'])):
        text += ('\t'.join(['{:.4f}'.format(info[c][i]) for c in cols]) + '\n')
    f.write(text)
