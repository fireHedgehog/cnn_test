from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Flatten, Dense
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
X_train /= 255.0
X_test /= 255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# shape able 2: The network architecture used for the MNIST classification task.
# Layer (type) Output shape Number of parameters
# conv2d_1 (Conv2D) (None, 28, 28, 40) 160
# max_pooling2d_1 (MaxPooling2D) (None, 14, 14, 40) 0
# conv2d_2 (Conv2D) (None, 14, 14, 40) 3200
# max_pooling2d_2 (MaxPooling2D) (None, 7, 7, 20) 0
# conv2d_3 (Conv2D) (None, 7, 7, 5) 400
# max_pooling2d_3 (MaxPooling2D) (None, 3, 3, 5) 0
# conv2d_4 (Conv2D) (None, 3, 3, 1) 20
# flatten_1 (Flatten) (None, 9) 0
# dense_1 (Dense) (None, 40) 360
# dense_2 (Dense) (None, 10)
model = Sequential()
model.add(Conv2D(40, (1, 1), activation='relu', use_bias=False, input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(40, (1, 1), activation='relu', use_bias=False))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(5, (1, 1), activation='relu', use_bias=False))
model.add(MaxPooling2D((1, 1)))
model.add(Conv2D(1, (2, 2), activation='relu', use_bias=False))
model.add(Flatten())
model.add(Dense(40, activation='relu', use_bias=False))
model.add(Dense(10, activation='softmax', use_bias=False))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

with open('GDCNN_MNIST.txt', 'a+') as f:
    info = history.history
    cols = list(info.keys())
    text = '\t'.join(cols) + '\n'
    for i in range(len(info['loss'])):
        text += ('\t'.join(['{:.4f}'.format(info[c][i]) for c in cols]) + '\n')
    f.write(text)
