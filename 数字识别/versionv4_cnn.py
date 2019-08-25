from keras.datasets import mnist
import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau


# use Keras to import pre-shuffled MNIST database
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.vstack((X_train, X_test))    #  按行对齐
y_train = np.concatenate([y_train, y_test])
X_train = X_train.reshape(-1, 28, 28, 1)
print(X_train.shape, y_train.shape)

train = pd.read_csv('data/train.csv').values
y_val = train[:,0].astype('int32')   # 第一列为label
X_val = train[:,1:].astype('float32')   # 第二列为piex
X_val = X_val.reshape(-1, 28, 28, 1)
print(X_val.shape, y_val.shape)

X_test = pd.read_csv('data/test.csv').values.astype('float32')
X_test = X_test.reshape(-1, 28, 28, 1)

# 标准化
X_train = X_train.astype('float32')/255
X_val = X_val.astype('float32')/255
X_test = X_test.astype('float32')/255

# print first ten (integer-valued) training labels
# print('Integer-valued labels:')
# print(y_train[:10])

# one-hot encode the labels
y_train = np_utils.to_categorical(y_train, 10)
y_val = np_utils.to_categorical(y_val, 10)

# print first ten (one-hot) training labels
# print('One-hot labels:')
# print(y_train[:10])

# define the model   定义模型
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=192, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=192, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2, padding='same'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))
# summarize the model
model.summary()

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model
#checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5',
#                               verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, verbose=1,
                              patience=2, min_lr=0.00000001)
hist = model.fit(X_train, y_train, batch_size=100, epochs=25,
          validation_data=(X_val, y_val), callbacks=[reduce_lr],
          verbose=1, shuffle=True)

testY = model.predict_classes(X_test, verbose=2)
sub = pd.read_csv('data/sample_submission.csv')
sub['Label'] = testY
sub.to_csv('data/version2/Result_keras.csv', index=False)
