import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPool2D, Conv2D
from keras.utils import np_utils

#Load training and data set:
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_val, y_val = X_train[50000:60000,:], y_train[50000:60000]
X_train, y_train = X_train[:50000,:], y_train[:50000]
print(X_train.shape)

#Reshape data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

#One hot encoding Y
Y_train = np_utils.to_categorical(y_train,10)
Y_val = np_utils.to_categorical(y_val,10)
Y_test = np_utils.to_categorical(y_test,10)
print("Dữ liệu y ban đầu ", y_train[0])
print("Dữ liệu y sau one-hot encoding ", Y_train[0])

#model definition
model = Sequential()

#
model.add(Conv2D(32,(3,3), activation = 'sigmoid', input_shape = (28,28,1)))

# Add conv layer
model.add(Conv2D(32,(3,3), activation = 'sigmoid'))

#Add pooling layer
model.add(MaxPool2D(pool_size = (2,2)))

#Add flatten layer
model.add(Flatten())

#Add FCN 
model.add(Dense(128, activation = 'sigmoid'))

#Add to output layer
model.add(Dense(10, activation = 'softmax'))

#Compile model
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', 
              metrics = ['accuracy'])

#Train model
H = model.fit(X_train, Y_train, validation_data= (X_val, Y_val),
              batch_size = 32, epochs = 3, verbose= 1)

#Draw the graph
fig = plt.figure()
numOfEpoch = 3
plt.plot(np.arange(0, numOfEpoch), H.history['loss'], label='training loss')
plt.plot(np.arange(0, numOfEpoch), H.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, numOfEpoch), H.history['accuracy'], label='accuracy')
plt.plot(np.arange(0, numOfEpoch), H.history['val_accuracy'], label='validation accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy')
plt.legend()

#Evaluate the model
score = model.evaluate(X_test, Y_test, verbose= 1)
print(score)

#Predict the image
plt.imshow(X_test[0].reshape(28,28), cmap='gray')
y_predict = model.predict(X_test[0].reshape(1,28,28,1))
print('Giá trị dự đoán: ', np.argmax(y_predict))





