from tensorflow import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras import Sequential, layers
from keras.layers import Flatten, Dense
from keras import utils
from keras.optimizers import Adam
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

plt.figure(figsize=(10, 5))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap="binary")

plt.show()

model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model.summary()

x_train = x_train / 255
x_test = x_test / 255
y_train_cat = utils.to_categorical(y_train, 10)
y_test_cat = utils.to_categorical(y_test, 10)

model.compile(optimizer=Adam(learning_rate=0.01),
              loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train_cat, batch_size=60, epochs=25, validation_split=0.1)

model.evaluate(x_test, y_test_cat)

n = 1
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)
print(res)
print(np.argmax(res))
plt.imshow(x_test[n], cmap="binary")
plt.show()

pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)
print(pred.shape)
print(pred[:20])
print(y_test[:20])

mask = pred == y_test
print(mask[:10])
x_false = x_test[~mask]
y_false = y_test[~mask]
pred_false = pred[~mask]
print(x_false.shape)
for i in range(5):
  print("Значение сети: "+str(pred_false[i]))
  print("Значение из датасета: "+str(y_false[i]))
  plt.imshow(x_false[i], cmap="binary")
  plt.show()