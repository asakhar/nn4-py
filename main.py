from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Flatten, Dense
from keras import utils
from keras.optimizers import Adam
import numpy as np
from colorama import Fore, Style

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# plt.figure(figsize=(10, 5))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(x_train[i], cmap="binary")
# plt.show()

model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
print(Fore.GREEN, "Model summary:", Style.RESET_ALL)
model.summary()

x_train = x_train / 255
x_test = x_test / 255
y_train_cat = utils.to_categorical(y_train, 10)
y_test_cat = utils.to_categorical(y_test, 10)

LEARNING_RATE = 0.01
model.compile(optimizer=Adam(LEARNING_RATE),
              loss='categorical_crossentropy', metrics=['accuracy'])

print(Fore.GREEN, "Training the model", Style.RESET_ALL)
BATCH_SIZE = 60
EPOCHS = 25
model.fit(x_train, y_train_cat, batch_size=BATCH_SIZE,
          epochs=EPOCHS, validation_split=0.1)

print(Fore.GREEN, "Model prediction occuracy", Style.RESET_ALL)
model.evaluate(x_test, y_test_cat)

n = 1
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)
print(Fore.GREEN,
      f"Evaluation on sample #{n+1} (certainties)", Style.RESET_ALL)
print([f"{int(100*r)}%" for r in res[0]])
print(Fore.GREEN, "Evaluation result", Style.RESET_ALL)
print(np.argmax(res))
# plt.imshow(x_test[n], cmap="binary")
# plt.show()

pred_prob = model.predict(x_test)
pred = np.argmax(pred_prob, axis=1)
print(Fore.GREEN, "Test set evaluation result shape", Style.RESET_ALL)
print(pred.shape)
N = 20
print(Fore.GREEN, f"{N} first predicted values", Style.RESET_ALL)
print(pred[:N])
print(Fore.GREEN, f"{N} first reference values", Style.RESET_ALL)
print(y_test[:N])

mask = pred == y_test
print(Fore.GREEN, f"{N} first truthiness mask values", Style.RESET_ALL)
print(mask[:N])
x_false = x_test[~mask]
y_false = y_test[~mask]
pred_false = pred[~mask]
pred_prob_false = pred_prob[~mask]
print(Fore.GREEN, f"Shape of subset of original test set values on which nn had wrong prediction", Style.RESET_ALL)
print(x_false.shape)
N = 5
print(Fore.GREEN,
      f"{N} first wrong nn predictions compared to reference", Style.RESET_ALL)
for i in range(5):
    print(
        f"Predicted value: {pred_false[i]}, Certainty: {pred_prob_false[i][pred_false[i]]*100:.1}%")
    print(f"Reference value: {y_false[i]}")
    # plt.imshow(x_false[i], cmap="binary")
    # plt.show()
