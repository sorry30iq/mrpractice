"""
Практическая работа 8, раздел 3.
Полносвязная (Dense) нейросетевая модель для классификации цифр MNIST.
"""

import keras
from keras.datasets import mnist
from keras import models, layers
from keras.utils import to_categorical

# загружаем датасет
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print("train_images.shape:", train_images.shape)  # (60000, 28, 28)
print("количество меток:", len(train_labels))

# создаём модель
model = models.Sequential()
model.add(layers.Dense(512, activation="relu", input_shape=(28 * 28,)))
model.add(layers.Dense(10, activation="softmax"))
model.summary()


# бинарная классификация  sigmoid + binary_crossentropy
# многоклассовая однозначная классификация   softmax + categorical_crossentropy
model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

train_images = train_images.reshape((60000, 28 * 28)).astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255

train_labels_cat = to_categorical(train_labels)
test_labels_cat = to_categorical(test_labels)

#обучаем и оцениваем
history = model.fit(train_images, train_labels_cat, epochs=5, batch_size=128)

test_loss, test_acc = model.evaluate(test_images, test_labels_cat)
print(f"\ntest_acc: {test_acc:.4f}")
print(f"test_loss: {test_loss:.4f}")

model.save("dense_mnist_model.keras")
print("Модель сохранена в dense_mnist_model.keras")
