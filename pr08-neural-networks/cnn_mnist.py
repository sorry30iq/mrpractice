"""
Практическая работа 8, раздел 4.
Свёрточная нейросетевая модель (CNN) для классификации MNIST -
для сравнения качества с полносвязной моделью из dense_mnist.py.
"""

import keras
from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical

# свёрточные слои
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))

# полносвязные слои классификации
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

model.summary()

# загружаем датасет и обучаем
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1)).astype("float32") / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype("float32") / 255

train_labels_cat = to_categorical(train_labels)
test_labels_cat = to_categorical(test_labels)

model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train_images, train_labels_cat, epochs=5, batch_size=64)

#оцениваем и сравниваем с Dense-моделью
test_loss, test_acc = model.evaluate(test_images, test_labels_cat)
print(f"\nCNN test_acc: {test_acc:.4f}")
print(f"CNN test_loss: {test_loss:.4f}")
print("\nСравнение: CNN обычно даёт точность выше, чем полносвязная сеть "
      "(в методичке: 0.9916 против 0.9785), т.к. свёрточные слои учитывают "
      "локальную пространственную структуру изображения (соседние пиксели), "
      "а не работают с ним как с плоским вектором признаков.")

model.save("cnn_mnist_model.keras")
print("Модель сохранена в cnn_mnist_model.keras")
