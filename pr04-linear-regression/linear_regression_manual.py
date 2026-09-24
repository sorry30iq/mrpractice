"""
Практическая работа 4, п.3.1-3.2.
Реализация линейной регрессии вручную (градиентный спуск) на сгенерированном датасете.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# === 3.1. Генерируем датасет ===
x = np.random.randn(1, 100)
a_true, b_true = 2, 1
eps = 0.1 * np.random.randn(1, 100)      # шум
y = b_true + a_true * x + eps

# разбиваем на train/test (70/30)
idx = np.arange(100)
np.random.shuffle(idx)
train_idx, test_idx = idx[:70], idx[70:]
x_train, y_train = x[0][train_idx], y[0][train_idx]
x_test, y_test = x[0][test_idx], y[0][test_idx]

# === 3.2. Обучение модели градиентным спуском ===
a = np.random.randn(1)
b = np.random.rand(1)
lr = 10e-3
epochs = 100

loss_history = []
for ep in range(epochs):
    y_pred = b + a * x_train
    error = y_pred - y_train
    loss = (error ** 2).mean()
    loss_history.append(loss)

    b_grad = 2 * error.mean()
    a_grad = 2 * (x_train * error).mean()

    a = a - lr * a_grad
    b = b - lr * b_grad

    if ep % 20 == 0 or ep == epochs - 1:
        print(f"ep: {ep:3d}  loss: {loss:8.6f}  a={a[0]:.4f}  b={b[0]:.4f}")

print(f"\nИстинные параметры: a={a_true}, b={b_true}")
print(f"Обученные параметры: a={a[0]:.4f}, b={b[0]:.4f}")

# оценка на тесте
y_test_pred = b + a * x_test
test_mse = ((y_test_pred - y_test) ** 2).mean()
print(f"MSE на тестовой выборке: {test_mse:.5f}")

# === графики ===
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(loss_history)
axes[0].set_title("Функция потерь (MSE) по эпохам")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("loss")

axes[1].scatter(x_test, y_test, color="black", label="реальные данные")
axes[1].plot(x_test, y_test_pred, color="blue", linewidth=3, label="предсказание модели")
axes[1].set_title("Линейная регрессия: тестовая выборка")
axes[1].legend()

plt.tight_layout()
plt.savefig("linear_regression_manual.png", dpi=120)
print("График сохранён в linear_regression_manual.png")
