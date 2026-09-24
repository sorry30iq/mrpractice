"""
Практическая работа 4, п.3.3.
Та же задача регрессии, но с использованием библиотеки sklearn -
для сравнения с результатом ручной реализации градиентного спуска.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(42)

# тот же датасет, что и в linear_regression_manual.py
x = np.random.randn(1, 100)
a_true, b_true = 2, 1
eps = 0.1 * np.random.randn(1, 100)
y = b_true + a_true * x + eps

idx = np.arange(100)
np.random.shuffle(idx)
train_idx, test_idx = idx[:70], idx[70:]
x_train, y_train = x[0][train_idx], y[0][train_idx]
x_test, y_test = x[0][test_idx], y[0][test_idx]

# sklearn ожидает 2D-массивы формы (n_samples, n_features)
xtr = x_train.reshape(70, 1)
ytr = y_train.reshape(70, 1)
xts = x_test.reshape(30, 1)
yts = y_test.reshape(30, 1)

linr = LinearRegression()
linr.fit(xtr, ytr)

print(f"Обученные параметры (sklearn): b={linr.intercept_[0]:.4f}  a={linr.coef_[0][0]:.4f}")
print(f"Истинные параметры: a={a_true}, b={b_true}")

score = linr.score(xts, yts)
print(f"R^2 на тестовой выборке: {score:.4f}")

y_test_pred = linr.predict(xts)

plt.figure(figsize=(5, 4))
plt.scatter(x_test, y_test, color="black", label="реальные данные")
plt.plot(x_test, y_test_pred, color="blue", linewidth=3, label="sklearn LinearRegression")
plt.legend()
plt.title("Линейная регрессия (sklearn)")
plt.tight_layout()
plt.savefig("linear_regression_sklearn.png", dpi=120)
print("График сохранён в linear_regression_sklearn.png")

print("\nСравнение: коэффициенты ручной реализации и sklearn должны совпадать "
      "с точностью до 2-3 знака (т.к. sklearn решает задачу аналитически методом "
      "наименьших квадратов, а не итеративным градиентным спуском).")
