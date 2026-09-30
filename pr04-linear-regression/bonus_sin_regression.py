"""
Практическая работа 4. Дополнительное задание.
1. Датасет - значения sin(x) с небольшим разбросом (шумом).
2. Модель, способная обучиться на таком датасете: простая ЛИНЕЙНАЯ регрессия
   с sin(x) не справится (данные нелинейны), поэтому используем полиномиальные
   признаки (PolynomialFeatures) + линейную регрессию поверх них -
   классический приём "линеаризации" нелинейной зависимости.
3. Обучение и оценка результатов.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(0)


x = np.linspace(-2 * np.pi, 2 * np.pi, 300).reshape(-1, 1)
noise = 0.15 * np.random.randn(*x.shape)
y = np.sin(x) + noise

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


degree = 7
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model.fit(x_train, y_train.ravel())


y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Степень полинома: {degree}")
print(f"MSE на тесте: {mse:.5f}")
print(f"R^2 на тесте: {r2:.5f}")


x_plot = np.linspace(-2 * np.pi, 2 * np.pi, 300).reshape(-1, 1)
y_plot = model.predict(x_plot)

plt.figure(figsize=(7, 4))
plt.scatter(x_train, y_train, s=10, color="gray", alpha=0.5, label="обучающие данные")
plt.plot(x_plot, np.sin(x_plot), color="green", label="истинная sin(x)")
plt.plot(x_plot, y_plot, color="red", linewidth=2, label=f"полином степени {degree}")
plt.legend()
plt.title("Аппроксимация sin(x) полиномиальной регрессией")
plt.tight_layout()
plt.savefig("bonus_sin_regression.png", dpi=120)
print("График сохранён в bonus_sin_regression.png")
