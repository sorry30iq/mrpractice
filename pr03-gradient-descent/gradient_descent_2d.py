"""
Практическая работа 3, п.4.3.
Алгоритм градиентного спуска для функции двух переменных (Six-Hump Camel Function):

    f(x, y) = 2*x^2 - 1.05*x^4 + x^6/6 + x*y + y^2

Область определения: -3 <= x, y <= 3.

Частные производные:
    df/dx = 4*x - 4.2*x^3 + x^5 + y
    df/dy = x + 2*y
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (нужен для 3D-проекции)


def objective(x, y):
    return 2 * x**2 - 1.05 * x**4 + x**6 / 6 + x * y + y**2


def gradient(x, y):
    df_dx = 4 * x - 4.2 * x**3 + x**5 + y
    df_dy = x + 2 * y
    return np.array([df_dx, df_dy])


def gradient_descent_2d(point0, lr, n_iter, bounds=(-3, 3)):
    point = np.array(point0, dtype=float)
    history = [point.copy()]

    for i in range(n_iter):
        grad = gradient(*point)
        point = point - lr * grad
        # ограничиваем область определения, как указано в задании
        point = np.clip(point, bounds[0], bounds[1])
        history.append(point.copy())
        if i % 5 == 0 or i == n_iter - 1:
            print(f"iter {i:3d}: x={point[0]:8.5f} y={point[1]:8.5f} "
                  f"f={objective(*point):8.5f}")

    return point, np.array(history)


if __name__ == "__main__":
    # несколько стартовых точек - функция невыпуклая, есть несколько локальных минимумов,
    # поэтому результат может зависеть от начальной точки (это стоит отметить в отчёте)
    starts = [(-2.5, 2.0), (1.5, -1.0), (0.5, 0.5)]

    best_point, best_value = None, np.inf
    for s in starts:
        point, history = gradient_descent_2d(s, lr=0.01, n_iter=200)
        value = objective(*point)
        print(f"Старт {s} -> минимум в {point}, f = {value:.5f}\n")
        if value < best_value:
            best_value, best_point = value, point

    print(f"Лучший найденный минимум: x={best_point[0]:.5f}, y={best_point[1]:.5f}, "
          f"f={best_value:.5f}")

    # визуализация поверхности функции
    xs = np.linspace(-3, 3, 100)
    ys = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(xs, ys)
    Z = objective(X, Y)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    ax.set_title("f(x,y) = 2x^2 - 1.05x^4 + x^6/6 + xy + y^2")
    plt.tight_layout()
    plt.savefig("gradient_descent_2d.png", dpi=120)
    print("График сохранён в gradient_descent_2d.png")
