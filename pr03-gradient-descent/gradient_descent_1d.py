"""
Практическая работа 3, п.4.2.
Алгоритм градиентного спуска (АГС) для функции:
    f(x) = (x - a)^2 + sin(3*(x - a)) + b

Производная (аналитически):
    f'(x) = 2*(x - a) + 3*cos(3*(x - a))
"""

import numpy as np
import matplotlib.pyplot as plt

# параметры функции (те же, что исследовались в DESMOS, п.3.1)
a = 2.4
b = 4.8


def objective(x):
    return (x - a) ** 2 + np.sin(3 * (x - a)) + b


def derivative(x):
    return 2 * (x - a) + 3 * np.cos(3 * (x - a))


def gradient_descent(x0, lr, n_iter):
    x = x0
    history = [x]
    for i in range(n_iter):
        grad = derivative(x)
        x = x - lr * grad          # шаг АГС: идём против направления градиента
        history.append(x)
        print(f"iter {i:3d}: x = {x:8.5f}  f(x) = {objective(x):8.5f}")
    return x, history


if __name__ == "__main__":
    x0 = 5.0      # начальная точка
    lr = 0.1      # шаг обучения (learning rate)
    n_iter = 30   # число итераций

    x_min, history = gradient_descent(x0, lr, n_iter)
    print(f"\nНайденный минимум: x = {x_min:.5f}, f(x) = {objective(x_min):.5f}")

    # график функции и траектории спуска
    xs = np.linspace(-2, 10, 400)
    plt.figure(figsize=(7, 4))
    plt.plot(xs, objective(xs), label="f(x)")
    plt.plot(history, [objective(x) for x in history], "ro-", markersize=3, label="шаги АГС")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Градиентный спуск для f(x) = (x-a)^2 + sin(3(x-a)) + b")
    plt.legend()
    plt.tight_layout()
    plt.savefig("gradient_descent_1d.png", dpi=120)
    print("График сохранён в gradient_descent_1d.png")
