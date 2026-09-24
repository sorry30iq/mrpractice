"""
Практическая работа 7. Банковский скоринг (бинарная классификация).

Датасет: bank-additional-full.csv (UCI Bank Marketing dataset)
Скачать: https://archive.ics.uci.edu/dataset/222/bank+marketing
Положи файл bank-additional-full.csv в эту же папку перед запуском.

Если файла нет - скрипт сгенерирует синтетический датасет с похожей
структурой, чтобы пайплайн можно было проверить без интернета
(результаты на синтетике будут другими, для отчёта нужен настоящий датасет!).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc,
)

DATA_PATH = "bank-additional-full.csv"


def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, header=0, sep=";")
        print(f"Загружен настоящий датасет: {df.shape}")
        return df

    print(f"[!] Файл {DATA_PATH} не найден - генерирую синтетический датасет "
          f"той же структуры для проверки пайплайна (см. docstring).")
    rng = np.random.default_rng(0)
    n = 5000
    df = pd.DataFrame({
        "job": rng.choice(["admin.", "technician", "services", "blue-collar", "retired"], n),
        "marital": rng.choice(["married", "single", "divorced"], n),
        "default": rng.choice(["no", "yes", "unknown"], n, p=[0.85, 0.05, 0.10]),
        "housing": rng.choice(["no", "yes"], n),
        "loan": rng.choice(["no", "yes"], n, p=[0.85, 0.15]),
        "poutcome": rng.choice(["nonexistent", "failure", "success"], n, p=[0.8, 0.15, 0.05]),
    })
    # искусственная зависимость целевой переменной, чтобы модель могла чему-то учиться
    score = (
        (df["poutcome"] == "success").astype(int) * 2
        + (df["default"] == "no").astype(int)
        - (df["loan"] == "yes").astype(int)
        + rng.normal(0, 1, n)
    )
    df["y"] = np.where(score > 1.0, "yes", "no")
    return df


# === 2.3-2.5: загрузка и очистка ===
df = load_data()
df = df.dropna()

# === 2.7: оставляем только категориальные признаки + целевую переменную
# (как в методичке - job, marital, default, housing, loan, poutcome, y)
cols_needed = ["job", "marital", "default", "housing", "loan", "poutcome", "y"]
df = df[[c for c in cols_needed if c in df.columns]]

# === 2.8: кодируем категориальные признаки (one-hot) ===
cat_cols = ["job", "marital", "default", "housing", "loan", "poutcome"]
cat_cols = [c for c in cat_cols if c in df.columns]
data = pd.get_dummies(df, columns=cat_cols)

# === 2.9: кодируем метки ===
# .astype(int) нужен явно - в новых версиях pandas (3.x) текстовые колонки
# имеют собственный строковый dtype, и после replace() тип может не привестись
# к числовому автоматически, из-за чего sklearn не распознаёт задачу классификации.
data["y"] = data["y"].replace({"yes": 1, "no": 0}).astype(int)

# === 2.10: массивы X и Y ===
X = data.drop(columns=["y"])
Y = data["y"]

# === 2.11: обучение логистической регрессии ===
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

classifier = LogisticRegression(solver="lbfgs", random_state=0, max_iter=1000)
classifier.fit(X_train, Y_train)

# === 2.13: качество модели ===
acc = classifier.score(X_test, Y_test)
print(f"\nLogisticRegression accuracy: {acc:.4f}")

# === 3. Матрица ошибок и метрики ===
y_pred_lr = classifier.predict(X_test)
print("\nМатрица ошибок (LogisticRegression):")
print(confusion_matrix(Y_test, y_pred_lr))
print("\nОтчёт по классификации (LogisticRegression):")
print(classification_report(Y_test, y_pred_lr, zero_division=0))

# === 3.4: сравнение с другой моделью - Gradient Boosting ===
clf_gbc = GradientBoostingClassifier(n_estimators=100, random_state=42)
clf_gbc.fit(X_train, Y_train)
y_pred_gbc = clf_gbc.predict(X_test)

print("\n=== Gradient Boosting ===")
print(f"Accuracy: {accuracy_score(Y_test, y_pred_gbc):.4f}")
print(classification_report(Y_test, y_pred_gbc, zero_division=0))

# матрица ошибок GBC
cm = confusion_matrix(Y_test, y_pred_gbc)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="g", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True Value")
plt.title("Confusion matrix - Gradient Boosting")
plt.tight_layout()
plt.savefig("confusion_matrix_gbc.png", dpi=120)

# === ROC-AUC для обеих моделей ===
plt.figure(figsize=(5, 5))
for name, model in [("LogisticRegression", classifier), ("GradientBoosting", clf_gbc)]:
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(Y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], "r--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=120)
print("\nГрафики сохранены: confusion_matrix_gbc.png, roc_curve.png")
