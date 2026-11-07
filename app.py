"""Локальная демонстрация готового пайплайна Wine Quality."""

import json
from pathlib import Path

import joblib
import matplotlib

# Streamlit renders charts in a worker thread without a desktop GUI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics


APP_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / "model.joblib")


@st.cache_data
def load_test_data():
    return pd.read_csv(APP_DIR / "test.csv")


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


def decisions(probabilities, threshold):
    return (np.asarray(probabilities) >= threshold).astype(int)


def validate_batch(df, feature_names):
    """Проверить CSV и вернуть признаки в порядке обучения модели."""
    if df.empty:
        raise ValueError("CSV не содержит строк с данными.")
    if not df.columns.is_unique:
        raise ValueError("Названия столбцов должны быть уникальными.")
    forbidden = {"target", "probability", "prediction"}.intersection(df.columns)
    if forbidden:
        raise ValueError("Удалите служебные столбцы: " + ", ".join(sorted(forbidden)))
    missing = set(feature_names).difference(df.columns)
    if missing:
        raise ValueError("Не хватает столбцов: " + ", ".join(sorted(missing)))
    try:
        X = df.loc[:, list(feature_names)].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("Все признаки должны содержать числа.") from exc
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Признаки не должны содержать пропуски, NaN или бесконечность.")
    if not X["is_red"].isin([0, 1]).all():
        raise ValueError("is_red должен быть равен 0 (белое) или 1 (красное).")
    return X


def predict_batch(model, df, threshold):
    X = validate_batch(df, model.feature_names_in_)
    result = df.copy()
    result["probability"] = model.predict_proba(X)[:, 1]
    result["prediction"] = decisions(result["probability"], threshold)
    return result


def show_single(model, X, threshold):
    st.subheader("Оценка одного вина")
    st.caption("Диапазоны и начальные значения взяты из тестовой выборки.")
    values = {}
    columns = st.columns(3)
    for index, name in enumerate(X.columns):
        with columns[index % 3]:
            if name == "is_red":
                values[name] = int(st.radio("Тип вина", ["Белое", "Красное"]) == "Красное")
            else:
                values[name] = st.number_input(
                    name,
                    min_value=float(X[name].min()),
                    max_value=float(X[name].max()),
                    value=float(X[name].median()),
                    step=0.00001 if name == "density" else 0.001,
                    format="%.5f" if name == "density" else "%.4f",
                )
    probability = float(model.predict_proba(pd.DataFrame([values], columns=X.columns))[0, 1])
    st.metric("Вероятность хорошего вина", f"{probability:.2%}")
    st.write("Решение: " + ("рекомендовать" if probability >= threshold else "не рекомендовать"))


def show_metrics(y, probabilities, threshold):
    st.subheader("Модельные метрики")
    predictions = decisions(probabilities, threshold)
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        y, predictions, average="binary", zero_division=0
    )
    for column, title, value in zip(st.columns(3), ["Precision", "Recall", "F1"], [precision, recall, f1]):
        column.metric(title, f"{value:.3f}")
    matrix = metrics.confusion_matrix(y, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    metrics.ConfusionMatrixDisplay(matrix, display_labels=[0, 1]).plot(ax=ax, colorbar=False)
    ax.set(xlabel="Предсказанный класс", ylabel="Истинный класс", title="Матрица ошибок")
    st.pyplot(fig)
    plt.close(fig)

    tn, fp, fn, tp = matrix.ravel()
    fpr, tpr, _ = metrics.roc_curve(y, probabilities)
    pr_precision, pr_recall, _ = metrics.precision_recall_curve(y, probabilities)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(fpr, tpr, label=f"ROC-AUC = {metrics.roc_auc_score(y, probabilities):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].scatter(fp / (fp + tn), tp / (tp + fn), color="red", label="Текущий порог", zorder=3)
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC-кривая")
    axes[1].plot(pr_recall, pr_precision, label=f"AP = {metrics.average_precision_score(y, probabilities):.3f}")
    axes[1].axhline(np.mean(y), linestyle="--", color="gray")
    axes[1].scatter(recall, precision, color="red", label="Текущий порог", zorder=3)
    axes[1].set(xlabel="Recall", ylabel="Precision", title="PR-кривая")
    for ax in axes:
        ax.legend()
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.write(
        "Повышение порога сокращает подборку и обычно повышает precision: "
        "меньше посредственных вин получают рекомендацию. Снижение порога расширяет "
        "подборку и повышает recall: меньше хороших вин пропускается, но возможны лишние рекомендации."
    )
    if not predictions.any():
        st.caption("Рекомендаций нет: precision принят равным 0, как и на отмеченной точке PR-кривой.")


def show_batch(model, threshold):
    st.subheader("Пакетное предсказание")
    st.caption("CSV с 12 признаками без target. Дополнительные столбцы будут сохранены.")
    uploaded = st.file_uploader("Выберите CSV", type=["csv"])
    if uploaded is None:
        return
    try:
        data = pd.read_csv(uploaded)
        result = predict_batch(model, data, threshold)
    except (ValueError, TypeError, UnicodeError, pd.errors.ParserError) as exc:
        st.error(f"Не удалось обработать CSV: {exc}")
        return
    st.dataframe(result, hide_index=True)
    st.download_button(
        "Скачать предсказания",
        result.to_csv(index=False).encode("utf-8-sig"),
        file_name="predictions.csv",
        mime="text/csv",
    )


def show_coefficients(model):
    st.subheader("Что выучила модель")
    coefficients = pd.Series(model.steps[-1][1].coef_[0], index=model.feature_names_in_).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(coefficients.index, coefficients.values, color=["tab:red" if v < 0 else "tab:green" for v in coefficients])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Коэффициент после масштабирования")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    positive = coefficients[coefficients > 0].tail(3).sort_values(ascending=False)
    negative = coefficients[coefficients < 0].head(3)
    st.write("Сильнее всего за хорошее вино: " + ", ".join(positive.index))
    st.write("Сильнее всего против: " + ", ".join(negative.index))
    st.caption("Коэффициенты относятся к стандартизованным признакам. Связь с прогнозом не доказывает причинность.")


def main():
    st.set_page_config(page_title="Качество вина", layout="wide")
    st.title("Качество вина")
    st.caption("Хорошее вино: экспертная оценка не ниже 7. Модель уже обучена.")
    try:
        model = load_model()
        data = load_test_data()
        threshold_default = float(json.loads((APP_DIR / "config.json").read_text(encoding="utf-8"))["threshold"])
        if not 0 <= threshold_default <= 1:
            raise ValueError("Порог в config.json должен находиться в [0, 1].")
        if list(model.classes_) != [0, 1]:
            raise ValueError("Ожидается модель с классами 0 и 1.")
        X = validate_batch(data.drop(columns="target"), model.feature_names_in_)
        y = data["target"].to_numpy()
        if not set(y) == {0, 1}:
            raise ValueError("Тестовая выборка должна содержать оба класса 0 и 1.")
        probabilities = test_proba(model, X)
    except Exception as exc:
        st.error(f"Не удалось загрузить артефакты приложения: {exc}")
        st.stop()
    threshold = st.sidebar.slider("Порог рекомендации", 0.0, 1.0, threshold_default, step=0.001, format="%.6f", key="threshold")
    st.sidebar.caption("Общий порог для всех вкладок. Рекомендация при вероятности ≥ порога.")
    tabs = st.tabs(["Оценка одного вина", "Модельные метрики", "Пакетное предсказание", "Что выучила модель"])
    with tabs[0]:
        show_single(model, X, threshold)
    with tabs[1]:
        show_metrics(y, probabilities, threshold)
    with tabs[2]:
        show_batch(model, threshold)
    with tabs[3]:
        show_coefficients(model)


if __name__ == "__main__":
    main()
