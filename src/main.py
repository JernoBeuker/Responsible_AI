import fairlearn.metrics as fm
import fairlearn.postprocessing as post
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.datasets import fetch_diabetes_hospital
from fairlearn.metrics import (
    MetricFrame,
    count,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from fairness_eval import feature_analysis

SEED = 42  # set seed for consistent results
np.random.seed(SEED)


def transform_data(data):
    """
    Preprocess the data to get X, X_ohe and y.
    Expanding on Fairlearns preprocessing, we will also extract the race column for analysis.
    See Fairlearns documentation here: https://github.com/fairlearn/talks/blob/main/2021_scipy_tutorial/preprocess.py
    """

    # Getting X without the targets.
    X = data.data.copy()
    X.drop(columns=["readmitted", "readmit_binary"], inplace=True)

    race = X["race"]
    age = X["age"]
    y = data.target
    x_ohe = pd.get_dummies(X)
    return race, age, x_ohe, y


def train_model(X_train, y_train):
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train, y_train)
    return classifier


def evaluate_model(classifier, X_test, y_test, sensitive_data):
    # Make the test predictions and evaluate the model using the MetricFrame
    # y_pred = classifier.predict_proba(X_test)[:, 1] >= 0.1
    y_pred = classifier.predict(
        X_test, sensitive_features=sensitive_data["Race"], random_state=SEED
    )
    racemf = MetricFrame(
        metrics={
            "balanced_accuracy": balanced_accuracy_score,
            "false positive rate": false_positive_rate,
            "false negative rate": false_negative_rate,
            "selection rate": selection_rate,
        },
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_data["Race"],
    )
    agemf = MetricFrame(
        metrics={
            "balanced_accuracy": balanced_accuracy_score,
            "false positive rate": false_positive_rate,
            "false negative rate": false_negative_rate,
            "selection rate": selection_rate,
        },
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_data["Age"],
    )
    intersectionmf = MetricFrame(
        metrics={
            "balanced_accuracy": balanced_accuracy_score,
            "false positive rate": false_positive_rate,
            "false negative rate": false_negative_rate,
            "selection rate": selection_rate,
        },
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_data,
    )
    allmf = [racemf, agemf, intersectionmf]
    for mf in allmf:
        # mf.overall.item()
        print(mf.by_group)

    for sensitive in ["Race", "Age"]:
        print(sensitive)
        print("demographic parity diff")
        print(
            fm.demographic_parity_difference(
                y_test, y_pred, sensitive_features=sensitive_data[sensitive]
            )
        )

        print("equal opportunity diff")

        print(
            fm.equal_opportunity_difference(
                y_test, y_pred, sensitive_features=sensitive_data[sensitive]
            )
        )

    metrics = {
        "accuracy": accuracy_score,
        # "precision": zero_div_precision_score,
        "false positive rate": false_positive_rate,
        "false negative rate": false_negative_rate,
        "selection rate": selection_rate,
        "count": count,
    }
    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_data,
    )
    plots = metric_frame.by_group.plot.bar(
        subplots=True,
        layout=[3, 3],
        legend=False,
        figsize=[12, 8],
        title="Show all metrics",
    )

    plots[0][0].figure.savefig("./plots/fairnessplots.png")


def main():
    # Load the diabetes hospital dataset and transform it
    data = fetch_diabetes_hospital(as_frame=True)
    print(data.data.columns.tolist())
    race, age, X_ohe, y = transform_data(data)

    # Analyze and print/plot by race and age
    feature_analysis(race, y, "race")
    feature_analysis(age, y, "age")

    # Train test split
    # X_train, X_test, y_train, y_test, race_train, race_test, age_train, age_test = (
    #     train_test_split(X_ohe, y, race, age, random_state=SEED, stratify=y)
    # )

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)

    train_idx, test_idx = next(gss.split(X_ohe, y, groups = data["patient_nbr"]))

    X_train, X_test = X_ohe.iloc[train_idx], X_ohe.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    race_train, race_test = race.iloc[train_idx], race.iloc[test_idx]
    age_train, age_test = age.iloc[train_idx], age.iloc[test_idx]






    # Train and Evaluate the model
    sensitive_data = pd.DataFrame(
        np.stack([race_test, age_test], axis=1), columns=["Race", "Age"]
    )
    # model = train_model(X_train, y_train)
    model = LogisticRegression(max_iter=1000)
    pipeline = Pipeline(steps=[("classifier", model)])

    threshold_optimizer = post.ThresholdOptimizer(
        estimator=pipeline,
        constraints="false_negative_rate_parity",
        objective="true_negative_rate",
        predict_method="predict_proba",
        prefit=False,
    )
    model = threshold_optimizer.fit(X_train, y_train, sensitive_features=race_train)
    evaluate_model(model, X_test, y_test, sensitive_data)
    post.plot_threshold_optimizer(threshold_optimizer)


if __name__ == "__main__":
    main()
