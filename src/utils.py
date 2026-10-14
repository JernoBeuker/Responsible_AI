import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    count,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
)
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import fairlearn.metrics as fm


SEED = 42
np.random.seed(SEED)  # set seed for consistent results

def feature_analysis(feature, y, str_name:str):
    # After fairlearns preprocessing there are no missing values.
    ft_counts = feature.value_counts()

    plt.figure(figsize=(8, 5))
    plt.bar(ft_counts.index, ft_counts.values)
    plt.title(f"Counts by {str_name}")
    plt.xlabel(str_name)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"./plots/counts_by_{str_name}.png", dpi=300, bbox_inches="tight")

    # Count positive cases per race
    ft_positive = pd.DataFrame({f"{str_name}": feature, "positive": y})
    positive_counts = ft_positive.groupby(str_name)["positive"].sum()
    positive_ratio = (positive_counts / ft_counts) * 100

    print(f"amount of positives per {str_name}: ")
    print(positive_counts)
    print(f"ratio of positives per {str_name}: ")
    print(positive_ratio)

def evaluate_model(classifier, X_test, y_test, sensitive_data, threshold=False):
    # Make the test predictions and evaluate the model using the MetricFrame
    if threshold:    
        y_pred = classifier.predict(
            X_test, sensitive_features=sensitive_data["Race"], random_state=SEED
        )
    else:
        y_pred = classifier.predict_proba(X_test)[:, 1] >= 0.1

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
