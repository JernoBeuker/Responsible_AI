import fairlearn.postprocessing as post
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .analysis import evaluate_model, feature_analysis

SEED = 42  # set seed for consistent results
np.random.seed(SEED)
THRESHOLD = True


def transform_data(data):
    """
    Preprocess the data to get X, X_ohe and y.
    Expanding on Fairlearns preprocessing, we will also extract the race column for analysis.
    See Fairlearns documentation here: https://github.com/fairlearn/talks/blob/main/2021_scipy_tutorial/preprocess.py
    """

    # Getting X without the targets.
    y = data["readmit_30_days"].copy()
    patient_nbr = data["patient_nbr"].copy()

    race = data["race"].copy()
    age = data["age"].copy()
    X = data.drop(
        columns=["readmitted", "readmit_binary", "readmit_30_days", "patient_nbr"]
    ).copy()

    X_ohe = pd.get_dummies(X)
    return patient_nbr, race, age, X_ohe, y


def train_model(X_train, y_train, X_val, y_val, race_val, threshold=True):
    # Train the model
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=10000, random_state=SEED)),
        ]
    )
    model.fit(X_train, y_train)

    if threshold:
        threshold_optimizer = post.ThresholdOptimizer(
            estimator=model,
            constraints="equalized_odds",  # this makes both true_positive and false_positive rate equal across groups
            objective="balanced_accuracy_score",
            predict_method="predict_proba",
            prefit=True,
        )
        model = threshold_optimizer.fit(X_val, y_val, sensitive_features=race_val)
        return model, threshold_optimizer
    else:
        return model, False


def train_val_test_split_data(X, y, race, age, patient_nbr):
    # First split: 80% train+validation, 20% test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    train_val_idx, test_idx = next(gss.split(X, y, groups=patient_nbr))

    train_idx_rel, val_idx_rel = next(
        gss.split(
            X.iloc[train_val_idx],
            y.iloc[train_val_idx],
            groups=patient_nbr.iloc[train_val_idx],
        )
    )

    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]

    def subset(indices):
        return (X.iloc[indices], y.iloc[indices], race.iloc[indices], age.iloc[indices])

    return (subset(train_idx), subset(val_idx), subset(test_idx))


# def train_test_split_data(patient_nbr,X_ohe, y, race, age, group_shuffle_split=True):
#     if group_shuffle_split:
#         gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)

#         train_idx, test_idx = next(gss.split(X_ohe, y, patient_nbr))

#         X_train, X_test = X_ohe.iloc[train_idx], X_ohe.iloc[test_idx]
#         y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
#         race_train, race_test = race.iloc[train_idx], race.iloc[test_idx]
#         age_train, age_test = age.iloc[train_idx], age.iloc[test_idx]

#         X_train.drop(columns=["patient_nbr"], inplace=True)
#         X_test.drop(columns=["patient_nbr"], inplace=True)

#     else:
#         X_train, X_test, y_train, y_test, race_train, race_test, age_train, age_test = (
#                     train_test_split(X_ohe, y, race, age, random_state=SEED, stratify=y)
#                 )

#     return X_train, X_test, y_train, y_test, race_train, race_test, age_train, age_test


def main():
    # Load the diabetes hospital dataset and transform it
    data = pd.read_csv("data/diabetic_preprocessed.csv")
    patient_nbr, race, age, X_ohe, y = transform_data(data)

    # Analyze and print/plot by race and age
    feature_analysis(race, y, "race")
    feature_analysis(age, y, "age")

    # Train test split, for every tuple it contans: X, Y, race, age
    train, val, test = train_val_test_split_data(X_ohe, y, race, age, patient_nbr)

    sensitive_data = pd.DataFrame(
        np.stack([test[2], test[3]], axis=1), columns=["Race", "Age"]
    )

    # Train and Evaluate the model
    model, threshold_optimizer = train_model(
        train[0], train[1], val[0], val[1], val[2], THRESHOLD
    )
    evaluate_model(model, test[0], test[1], sensitive_data, THRESHOLD)

    if threshold_optimizer:
        post.plot_threshold_optimizer(threshold_optimizer)


if __name__ == "__main__":
    main()
