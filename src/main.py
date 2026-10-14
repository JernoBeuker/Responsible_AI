import fairlearn.postprocessing as post
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .analysis import feature_analysis, evaluate_model

SEED = 42  # set seed for consistent results
np.random.seed(SEED)
GSS = True
THRESHOLD = True


def transform_data(data):
    """
    Preprocess the data to get X, X_ohe and y.
    Expanding on Fairlearns preprocessing, we will also extract the race column for analysis.
    See Fairlearns documentation here: https://github.com/fairlearn/talks/blob/main/2021_scipy_tutorial/preprocess.py
    """

    # Getting X without the targets.
    X = data
    y = data["readmit_30_days"]
    patient_nbr=data["patient_nbr"]
    X.drop(columns=["readmitted", "readmit_binary", "readmit_30_days"], inplace=True)

    race = X["race"]
    age = X["age"]
    x_ohe = pd.get_dummies(X)
    return patient_nbr, race, age, x_ohe, y

def train_model(X_train, y_train, race_train, threshold=True):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=SEED,
        ))
    ])
    if threshold:
        threshold_optimizer = post.ThresholdOptimizer(
            estimator=model,
            constraints="false_negative_rate_parity",
            objective="true_positive_rate",
            predict_method="predict_proba",
            prefit=False,
        )
        model = threshold_optimizer.fit(X_train, y_train, sensitive_features=race_train)
        return model, threshold_optimizer
    else:
        model.fit(X_train, y_train)
        return model, False

def train_test_split_data(patient_nbr,X_ohe, y, race, age, group_shuffle_split=True):
    if group_shuffle_split:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    
        train_idx, test_idx = next(gss.split(X_ohe, y, patient_nbr))
    
        X_train, X_test = X_ohe.iloc[train_idx], X_ohe.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        race_train, race_test = race.iloc[train_idx], race.iloc[test_idx]
        age_train, age_test = age.iloc[train_idx], age.iloc[test_idx]

    else:
        X_train, X_test, y_train, y_test, race_train, race_test, age_train, age_test = (
                    train_test_split(X_ohe, y, race, age, random_state=SEED, stratify=y)
                )
        
    return X_train, X_test, y_train, y_test, race_train, race_test, age_train, age_test
    
def main():
    # Load the diabetes hospital dataset and transform it
    data = pd.read_csv("data/diabetic_preprocessed.csv")
    patient_nbr, race, age, X_ohe, y = transform_data(data)
    

    # Analyze and print/plot by race and age
    feature_analysis(race, y, "race")
    feature_analysis(age, y, "age")

    # Train test split
    X_train, X_test, y_train, y_test, race_train, race_test, \
    age_train, age_test = train_test_split_data(patient_nbr, X_ohe, y, race, age, GSS)

    # Train and Evaluate the model
    sensitive_data = pd.DataFrame(
        np.stack([race_test, age_test], axis=1), columns=["Race", "Age"]
    )

    model, threshold_optimizer = train_model(X_train, y_train, race_train, THRESHOLD)
    evaluate_model(model, X_test, y_test, sensitive_data, THRESHOLD)
    if threshold_optimizer:    
        post.plot_threshold_optimizer(threshold_optimizer)


if __name__ == "__main__":
    main()
