import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.datasets import fetch_diabetes_hospital

np.random.seed(42)  # set seed for consistent results

# preprocess
# Fairlearn already did some of this and replaced values, you can find their preprocessing at https://github.com/fairlearn/talks/blob/main/2021_scipy_tutorial/preprocess.py

data = fetch_diabetes_hospital(as_frame=True)
X = data.data.copy()
X.drop(columns=["readmitted", "readmit_binary"], inplace=True)
y = data.target
X_ohe = pd.get_dummies(X)
race = X["race"]
age = X["age"]
print(race.value_counts())
print(age.value_counts())
# missingness
# It seems that after fairlearns preprocessing there are no more missing values.
# print(X.columns.tolist())

# RACE
race_counts = race.value_counts()
plt.figure(figsize=(8, 5))
plt.bar(race_counts.index, race_counts.values)

plt.title("Counts by race")
plt.xlabel("Race")
plt.ylabel("Count")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.savefig("./plots/counts_by_race.png", dpi=300, bbox_inches="tight")
# plt.show()


race_positive = pd.DataFrame({"race": race, "positive": y})

# Count positive cases per race
positive_counts = race_positive.groupby("race")["positive"].sum()
positive_ratio = race_positive.groupby("race")["positive"].sum() / race_counts
print("amount of positives per race")
print(positive_counts)
print("ratio of positives per race")
print(positive_ratio)


# AGE
# same steps as for race

age_counts = age.value_counts()
plt.figure(figsize=(8, 5))
plt.bar(age_counts.index, age_counts.values)

plt.title("Counts by age")
plt.xlabel("age")
plt.ylabel("Count")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.savefig("./plots/counts_by_age.png", dpi=300, bbox_inches="tight")
# plt.show()


age_positive = pd.DataFrame({"age": age, "positive": y})

# Count positive cases per race
positive_counts_age = age_positive.groupby("age")["positive"].sum()
positive_ratio_age = age_positive.groupby("age")["positive"].sum() / age_counts
print("amount of positives per age")
print(positive_counts_age)
print("ratio of positives per age")
print(positive_ratio_age)
