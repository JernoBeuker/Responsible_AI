import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.datasets import fetch_diabetes_hospital

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

# # AGE
# # same steps as for race

# age_counts = age.value_counts()
# plt.figure(figsize=(8, 5))
# plt.bar(age_counts.index, age_counts.values)

# plt.title("Counts by age")
# plt.xlabel("age")
# plt.ylabel("Count")

# plt.xticks(rotation=45, ha="right")

# plt.tight_layout()
# plt.savefig("./plots/counts_by_age.png", dpi=300, bbox_inches="tight")
# # plt.show()


# age_positive = pd.DataFrame({"age": age, "positive": y})

# # Count positive cases per race
# positive_counts_age = age_positive.groupby("age")["positive"].sum()
# positive_ratio_age = age_positive.groupby("age")["positive"].sum() / age_counts
# print("amount of positives per age")
# print(positive_counts_age)
# print("ratio of positives per age")
# print(positive_ratio_age)
