import numpy as np
import pandas as pd

# Load the given dataset
given_df = pd.read_csv("given_dataset.csv", sep=";")

# Analyze categorical distribution
category_probabilities = given_df["Category1"].value_counts(normalize=True)
print("Given Category1 distribution:\n", category_probabilities)

# Analyze Value1 and Value2 distributions
mean_val1, std_val1 = given_df["Value1"].mean(), given_df["Value1"].std()
mean_val2, std_val2 = given_df["Value2"].mean(), given_df["Value2"].std()

print(f"Given Value1: mean={mean_val1:.2f}, std={std_val1:.2f}")
print(f"Given Value2: mean={mean_val2:.2f}, std={std_val2:.2f}")

# Step 1: Altered parameters
new_num_samples = 750  # generate more samples
category_probabilities__new = category_probabilities + np.random.normal(0, 0.01, len(category_probabilities))  # add small noise
category_probabilities__new = np.clip(category_probabilities__new, 0.01, 0.9)  # avoid negative probs
category_probabilities__new = category_probabilities__new / category_probabilities__new.sum()  # normalize to 1
print("New Category1 distribution:\n", category_probabilities__new)

# Sample new dataset
np.random.seed(123)
new_df = pd.DataFrame({
    "Category1": np.random.choice(category_probabilities.index, size=new_num_samples, p=category_probabilities__new.values),
    "Value1": np.random.normal(loc=mean_val1 * 1.01, scale=std_val1 * 0.99, size=new_num_samples),
    "Value2": np.random.normal(loc=mean_val2 * 0.98, scale=std_val2 * 1.02, size=new_num_samples),
})

# Save new dataset
new_df.to_csv("new_dataset.csv", sep=";", index=False)

print("\nNew dataset has been successfully generated and saved as 'new_dataset.csv'")
