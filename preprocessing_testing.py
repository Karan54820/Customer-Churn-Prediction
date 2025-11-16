import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/customer_churn_dataset-testing-master.csv")

sampled_df = df.sample(n=2500, replace=False, random_state=42)

sampled_df = sampled_df.drop_duplicates()

sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)



churn_counts = sampled_df['Churn'].value_counts()

plt.figure(figsize=(6, 4))
churn_counts.plot(kind='bar')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.title('Churn Distribution')
plt.tight_layout()
plt.show()

from sklearn.preprocessing import LabelEncoder

# cat_cols = sampled_df.select_dtypes(include=['object']).columns
# print(cat_cols)

cols = ['Gender', 'Subscription Type']
le = LabelEncoder()
for c in cols:
    sampled_df[c] = le.fit_transform(sampled_df[c])
    print("Category Mapping:", le.classes_)

# print(sampled_df)
# sampled_df.head()

# mapping Contract duration to months

mapping = {
    'Annual': 12,
    'Quarterly': 3,
    'Monthly': 1
}

sampled_df['Contract Duration (Months)'] = df['Contract Length'].map(mapping)
sampled_df.drop('Contract Length', axis=1, inplace=True)
sampled_df.head()

# adding churn col in the back.
# dropping customer id col.

col_data = sampled_df.pop("Churn")
sampled_df.insert(len(sampled_df.columns), "Churn", col_data)
sampled_df.drop('CustomerID', axis=1, inplace=True)
sampled_df.head()

sampled_df.to_csv('dataset/preprocessed_testing_data.csv', index=False)

