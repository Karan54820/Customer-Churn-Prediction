import pandas as pd

df = pd.read_csv("dataset/preprocessed_testing_data.csv")

df.head()

def create_tenure_bucket(tenure):
    if tenure < 12:
        return 0
    elif tenure <= 36:
        return 1
    else:
        return 2

def calculate_support_call_intensity(support_calls, tenure):
    if tenure == 0:
        return 0
    return support_calls / tenure

def calculate_interaction_frequency(last_interaction, usage_frequency):
    return usage_frequency / (last_interaction + 1)

tenure_bucket = []
for tenure in df['Tenure']:
    if tenure < 12:
        tenure_bucket.append(0)
    elif tenure <= 36:
        tenure_bucket.append(1)
    else:
        tenure_bucket.append(2)
df['Tenure_Bucket'] = tenure_bucket

support_call_intensity = []
for i in range(len(df)):
    support_calls = df['Support Calls'].iloc[i]
    tenure = df['Tenure'].iloc[i]
    if tenure == 0:
        support_call_intensity.append(0)
    else:
        support_call_intensity.append(support_calls / tenure)
df['Support_Call_Intensity'] = support_call_intensity

interaction_frequency = []
for i in range(len(df)):
    last_interaction = df['Last Interaction'].iloc[i]
    usage_frequency = df['Usage Frequency'].iloc[i]
    interaction_frequency.append(usage_frequency / (last_interaction + 1))
df['Interaction_Frequency'] = interaction_frequency

df.head()

output_path = "dataset/engineered_testing_data.csv"
df.to_csv(output_path, index=False)

print("DATA SAVED SUCCESSFULLY")