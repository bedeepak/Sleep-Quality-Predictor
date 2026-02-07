import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    df = df.dropna()

    # Convert Quality of Sleep score into classes
    if 'Quality of Sleep' in df.columns:
        df['Quality of Sleep'] = pd.cut(
            df['Quality of Sleep'],
            bins=[0, 5, 7, 10],
            labels=[0, 1, 2]  # 0=Poor, 1=Average, 2=Good
        )

    # Encode categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    return df
