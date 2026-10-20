import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error


def read_dataframe(path):
    df = pd.read_parquet(path)

    # Target: trip duration in minutes
    df["duration"] = (
        df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    ).dt.total_seconds() / 60

    # Remove unrealistic trips (under 1 min or over 1 hour)
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    # Location IDs are categories, not numbers
    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)
    return df


df_train = read_dataframe("data/green_tripdata_2021-01.parquet")
df_val = read_dataframe("data/green_tripdata_2021-02.parquet")
print(f"Train rows: {len(df_train)}, Validation rows: {len(df_val)}")

# Turn categories into one-hot features
features = ["PULocationID", "DOLocationID"]
dv = DictVectorizer()
X_train = dv.fit_transform(df_train[features].to_dict(orient="records"))
X_val = dv.transform(df_val[features].to_dict(orient="records"))

y_train = df_train["duration"].values
y_val = df_val["duration"].values

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
rmse = root_mean_squared_error(y_val, y_pred)
print(f"Validation RMSE: {rmse:.2f} minutes")
