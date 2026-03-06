from sklearn.preprocessing import MinMaxScaler
import joblib

Model_version='1.0.0'
model = joblib.load("model/Hydrology_Engine_v1.pkl")
mscaler=MinMaxScaler()

def predict(df):
    df=df.copy()
    df['cluster']=model.predict(df)
    df[["rain_norm", "elev_norm"]] = mscaler.fit_transform(df[["rain_95p_mm", "elevation_m"]])
    df["inundation_weight"] = df["rain_norm"] * (1.1 - df["elev_norm"])
    return df