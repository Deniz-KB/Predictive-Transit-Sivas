import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

def main():
    print("Veri seti yükleniyor...")
    try:
        df = pd.read_csv("data/bus_trips.csv")
    except FileNotFoundError:
        print("Hata: 'data/bus_trips.csv' bulunamadı. Lütfen dosya yolunu kontrol edin.")
        return

    print("Eksik veriler (NaN) 0 ile dolduruluyor...")
    df = df.fillna(0)

    features = [
        'day_of_week', 'is_weekend', 'planned_duration_min', 'num_stops',
        'weather_condition', 'temperature_c', 'precipitation_mm',
        'wind_speed_kmh', 'humidity_pct', 'traffic_level', 'bus_capacity'
    ]
    target = 'total_delay_min'

    print("Kategorik özellikler LabelEncoder ile dönüştürülüyor...")
    # weather_condition ve traffic_level sütunlarını sayısal değerlere dönüştürüyoruz
    categorical_cols = ['weather_condition', 'traffic_level']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    X = df[features]
    y = df[target]

    print("Eğitim ve test setleri (%80 Train, %20 Test) bölünüyor...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": xgb.XGBRegressor(random_state=42)
    }

    results = []

    print("Modeller eğitiliyor ve değerlendiriliyor...\n")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R² Score": r2
        })

    # Sonuçları DataFrame'e çevirip Hata (RMSE) oranına göre küçükten büyüğe sıralıyoruz
    results_df = pd.DataFrame(results).sort_values(by="RMSE", ascending=True)
    
    print("="*65)
    print("                   MODEL KARŞILAŞTIRMA TABLOSU")
    print("="*65)
    print(results_df.to_markdown(index=False, floatfmt=".4f"))
    print("="*65 + "\n")

if __name__ == "__main__":
    main()