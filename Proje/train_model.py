import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import pickle

def main():
    print("Veri seti yükleniyor...")
    try:
        # Veriyi belirtilen klasörden okuyoruz
        df = pd.read_csv("data/bus_trips.csv")
    except FileNotFoundError:
        print("Hata: Veri seti bulunamadı. Lütfen 'Sivas Hackathon Proje Data Verileri/bus_trips.csv' yolunun doğru olduğundan emin olun.")
        return

    print("Eksik veriler (NaN) 0 ile dolduruluyor...")
    df = df.fillna(0)

    # 11 adet özelliğimiz ve hedef değişkenimiz
    features = [
        'day_of_week', 'is_weekend', 'planned_duration_min', 'num_stops',
        'weather_condition', 'temperature_c', 'precipitation_mm',
        'wind_speed_kmh', 'humidity_pct', 'traffic_level', 'bus_capacity'
    ]
    target = 'total_delay_min'

    print("Kategorik özellikler (metin verileri) sayısal formata dönüştürülüyor...")
    for col in features:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    print("Veri X (özellikler) ve y (hedef) olarak ayrılıyor...")
    X = df[features]
    y = df[target]

    print("Eğitim ve test setleri (%80 Train, %20 Test) bölünüyor...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("XGBoost modeli eğitiliyor...")
    model = xgb.XGBRegressor(random_state=42)
    model.fit(X_train, y_train)

    print("Test verisi üzerinde tahmin yapılıyor...")
    y_pred = model.predict(X_test)

    # Metriklerin hesaplanması
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n" + "="*45)
    print("        MODEL PERFORMANS METRİKLERİ")
    print("="*45)
    print(f" 🎯 MAE  (Ortalama Mutlak Hata)   : {mae:.4f}")
    print(f" 🎯 RMSE (Kök Ortalama Kare Hata) : {rmse:.4f}")
    print("="*45 + "\n")

    print("Model 'transit_model.pkl' olarak kaydediliyor...")
    with open("transit_model.pkl", "wb") as f:
        pickle.dump(model, f)
        
    print("İşlem başarıyla tamamlandı!")

if __name__ == "__main__":
    main()