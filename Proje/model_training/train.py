import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Proje ana dizini ve klasör yollarının dinamik olarak belirlenmesi
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
APP_DIR = os.path.join(BASE_DIR, 'app')

STOP_ARRIVALS_PATH = os.path.join(DATA_DIR, 'stop_arrivals.csv')
WEATHER_PATH = os.path.join(DATA_DIR, 'weather_observations.csv')
MODEL_SAVE_PATH = os.path.join(APP_DIR, 'transit_model.pkl')

def main():
    print("Veri setleri yükleniyor...")
    try:
        df_stops = pd.read_csv(STOP_ARRIVALS_PATH)
        df_weather = pd.read_csv(WEATHER_PATH)
    except FileNotFoundError as e:
        print(f"Hata: Veri dosyası bulunamadı. Lütfen 'data' klasöründe dosyaların olduğundan emin olun.\n{e}")
        return

    print("Zaman hizalaması yapılıyor...")
    
    # Zaman kolonlarını datetime formatına çevir ve adlarını eşitle
    df_stops['timestamp'] = pd.to_datetime(df_stops['actual_arrival'])
    df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp'])
    
    df_stops = df_stops.sort_values('timestamp')
    df_weather = df_weather.sort_values('timestamp')

    # İki tabloda da weather_condition olduğu için çakışmayı önlüyoruz:
    if 'weather_condition' in df_weather.columns:
        df_weather = df_weather.drop(columns=['weather_condition'])

    # İki veriyi zamana göre en yakın eşleşmeyle birleştir
    df_merged = pd.merge_asof(df_stops, df_weather, on='timestamp', direction='nearest')

    # Özellik mühendisliği: Günün saati
    df_merged['hour_of_day'] = df_merged['timestamp'].dt.hour

    # Kullanılacak hedef ve öznitelikler
    features = ['traffic_level', 'weather_condition', 'cumulative_delay_min', 'hour_of_day']
    target = 'delay_min'

    missing_cols = [col for col in features + [target] if col not in df_merged.columns]
    if missing_cols:
        print(f"Hata: İstenen özellikler veya hedef değişken veri setinde yok: {missing_cols}")
        return

    print("Eksik veriler temizleniyor ve Encoding işlemi uygulanıyor...")
    df_model = df_merged[features + [target]].dropna().copy()

    # Kategorik değişkenlerin (traffic_level, weather_condition) sayısallaştırılması
    categorical_cols = ['traffic_level', 'weather_condition']
    for col in categorical_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))

    X = df_model[features]
    y = df_model[target]

    print("Eğitim ve test veri setleri ayrılıyor...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("XGBoost Regressor modeli eğitiliyor...")
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    print(f"Eğitim tamamlandı! Modelin Test Seti R^2 Skoru: {score:.4f}")

    print("Model klasöre kaydediliyor...")
    os.makedirs(APP_DIR, exist_ok=True)
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model, f)

    print(f"İşlem başarılı! Model {MODEL_SAVE_PATH} konumuna kaydedildi.")

if __name__ == "__main__":
    main()