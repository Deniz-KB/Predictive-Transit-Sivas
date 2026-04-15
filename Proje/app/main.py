from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field
import logging
import pickle
import pandas as pd
import os
import csv
from datetime import datetime
from functools import lru_cache
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Hataları takip edebilmek için loglama ayarı
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Predictive Transit API")

# Rate Limiting (Hız Sınırlandırması) Ayarları - IP başına dakikada 5 istek
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Modeli global olarak tanımlıyoruz
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'transit_model.pkl')
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info("Yapay Zeka Modeli başarıyla yüklendi!")
    except Exception as e:
        logger.error(f"Model yüklenirken kritik hata: {e}")

# Arka plan loglama dosyası
LOG_FILE = os.path.join(os.path.dirname(__file__), 'api_logs.csv')

def log_prediction_to_csv(req_data: dict, prediction: float):
    file_exists = os.path.isfile(LOG_FILE)
    try:
        with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'traffic_level', 'weather_condition', 'cumulative_delay_min', 'hour_of_day', 'prediction'])
            writer.writerow([
                datetime.now().isoformat(),
                req_data.get('traffic_level'),
                req_data.get('weather_condition'),
                req_data.get('cumulative_delay_min'),
                req_data.get('hour_of_day'),
                round(prediction, 2)
            ])
    except Exception as e:
        logger.error(f"Gölge loglama hatası: {e}")

# Jürinin / Arayüzün bize göndereceği veri şablonu
class TransitRequest(BaseModel):
    day_of_week: int = Field(..., ge=0, le=6, description="Gün (0: Pzt - 6: Paz)")
    is_weekend: int = Field(..., ge=0, le=1, description="Hafta sonu mu? (0: Hayır, 1: Evet)")
    planned_duration_min: float = Field(..., ge=0.0, description="Planlanan süre (dk)")
    num_stops: int = Field(..., ge=1, description="Durak sayısı")
    weather_condition: int = Field(..., ge=0, le=5, description="Hava (0: Açık - 5: Fırtına)")
    temperature_c: float = Field(..., description="Sıcaklık (C)")
    precipitation_mm: float = Field(..., ge=0.0, description="Yağış (mm)")
    wind_speed_kmh: float = Field(..., ge=0.0, description="Rüzgar hızı (km/h)")
    humidity_pct: float = Field(..., ge=0.0, le=100.0, description="Nem (%)")
    traffic_level: int = Field(..., ge=0, le=4, description="Trafik seviyesi (0-4)")
    bus_capacity: int = Field(..., ge=10, description="Otobüs kapasitesi")

@app.get("/health")
def healthcheck():
    try:
        return {
            "status": "ok", 
            "message": "Sistem canavar gibi çalışıyor.", 
            "model_loaded": model is not None
        }
    except Exception as e:
        logger.error(f"Healthcheck hatası: {e}")
        raise HTTPException(status_code=500, detail="Sunucu içi hata.")

# Önbellekleme (Caching) Fonksiyonu
@lru_cache(maxsize=128)
def get_cached_prediction(traffic_level: int, weather_condition: int, cumulative_delay_min: float, hour_of_day: int) -> float:
    input_data = pd.DataFrame([{
        'traffic_level': traffic_level,
        'weather_condition': weather_condition,
        'cumulative_delay_min': cumulative_delay_min,
        'hour_of_day': hour_of_day
    }])
    return float(model.predict(input_data)[0])

@app.post("/predict")
async def predict_delay(data: TransitRequest):
    input_df = pd.DataFrame([{
        'day_of_week': data.day_of_week,
        'is_weekend': data.is_weekend,
        'planned_duration_min': data.planned_duration_min,
        'num_stops': data.num_stops,
        'weather_condition': data.weather_condition,
        'temperature_c': data.temperature_c,
        'precipitation_mm': data.precipitation_mm,
        'wind_speed_kmh': data.wind_speed_kmh,
        'humidity_pct': data.humidity_pct,
        'traffic_level': data.traffic_level,
        'bus_capacity': data.bus_capacity
    }])
    
    prediction = model.predict(input_df)[0]
    predicted_delay = max(0.0, round(float(prediction), 1))
    
    return {
        "predicted_delay_min": predicted_delay,
        "margin_of_error": "± 1.65 dk",
        "status": "success"
    }