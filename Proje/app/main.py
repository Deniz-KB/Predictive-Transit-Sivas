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
    traffic_level: int = Field(..., ge=0, description="Trafik yoğunluk seviyesi (negatif olamaz)")
    weather_condition: int = Field(..., ge=0, description="Hava durumu kodu (negatif olamaz)")
    cumulative_delay_min: float = Field(..., ge=0.0, description="Şu ana kadarki gecikme süresi")
    hour_of_day: int = Field(..., ge=0, le=23, description="Günün saati (0-23 arası olmalı)")

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
@limiter.limit("5/minute")
def predict_delay(request: Request, req: TransitRequest, background_tasks: BackgroundTasks):
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model henüz yüklenmedi veya çöktü.")
        
        # 1. Önbellekli Tahmin (Caching) - Aynı veri gelirse model çağrılmaz
        prediction = get_cached_prediction(
            req.traffic_level,
            req.weather_condition,
            req.cumulative_delay_min,
            req.hour_of_day
        )
        
        # 2. Gölge Loglama (Shadow Logging) - Arka planda çalışır
        background_tasks.add_task(log_prediction_to_csv, req.dict(), prediction)
        
        # 3. Hata Payı ve Güven Aralığı (Margin of Error)
        rmse = 0.64
        pred_rounded = round(prediction, 1)
        lower_bound = round(prediction - rmse, 1)
        upper_bound = round(prediction + rmse, 1)
        
        return {
            "predicted_delay_min": pred_rounded,
            "margin_of_error": f"± {rmse}",
            "estimated_range": f"{lower_bound} - {upper_bound} dakika",
            "message": f"Tahmini gecikme: {pred_rounded} dakika"
        }
    except Exception as e:
        logger.error(f"Tahmin sırasında hata patladı: {e}")
        raise HTTPException(status_code=500, detail="Tahmin yapılamadı, verileri kontrol edin.")