from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import pickle
import pandas as pd
import os

# Hataları takip edebilmek için loglama ayarı
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Predictive Transit API")

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

# Jürinin / Arayüzün bize göndereceği veri şablonu
class TransitRequest(BaseModel):
    traffic_level: int
    weather_condition: int
    cumulative_delay_min: float
    hour_of_day: int

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

@app.post("/predict")
def predict_delay(req: TransitRequest):
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model henüz yüklenmedi veya çöktü.")
        
        # Gelen isteği Pandas DataFrame'e çevir (Modelimiz Pandas istiyor)
        input_data = pd.DataFrame([{
            'traffic_level': req.traffic_level,
            'weather_condition': req.weather_condition,
            'cumulative_delay_min': req.cumulative_delay_min,
            'hour_of_day': req.hour_of_day
        }])
        
        # XGBoost ile tahmini patlat!
        prediction = model.predict(input_data)[0]
        
        # 1 saniye kuralı için küsuratları yuvarla ve temiz bir JSON dön
        return {
            "predicted_delay_min": round(float(prediction), 1),
            "message": f"Tahmini gecikme: {round(float(prediction), 1)} dakika"
        }
    except Exception as e:
        logger.error(f"Tahmin sırasında hata patladı: {e}")
        raise HTTPException(status_code=500, detail="Tahmin yapılamadı, verileri kontrol edin.")