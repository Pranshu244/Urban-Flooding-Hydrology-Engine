from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, Response
from model.predict import model, Model_version, predict
import pandas as pd
import logging
import sys
import io

logger = logging.getLogger("hydrology_backend")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler("api.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


app = FastAPI()


@app.get("/")
def home():
    return {"message": "Urban Flooding & Hydrology Engine"}


@app.get("/health")
def ai_model_health_check():
    return {
        "Status": "ok",
        "Version": Model_version,
        "Model Loaded": model is not None
    }


@app.post("/predict")
def predict_flood(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".csv"):
            logger.warning("Invalid file type received")
            return JSONResponse(
                status_code=400,
                content={"error": "Only CSV files are supported"}
            )
        df = pd.read_csv(file.file)
        logger.info(f"Received CSV with shape {df.shape}")
        output_df = predict(df)
        buffer = io.StringIO()
        output_df.to_csv(buffer, index=False)
        logger.info("Prediction completed successfully")
        return Response(
            content=buffer.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=prediction.csv"}
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )