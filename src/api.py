import base64
import io
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from .database import init_db, get_db, RequestHistory
from .inference import SegmentationModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "models/segmentation_model.pth"
model = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model
    try:
        model = SegmentationModel(model_path=MODEL_PATH, device="cpu")
        logger.info("Модель успешно загружена")
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        raise
    yield


app = FastAPI(title="ISIC Segmentation API", version="1.0.0", lifespan=lifespan)

init_db()


@app.post("/forward")
async def forward_endpoint(
    image: UploadFile = File(...), db: Session = Depends(get_db)
):
    start_time = time.time()

    try:
        if not image:
            raise HTTPException(status_code=400, detail="bad request")

        image_bytes = await image.read()

        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="bad request")

        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            img_width, img_height = pil_image.size
        except Exception:
            raise HTTPException(status_code=400, detail="bad request")

        try:
            mask = model.predict_segmentation(image_bytes)
        except Exception as e:
            logger.error(f"Ошибка инференса модели: {e}")

            processing_time = time.time() - start_time
            request_log = RequestHistory(
                processing_time=processing_time,
                image_width=img_width,
                image_height=img_height,
                success=False,
                error_message="модель не смогла обработать данные",
            )
            db.add(request_log)
            db.commit()

            raise HTTPException(
                status_code=403, detail="модель не смогла обработать данные"
            )

        mask_image = Image.fromarray(mask)
        buffer = io.BytesIO()
        mask_image.save(buffer, format="PNG")
        mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        processing_time = time.time() - start_time

        request_log = RequestHistory(
            processing_time=processing_time,
            image_width=img_width,
            image_height=img_height,
            success=True,
            error_message=None,
        )
        db.add(request_log)
        db.commit()

        return JSONResponse(
            content={
                "mask": mask_base64,
                "processing_time": processing_time,
                "image_size": {"width": img_width, "height": img_height},
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")

        try:
            processing_time = time.time() - start_time
            request_log = RequestHistory(
                processing_time=processing_time,
                image_width=0,
                image_height=0,
                success=False,
                error_message=str(e),
            )
            db.add(request_log)
            db.commit()
        except:
            pass

        raise HTTPException(status_code=400, detail="bad request")


@app.get("/history")
async def get_history(limit: Optional[int] = 100, db: Session = Depends(get_db)):
    try:
        records = (
            db.query(RequestHistory)
            .order_by(RequestHistory.timestamp.desc())
            .limit(limit)
            .all()
        )

        history = []
        for record in records:
            history.append(
                {
                    "id": record.id,
                    "timestamp": record.timestamp.isoformat(),
                    "processing_time": record.processing_time,
                    "image_width": record.image_width,
                    "image_height": record.image_height,
                    "success": bool(record.success),
                    "error_message": record.error_message,
                }
            )

        return JSONResponse(content={"total": len(history), "records": history})

    except Exception as e:
        logger.error(f"Ошибка получения истории: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    try:
        successful_records = (
            db.query(RequestHistory).filter(RequestHistory.success == True).all()
        )

        if not successful_records:
            return JSONResponse(
                content={"message": "Нет доступных данных", "total_requests": 0}
            )

        processing_times = [r.processing_time for r in successful_records]
        image_widths = [r.image_width for r in successful_records]
        image_heights = [r.image_height for r in successful_records]

        stats = {
            "total_requests": db.query(RequestHistory).count(),
            "successful_requests": len(successful_records),
            "failed_requests": db.query(RequestHistory)
            .filter(RequestHistory.success == False)
            .count(),
            "processing_time": {
                "mean": float(np.mean(processing_times)),
                "p50": float(np.percentile(processing_times, 50)),
                "p95": float(np.percentile(processing_times, 95)),
                "p99": float(np.percentile(processing_times, 99)),
                "min": float(np.min(processing_times)),
                "max": float(np.max(processing_times)),
            },
            "image_sizes": {
                "width": {
                    "mean": float(np.mean(image_widths)),
                    "min": int(np.min(image_widths)),
                    "max": int(np.max(image_widths)),
                },
                "height": {
                    "mean": float(np.mean(image_heights)),
                    "min": int(np.min(image_heights)),
                    "max": int(np.max(image_heights)),
                },
            },
        }

        return JSONResponse(content=stats)

    except Exception as e:
        logger.error(f"Ошибка вычисления статистики: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.get("/health")
async def root():
    return {"status": "ok"}
