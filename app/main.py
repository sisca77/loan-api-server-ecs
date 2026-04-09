from contextlib import asynccontextmanager
from datetime import datetime
from http.client import HTTPException
import json
import logging
import uuid

from app.model import LoanModel
from app.schemas import LoanRequest, LoanResponse
from fastapi import FastAPI
from dotenv import load_dotenv

from pathlib import Path
env_path = Path("__file__").resolve().parent / ".env"
load_dotenv(dotenv_path= env_path, override=False)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('대출 심사 모델을 로드합니다.')
    model = LoanModel()
    try:
        model.load()
        logger.info('모델 로드 성공')
    except Exception as e:
        logger.error(f'모델 로드 실패: {e}')
        logger.warning('/predict 엔드포인트는 모델 로드 후 사용가능')
    app.state.model = model

    yield

    logger.info('대출 심사 API를 종료합니다.')

app = FastAPI(
    title = '대출 심사 예측 API',
    description = 'ML 모델 기반 대출 승인 여부를 예측하는 API',
    version = '1.0.0',
    lifespan = lifespan
)

@app.get('/')
async def root():
    return {"message": "환영합니다!!! 대출심사 사이트입니다. -- docker version"}

@app.get('/health')
async def health_check():
    model = app.state.model 
    model_loaded = model.pipeline is not None
    return {
        "status" : "healthy" if model_loaded else "degraded",
        "model_loaded" : model_loaded
    }

@app.post("/predict", response_model=LoanResponse)
async def predict(request: LoanRequest):
    model = app.state.model
    request_id = str(uuid.uuid4())
    start_time = datetime.now()

    try:
        result = model.predict(request.model_dump())
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000

        # CloudWatch에 남을 예측 로그 - mlops는 로그를 잘 남기는 것도 중요합니다.
        log_data = {
            "request_id": request_id,
            "timestamp": start_time.isoformat(),
            **request.model_dump(),
            "approved": result["approved"],
            "probability": result["probability"],
            "risk_grade": result["risk_grade"],
            "model_version": model.model_version,
            "latency_ms": round(latency_ms, 2)
        }
        logger.info(f"PREDICTION_LOG: {json.dumps(log_data, ensure_ascii=False)}") 
        # 예측 로그는 PREDICTION_LOG라는 고유한 태그로 CloudWatch에 남겨서, 나중에 이 로그만 필터링해서 볼 수 있도록 했습니다.

        return LoanResponse(**result)

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail="입력값처리오류")
    except Exception as e:
        raise HTTPException(status_code=500)