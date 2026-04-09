# 1) 베이스 이미지: Python 3.10이 설치된 가벼운(slim) 리눅스 환경을 사용
FROM python:3.10-slim

# 2) 작업 디렉토리 설정: 컨테이너 내부에서 /app 폴더를 기본 경로로 사용
WORKDIR /app

# 3) 의존성 파일만 먼저 복사 (캐시 활용 - 코드가 바뀌어도 패키지 재설치 방지)
COPY requirements.txt .

# 4) 패키지 설치: --no-cache-dir로 캐시를 남기지 않아 이미지 크기 절약
RUN pip install --no-cache-dir -r requirements.txt

# 5) 나머지 소스 코드 전체를 컨테이너의 /app 폴더로 복사
COPY . .

# 6) 컨테이너가 8000번 포트를 사용한다는 것을 명시 (문서 역할, 실제 포트 열기는 docker run -p 로)
EXPOSE 8000

# 7) 컨테이너 시작 시 실행할 명령어: FastAPI 서버를 0.0.0.0:8000에서 시작
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
