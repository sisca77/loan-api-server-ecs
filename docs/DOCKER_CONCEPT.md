# Dockerfile, Image, Container 관계

요리에 비유하면 이해하기 쉽습니다.

| 개념 | 비유 | 설명 |
|------|------|------|
| **Dockerfile** | 레시피 (조리법) | 이미지를 어떻게 만들지 적어놓은 설명서 |
| **Image** | 도시락 완성품 | Dockerfile로 빌드한 실행 가능한 패키지 (파일) |
| **Container** | 도시락을 열어서 먹는 것 | Image를 실행한 상태 (프로세스) |

## 흐름

```
Dockerfile  ---(docker build)--->  Image  ---(docker run)--->  Container
 (설계도)                          (결과물)                      (실행중인 서버)
```

## 실제 명령어로 보면

```bash
# 1. Dockerfile로 이미지 빌드 (레시피로 도시락 만들기)
docker build -t loan-api .

# 2. 이미지로 컨테이너 실행 (도시락 열어서 먹기)
docker run -p 8000:8000 loan-api

# 3. 같은 이미지로 컨테이너 여러 개 실행 가능 (도시락 여러 개 복사)
docker run -p 8001:8000 loan-api
docker run -p 8002:8000 loan-api
```

## 핵심 차이

- **Dockerfile** → 텍스트 파일, 사람이 작성
- **Image** → 읽기 전용, 변하지 않음, 공유 가능 (DockerHub 등)
- **Container** → Image의 실행 인스턴스, 여러 개 만들 수 있고, 각각 독립적

하나의 Image에서 Container를 여러 개 띄울 수 있다는 게 Docker의 핵심입니다.

## .dockerignore란?

`.gitignore`와 같은 개념입니다. `docker build` 시 **이미지에 포함하지 않을 파일/폴더**를 지정합니다.

### 왜 필요한가?

Dockerfile의 `COPY . .` 명령은 현재 디렉토리의 **모든 파일**을 이미지에 복사합니다. `.dockerignore`가 없으면 불필요한 파일까지 전부 들어가게 됩니다.

| 문제 | 설명 |
|------|------|
| 이미지 크기 증가 | 노트북, 문서, 테스트 데이터 등이 포함됨 |
| 보안 위험 | `.env`, `.git` 등 민감한 정보가 이미지에 포함될 수 있음 |
| 빌드 속도 저하 | 불필요한 파일까지 Docker 데몬에 전송 |

### 예시 (.dockerignore)

```
# Git 관련
.git
.gitignore

# 개발/문서
docs/
01_notebooks/
README.md

# 환경 설정 (민감 정보)
.env
*.log

# Python 캐시
__pycache__/
*.pyc

# IDE 설정
.vscode/
.idea/
```

### .gitignore vs .dockerignore 비교

| | .gitignore | .dockerignore |
|---|---|---|
| 역할 | Git에 올리지 않을 파일 | Docker 이미지에 넣지 않을 파일 |
| 적용 시점 | `git add` | `docker build` |
| 목적 | 저장소 관리 | 이미지 경량화 + 보안 |
