# MLOps 대출심사 API - CI/CD 배포 가이드

> Docker + GitHub Actions + AWS ECR + ECS를 활용한 CI/CD 파이프라인 구축 가이드
> 처음 접하는 학습생을 위해 단계별로 상세하게 설명합니다.

---

## 목차

1. [전체 아키텍처 이해하기](#1-전체-아키텍처-이해하기)
2. [사전 준비사항](#2-사전-준비사항)
3. [Step 1: Docker로 로컬 실행하기](#3-step-1-docker로-로컬-실행하기)
4. [Step 2: AWS 환경 설정하기](#4-step-2-aws-환경-설정하기)
5. [Step 3: S3에 모델/데이터 업로드하기](#5-step-3-s3에-모델데이터-업로드하기)
6. [Step 4: ECR 저장소 만들기](#6-step-4-ecr-저장소-만들기)
7. [Step 5: ECS 클러스터 & 서비스 만들기](#7-step-5-ecs-클러스터--서비스-만들기)
8. [Step 6: GitHub Actions CI/CD 설정하기](#8-step-6-github-actions-cicd-설정하기)
9. [Step 7: 배포 실행 및 확인하기](#9-step-7-배포-실행-및-확인하기)
10. [트러블슈팅 가이드](#10-트러블슈팅-가이드)
11. [용어 사전](#11-용어-사전)

---

## 1. 전체 아키텍처 이해하기

### CI/CD란?

- **CI (Continuous Integration)**: 코드를 변경할 때마다 자동으로 빌드/테스트하는 것
- **CD (Continuous Deployment)**: 테스트를 통과한 코드를 자동으로 배포하는 것
- 즉, **"코드를 push하면 자동으로 서비스가 업데이트되는 것"**

### 전체 흐름도

```
개발자 PC                    GitHub                         AWS
┌──────────┐    git push    ┌──────────────┐              ┌─────────────┐
│  코드 수정 │ ──────────→  │ GitHub Repo  │              │    S3       │
│  git push │              │              │              │ (모델/데이터) │
└──────────┘              │  ↓ 자동실행    │              └──────┬──────┘
                          │ GitHub Actions│                     │
                          │              │                     │
                          │ 1.Docker빌드  │    push image      │
                          │ 2.ECR push   │ ──────────→ ECR    │
                          │ 3.ECS 배포   │              (이미지 │
                          └──────────────┘              저장소) │
                                                         │     │
                                                         ↓     │
                                                     ┌────────────┐
                                                     │   ECS      │
                                                     │ (컨테이너   │←── 모델 다운로드
                                                     │  실행환경)  │
                                                     └─────┬──────┘
                                                           │
                                                           ↓
                                                     사용자 접속 가능!
                                                  http://xxx.amazonaws.com
```

### 각 서비스의 역할

| 서비스 | 역할 | 비유 |
|--------|------|------|
| **Docker** | 앱을 컨테이너(상자)에 담기 | 이사할 때 짐을 박스에 포장 |
| **GitHub Actions** | 자동화 실행기 | 택배 기사 (자동으로 배달) |
| **ECR** | Docker 이미지 저장소 | 택배 물류 창고 |
| **ECS** | 컨테이너 실행 환경 | 새 집 (앱이 실행되는 곳) |
| **S3** | 파일 저장소 | USB 드라이브 (모델/데이터 보관) |

---

## 2. 사전 준비사항

### 2-1. 필수 설치 프로그램

```bash
# 1. Docker Desktop 설치
# https://www.docker.com/products/docker-desktop/ 에서 다운로드

# 2. AWS CLI 설치
# https://aws.amazon.com/ko/cli/ 에서 다운로드

# 3. Git 설치 (이미 설치되어 있을 가능성 높음)
git --version

# 설치 확인
docker --version      # Docker version 24.x.x 이상
aws --version         # aws-cli/2.x.x 이상
```

### 2-2. AWS 계정 준비

1. [AWS 가입](https://aws.amazon.com/ko/) (신용카드 필요, 프리티어 1년 무료)
2. **IAM 사용자 생성** (루트 계정 직접 사용 금지!)

```
AWS 콘솔 로그인 → IAM → 사용자 → 사용자 생성
  - 사용자 이름: mlops-deploy
  - 액세스 유형: 프로그래밍 방식 액세스 (Access Key)
  - 권한 정책 연결:
    ✅ AmazonECS_FullAccess
    ✅ AmazonEC2ContainerRegistryFullAccess
    ✅ AmazonS3FullAccess
    ✅ AmazonSSMReadOnlyAccess
    ✅ CloudWatchLogsFullAccess
  - Access Key ID와 Secret Access Key를 반드시 저장!
```

### 2-3. AWS CLI 설정

```bash
aws configure
# AWS Access Key ID: (위에서 받은 Access Key ID)
# AWS Secret Access Key: (위에서 받은 Secret Access Key)
# Default region name: ap-northeast-2
# Default output format: json
```

### 2-4. GitHub 저장소 준비

```bash
# 프로젝트 폴더에서 실행
cd mlops-edu-main

# GitHub에 새 저장소를 만든 후
git init
git remote add origin https://github.com/<본인계정>/mlops-loan-api.git
git add .
git commit -m "Initial commit"
git push -u origin main
```

---

## 3. Step 1: Docker로 로컬 실행하기

> 먼저 로컬에서 Docker가 잘 동작하는지 확인합니다.

### 3-1. 프로젝트 구조 확인

```
mlops-edu-main/
├── app/
│   ├── __init__.py          ← 패키지 인식용
│   ├── main.py              ← FastAPI 앱
│   ├── model.py             ← 모델 로드 + S3 다운로드
│   ├── schemas.py           ← 요청/응답 스키마
│   └── gemini_client.py     ← OpenAI API 클라이언트
├── models/                  ← 로컬 모델 파일 (S3에도 업로드)
│   ├── loan_pipeline.pkl
│   ├── label_encoders.pkl
│   └── feature_names.pkl
├── data/                    ← 데이터 파일 (S3에도 업로드)
├── .github/workflows/
│   └── deploy.yml           ← CI/CD 파이프라인
├── .aws/
│   └── task-definition.json ← ECS 태스크 정의
├── scripts/
│   └── upload_models_to_s3.sh
├── Dockerfile               ← Docker 빌드 설정
├── .dockerignore             ← Docker에서 제외할 파일
├── requirements.txt
└── .env                      ← 환경 변수 (Git에 올리지 않음!)
```

### 3-2. Docker 이미지 빌드

```bash
# 프로젝트 루트에서 실행
cd mlops-edu-main

# Docker 이미지 빌드 (-t는 이미지에 이름 붙이기)
docker build -t mlops-loan-api .
```

빌드 과정 설명:
```
Step 1: Python 3.10 이미지 다운로드
Step 2: /app 디렉토리 생성
Step 3: 시스템 패키지 설치
Step 4: pip install (requirements.txt)
Step 5: app/ 소스 코드 복사
Step 6: 포트 8000 설정
Step 7: 헬스체크 설정
Step 8: uvicorn 실행 명령어 설정
```

### 3-3. 로컬에서 Docker 컨테이너 실행

```bash
# 로컬 테스트 시: models 폴더를 마운트하고 환경변수 전달
docker run -d \
  --name loan-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -e OPENAI_API_KEY=your-openai-key-here \
  mlops-loan-api
```

> **Windows PowerShell인 경우:**
> ```powershell
> docker run -d `
>   --name loan-api `
>   -p 8000:8000 `
>   -v ${PWD}/models:/app/models `
>   -e OPENAI_API_KEY=your-openai-key-here `
>   mlops-loan-api
> ```

### 3-4. 동작 확인

```bash
# 헬스체크
curl http://localhost:8000/health
# 응답: {"status":"healthy","model_loaded":true}

# 대출 예측 테스트
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "gender": "남",
    "annual_income": 5000.0,
    "employment_years": 5,
    "housing_type": "자가",
    "credit_score": 720,
    "existing_loan_count": 2,
    "annual_card_usage": 2400.0,
    "debt_ratio": 35.5,
    "loan_amount": 3000.0,
    "loan_purpose": "주택구입",
    "repayment_method": "원리금균등",
    "loan_period": 36
  }'

# 컨테이너 중지 & 삭제
docker stop loan-api && docker rm loan-api
```

---

## 4. Step 2: AWS 환경 설정하기

### 4-1. ECS 태스크 실행 IAM 역할 만들기

```
AWS 콘솔 → IAM → 역할 → 역할 생성

역할 1: ecsTaskExecutionRole (ECS가 ECR에서 이미지를 받고 로그를 기록하는 역할)
  - 신뢰할 수 있는 엔터티: AWS 서비스 > Elastic Container Service Task
  - 정책 연결:
    ✅ AmazonECSTaskExecutionRolePolicy
    ✅ AmazonSSMReadOnlyAccess        ← Secrets Manager/SSM 파라미터 읽기용

역할 2: ecsTaskRole (컨테이너 안에서 S3에 접근하는 역할)
  - 신뢰할 수 있는 엔터티: AWS 서비스 > Elastic Container Service Task
  - 정책 연결:
    ✅ AmazonS3ReadOnlyAccess         ← S3에서 모델 다운로드용
```

### 4-2. OpenAI API 키를 AWS Systems Manager에 저장

API 키를 안전하게 보관하기 위해 SSM Parameter Store를 사용합니다.

```bash
aws ssm put-parameter \
  --name "/mlops/openai-api-key" \
  --value "sk-proj-여기에실제키입력" \
  --type "SecureString" \
  --region ap-northeast-2
```

### 4-3. CloudWatch 로그 그룹 만들기

```bash
aws logs create-log-group \
  --log-group-name /ecs/mlops-loan-api \
  --region ap-northeast-2
```

---

## 5. Step 3: S3에 모델/데이터 업로드하기

### 5-1. 스크립트로 업로드

```bash
# 프로젝트 루트에서 실행
bash scripts/upload_models_to_s3.sh
```

### 5-2. 수동으로 업로드 (스크립트가 안 될 경우)

```bash
# S3 버킷 만들기
aws s3 mb s3://mlops-loan-models --region ap-northeast-2

# 모델 파일 업로드
aws s3 cp models/loan_pipeline.pkl s3://mlops-loan-models/models/
aws s3 cp models/label_encoders.pkl s3://mlops-loan-models/models/
aws s3 cp models/feature_names.pkl s3://mlops-loan-models/models/

# 데이터 파일 업로드
aws s3 cp data/ s3://mlops-loan-models/data/ --recursive

# 업로드 확인
aws s3 ls s3://mlops-loan-models/ --recursive
```

**예상 결과:**
```
2024-01-01 12:00:00     217088 models/loan_pipeline.pkl
2024-01-01 12:00:00        214 models/feature_names.pkl
2024-01-01 12:00:00       1600 models/label_encoders.pkl
2024-01-01 12:00:00     110000 data/loan_data.csv
...
```

---

## 6. Step 4: ECR 저장소 만들기

> ECR = Docker 이미지를 저장하는 AWS의 창고

### 6-1. ECR 저장소 생성

```bash
aws ecr create-repository \
  --repository-name mlops-loan-api \
  --region ap-northeast-2
```

**응답에서 `repositoryUri` 값을 기록해두세요:**
```
123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/mlops-loan-api
```

### 6-2. task-definition.json 수정

`.aws/task-definition.json` 파일에서 `<AWS_ACCOUNT_ID>`를 본인의 AWS 계정 ID로 변경:

```
# 변경 전
"arn:aws:iam::<AWS_ACCOUNT_ID>:role/ecsTaskExecutionRole"

# 변경 후 (예시)
"arn:aws:iam::123456789012:role/ecsTaskExecutionRole"
```

**총 4곳**을 변경해야 합니다:
- `executionRoleArn`
- `taskRoleArn`
- `image`
- `secrets > valueFrom`

---

## 7. Step 5: ECS 클러스터 & 서비스 만들기

> ECS = Docker 컨테이너를 실행하는 AWS의 서비스

### 7-1. ECS 클러스터 생성

```bash
aws ecs create-cluster \
  --cluster-name mlops-cluster \
  --region ap-northeast-2
```

### 7-2. 기본 VPC 및 서브넷 확인

```bash
# 기본 VPC ID 확인
aws ec2 describe-vpcs \
  --filters "Name=isDefault,Values=true" \
  --query "Vpcs[0].VpcId" --output text

# 서브넷 ID 확인 (2개 이상 필요)
aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=<위에서_받은_VPC_ID>" \
  --query "Subnets[*].SubnetId" --output text

# 보안 그룹 생성 (포트 8000 열기)
aws ec2 create-security-group \
  --group-name mlops-ecs-sg \
  --description "ECS Loan API" \
  --vpc-id <VPC_ID>

# 포트 8000 인바운드 규칙 추가
aws ec2 authorize-security-group-ingress \
  --group-id <SECURITY_GROUP_ID> \
  --protocol tcp \
  --port 8000 \
  --cidr 0.0.0.0/0
```

### 7-3. 태스크 정의 등록

```bash
aws ecs register-task-definition \
  --cli-input-json file://.aws/task-definition.json \
  --region ap-northeast-2
```

### 7-4. ECS 서비스 생성

```bash
aws ecs create-service \
  --cluster mlops-cluster \
  --service-name mlops-loan-service \
  --task-definition mlops-loan-task \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={
    subnets=[<SUBNET_ID_1>,<SUBNET_ID_2>],
    securityGroups=[<SECURITY_GROUP_ID>],
    assignPublicIp=ENABLED
  }" \
  --region ap-northeast-2
```

---

## 8. Step 6: GitHub Actions CI/CD 설정하기

### 8-1. GitHub Secrets 등록

> 중요! AWS 키를 코드에 직접 넣지 않고, GitHub Secrets에 안전하게 저장합니다.

```
GitHub 저장소 → Settings → Secrets and variables → Actions → New repository secret

등록할 Secrets:
┌────────────────────────────┬─────────────────────────┐
│ Name                       │ Value                   │
├────────────────────────────┼─────────────────────────┤
│ AWS_ACCESS_KEY_ID          │ AKIA...                 │
│ AWS_SECRET_ACCESS_KEY      │ wJal...                 │
└────────────────────────────┴─────────────────────────┘
```

**등록 방법 (스크린샷 대체 설명):**
1. GitHub에서 본인 저장소 클릭
2. 상단 탭에서 **Settings** 클릭
3. 왼쪽 메뉴에서 **Secrets and variables** → **Actions** 클릭
4. **New repository secret** 버튼 클릭
5. Name에 `AWS_ACCESS_KEY_ID`, Secret에 키 값 입력 후 **Add secret**
6. 같은 방법으로 `AWS_SECRET_ACCESS_KEY`도 등록

### 8-2. 워크플로우 파일 확인

`.github/workflows/deploy.yml` 파일이 이미 작성되어 있습니다.
이 파일이 하는 일:

```
main 브랜치에 push ──→ GitHub Actions 자동 실행
                        │
                        ├─ 1. 소스 코드 체크아웃
                        ├─ 2. AWS 인증 (Secrets 사용)
                        ├─ 3. ECR 로그인
                        ├─ 4. Docker 빌드 & ECR Push
                        ├─ 5. ECS 태스크 정의 업데이트
                        └─ 6. ECS 서비스 배포
```

---

## 9. Step 7: 배포 실행 및 확인하기

### 9-1. 코드 push로 자동 배포

```bash
# 코드 수정 후
git add .
git commit -m "feat: CI/CD 파이프라인 구축"
git push origin main
```

push하는 순간 GitHub Actions가 자동으로 실행됩니다!

### 9-2. 배포 진행 상황 확인

```
GitHub 저장소 → Actions 탭 → 실행 중인 워크플로우 클릭
```

각 단계별로 초록색 체크가 뜨면 성공:
```
✅ Checkout
✅ Configure AWS credentials
✅ Login to Amazon ECR
✅ Build, tag, and push image to Amazon ECR
✅ Fill in the new image ID in the ECS task definition
✅ Deploy Amazon ECS task definition
```

### 9-3. 배포된 API 접속

```bash
# ECS 태스크의 퍼블릭 IP 확인
aws ecs list-tasks --cluster mlops-cluster --service-name mlops-loan-service
aws ecs describe-tasks --cluster mlops-cluster --tasks <TASK_ARN>
# → "publicIp" 값 확인

# 또는 AWS 콘솔에서 확인:
# ECS → 클러스터 → mlops-cluster → 서비스 → Tasks 탭 → 태스크 클릭 → 퍼블릭 IP

# API 테스트
curl http://<PUBLIC_IP>:8000/health
curl http://<PUBLIC_IP>:8000/docs     # ← Swagger UI (브라우저로 접속)
```

### 9-4. 수동 배포 (workflow_dispatch)

코드 변경 없이도 배포를 다시 하고 싶다면:
```
GitHub 저장소 → Actions → Deploy to AWS ECS → Run workflow → Run workflow 버튼
```

---

## 10. 트러블슈팅 가이드

### 자주 발생하는 에러와 해결법

| 에러 | 원인 | 해결법 |
|------|------|--------|
| `Error: Cannot connect to the Docker daemon` | Docker Desktop이 꺼져 있음 | Docker Desktop 실행 |
| `Error: AccessDeniedException` | AWS IAM 권한 부족 | IAM 사용자/역할에 필요한 정책 추가 |
| `Error: Repository does not exist` | ECR 저장소가 없음 | `aws ecr create-repository` 실행 |
| `ECS service is not stable` | 컨테이너가 계속 재시작됨 | CloudWatch 로그 확인 |
| `CannotPullContainerError` | ECR 이미지를 못 가져옴 | `ecsTaskExecutionRole` 권한 확인 |
| `ResourceNotFoundException` | S3에 모델 파일이 없음 | `upload_models_to_s3.sh` 실행 |
| `Model load failed` | S3 접근 권한 없음 | `ecsTaskRole`에 S3 읽기 권한 추가 |

### CloudWatch에서 로그 확인하기

```bash
# 최근 로그 확인
aws logs tail /ecs/mlops-loan-api --follow --region ap-northeast-2

# 또는 AWS 콘솔:
# CloudWatch → 로그 그룹 → /ecs/mlops-loan-api → 최신 로그 스트림 클릭
```

### Docker 이미지 로컬 디버깅

```bash
# 컨테이너 안으로 들어가서 확인
docker run -it mlops-loan-api /bin/bash

# 안에서 확인
ls /app/           # 소스 코드 확인
ls /app/models/    # 모델 파일 확인 (비어 있으면 정상 - S3에서 받기 때문)
pip list           # 설치된 패키지 확인
```

---

## 11. 용어 사전

| 용어 | 설명 |
|------|------|
| **Docker** | 앱을 격리된 환경(컨테이너)에서 실행하게 해주는 도구. "내 PC에서는 되는데 서버에서 안 돼요" 문제를 해결 |
| **Docker Image** | 컨테이너를 만들기 위한 설계도. Dockerfile로 생성 |
| **Docker Container** | 이미지를 실행한 것. 실제로 돌아가는 앱 |
| **Dockerfile** | Docker 이미지를 빌드하는 레시피 파일 |
| **.dockerignore** | Docker 빌드 시 제외할 파일 목록 (`.gitignore`와 비슷) |
| **GitHub Actions** | GitHub에서 제공하는 자동화 도구. 코드 push 시 자동으로 빌드/배포 |
| **Workflow** | GitHub Actions에서 실행되는 자동화 작업 단위 (.yml 파일) |
| **ECR** | Elastic Container Registry. AWS의 Docker 이미지 저장소 (Docker Hub의 AWS 버전) |
| **ECS** | Elastic Container Service. AWS에서 Docker 컨테이너를 실행/관리하는 서비스 |
| **Fargate** | ECS의 서버리스 모드. 서버를 직접 관리하지 않아도 됨 |
| **태스크 정의** | ECS에게 "어떤 이미지로, 어떤 설정으로 컨테이너를 실행할지" 알려주는 설정 |
| **S3** | Simple Storage Service. AWS의 파일 저장 서비스 (무제한 용량) |
| **IAM** | Identity and Access Management. AWS의 권한/보안 관리 서비스 |
| **SSM Parameter Store** | 비밀번호나 API 키를 안전하게 저장하는 서비스 |
| **CloudWatch** | AWS의 모니터링/로그 서비스. 컨테이너 로그를 여기서 확인 |
| **VPC** | Virtual Private Cloud. AWS 안의 가상 네트워크 |
| **서브넷** | VPC 안의 작은 네트워크 구역 |
| **보안 그룹** | AWS의 방화벽. 어떤 포트를 열고 닫을지 설정 |

---

## 파일별 역할 요약

```
프로젝트 루트
│
├── Dockerfile                      ← Docker: 이미지 빌드 레시피
├── .dockerignore                   ← Docker: 빌드에서 제외할 파일
├── requirements.txt                ← Python: 필요한 패키지 목록
├── .env                            ← 로컬 개발용 환경 변수 (Git 제외)
│
├── app/
│   ├── __init__.py                 ← Python 패키지 인식용
│   ├── main.py                     ← FastAPI 앱 (엔드포인트 정의)
│   ├── model.py                    ← ML 모델 로드 + S3 다운로드
│   ├── schemas.py                  ← Pydantic 요청/응답 스키마
│   └── gemini_client.py            ← OpenAI API 클라이언트
│
├── .github/workflows/
│   └── deploy.yml                  ← GitHub Actions: CI/CD 파이프라인
│
├── .aws/
│   └── task-definition.json        ← ECS: 태스크(컨테이너) 설정
│
└── scripts/
    └── upload_models_to_s3.sh      ← S3에 모델/데이터 업로드 스크립트
```

---

## 빠른 시작 체크리스트

- [ ] Docker Desktop 설치 및 실행
- [ ] AWS CLI 설치 및 `aws configure` 완료
- [ ] AWS IAM 사용자 생성 (Access Key 발급)
- [ ] AWS IAM 역할 생성 (ecsTaskExecutionRole, ecsTaskRole)
- [ ] S3 버킷 생성 및 모델/데이터 업로드
- [ ] ECR 저장소 생성
- [ ] task-definition.json에서 `<AWS_ACCOUNT_ID>` 수정
- [ ] CloudWatch 로그 그룹 생성
- [ ] ECS 클러스터 생성
- [ ] 보안 그룹 생성 (포트 8000 열기)
- [ ] ECS 서비스 생성
- [ ] GitHub 저장소 생성 및 코드 push
- [ ] GitHub Secrets 등록 (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
- [ ] git push → Actions 탭에서 배포 성공 확인
- [ ] 퍼블릭 IP:8000/docs 접속 확인
