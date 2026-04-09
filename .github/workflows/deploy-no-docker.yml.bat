name: Deploy to EC2 (No Docker)

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Deploy to EC2 via SSH
        uses: appleboy/ssh-action@v1
        with:
          host: ${{ secrets.EC2_HOST }}
          username: ${{ secrets.EC2_USERNAME }}
          key: ${{ secrets.EC2_SSH_KEY }}
          script_stop: false
          script: |
            cd ~/loan-api-server

            # 코드 최신화 (Private 레포 인증)
            git pull https://${{ secrets.GIT_TOKEN }}@github.com/blockenters/loan-api-server.git main

            # miniconda 환경 활성화 및 패키지 설치
            source ~/miniconda3/etc/profile.d/conda.sh
            conda activate mlops
            pip install --no-cache-dir --no-build-isolation -r requirements.txt

            # 기존 서버 종료 (8000 포트 사용 프로세스)
            fuser -k 8000/tcp || true
            sleep 2

            # 서버 시작
            uvicorn app.main:app --host 0.0.0.0 --port 8000 > ~/loan-api.log 2>&1 &
