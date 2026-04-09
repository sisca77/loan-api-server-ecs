name: Deploy to EC2 (Docker)

# on:
#   pull request:
#    branches: [main]

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - name: Deploy to EC2 via SSH
        uses: appleboy/ssh-action@v1
        with:
          host: ${{ secrets.EC2_HOST }}
          username: ${{ secrets.EC2_USERNAME }}
          key: ${{ secrets.EC2_SSH_KEY }}
          script: |
            # Docker 설치 확인 및 설치 (Ubuntu용)
            if ! command -v docker &> /dev/null; then
              echo "Docker 미설치 - 설치 시작"
              sudo apt-get update -y
              sudo apt-get install -y docker.io
              sudo systemctl start docker
              sudo systemctl enable docker
              sudo usermod -aG docker $USER
            fi

            cd /home/ubuntu/mlops_edu_main

            # 코드 최신화 (강제 덮어쓰기로 충돌 방지)
            git fetch https://${{ secrets.GIT_TOKEN }}@github.com/sisca77/mlops_edu_main.git main
            git reset --hard FETCH_HEAD

            # Docker 이미지 빌드
            sudo docker build -t mlops_edu_main .

            # 기존 컨테이너 중지 및 삭제
            sudo docker stop mlops_edu_main || true
            sudo docker rm mlops_edu_main || true

            fuser -k 8000/tcp || true
            sleep 2

            # 새 컨테이너 실행
            sudo docker run -d --name mlops_edu_main -p 8000:8000 mlops_edu_main
