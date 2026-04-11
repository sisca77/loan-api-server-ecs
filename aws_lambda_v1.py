import json
import boto3
from datetime import datetime

# ECS 클라이언트
ecs_client = boto3.client('ecs', region_name='ap-northeast-2')

# === 본인 번호로 수정하세요 ===
CLUSTER_NAME = 'mlops-lab-cluster'
SERVICE_NAME = 'student-20-api-service'  # XX를 본인 번호로!
# ==============================

def lambda_handler(event, context):
    """
    S3 업로드 이벤트를 받아 ECS 서비스를 재시작합니다.
    새 태스크가 시작되면 FastAPI 서버가 기동하면서 boto3로 S3에서 최신 모델을 로드합니다.
    """
    print(f"Event: {json.dumps(event, indent=2)}")

    # S3 이벤트에서 업로드된 파일 정보 추출
    for record in event.get('Records', []):
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        print(f"  업로드된 파일: s3://{bucket}/{key}")

        if key == 'student-20/loan_pipeline.pkl':
            # pkl 파일 처리 로직 (다운로드, 로드 등)
            

            # ECS 서비스 업데이트 
            try:
                response = ecs_client.update_service(
                    cluster=CLUSTER_NAME,
                    service=SERVICE_NAME,
                    forceNewDeployment=True
                )

                deployment_id = response['service']['deployments'][0]['id']
                print(f"  ECS 재배포 시작!")
                print(f"  Cluster: {CLUSTER_NAME}")
                print(f"  Service: {SERVICE_NAME}")
                print(f"  Deployment ID: {deployment_id}")

                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'message': 'ECS service redeployment triggered',
                        'cluster': CLUSTER_NAME,
                        'service': SERVICE_NAME,
                        'deployment_id': deployment_id
                    })
                }
            except Exception as e:
                print(f"  오류 발생: {str(e)}")
                return {
                    'statusCode': 500,
                    'body': json.dumps({
                        'error': str(e)
                    })
                }
