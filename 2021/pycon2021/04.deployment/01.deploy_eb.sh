# MLFLOW RESOURCE
EXPERIMENT_NUMBER=1
RUN_ID="84843ed82bf24c6792344b68b60f2d69"

MODEL_URI="s3://pycon2021/mlflow/$EXPERIMENT_NUMBER/$RUN_ID/artifacts/model"
IMAGE_NAME="pydeployment"

# BUILD & PUSH IMAGE
mlflow models build-docker -m $MODEL_URI -n $IMAGE_NAME

# ECR RESOURCE
AWS_ACCOUNT_ID=""
REGION="ap-northeast-2"
ECR_REPO_NAME="pycon2021/pydeployment"
ECR_IMAGE="$AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO_NAME"

DEPLOY_APP="pycon-app"
DEPLOY_ENV="pycon-env"

# CREATE ECR REPOSITORY
aws ecr describe-repositories --repository-names ${ECR_REPO_NAME} || aws ecr create-repository --repository-name ${ECR_REPO_NAME}

# PUSH TO ECR
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com
docker tag ${IMAGE_NAME} ${ECR_IMAGE}
docker push ${ECR_IMAGE}

# ADD EB SETTINGS
cat << EOL > Dockerrun.aws.json
{
  "AWSEBDockerrunVersion": "1",
  "Logging": "/tmp/mlflow",
  "Image": {
    "Name": "${ECR_IMAGE}",
    "Update": "true"
  },
  "Ports": [
    {
        "ContainerPort": 8080,
        "HostPort": 80
    }
  ]
}
EOL

# DEPLOY TO ElasticBeanstalk
git add Dockerrun.aws.json
eb init -r ${REGION} ${DEPLOY_APP}
eb use ${DEPLOY_ENV}
eb deploy --staged --timeout=30 "${DEPLOY_ENV}"
