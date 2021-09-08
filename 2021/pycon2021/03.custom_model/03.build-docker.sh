RUN_ID=$1
MODEL_DIR="model"
MLFLOW_TRACKING_URI="http://localhost:5000"
BUILD_NAME="custom_img"

jsonBody=$(curl -s "${MLFLOW_TRACKING_URI}/api/2.0/mlflow/runs/get?run_id=${RUN_ID}")
s3Url=$(echo "${jsonBody}" | sed -n 's/.*"artifact_uri": "\(.*\)",/\1/p')

mkdir model
aws s3 cp --recursive "${s3Url}/${MODEL_DIR}" "./model/"
docker build -t "${BUILD_NAME}" .
rm -rf model
