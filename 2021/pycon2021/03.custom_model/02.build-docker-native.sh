EXPERIMENT=1
RUN_ID="cadf9803786f48edafe0ea8d15c9f0d0"

mlflow models build-docker \
    --model-uri "s3://pycon2021/mlflow/$EXPERIMENT/$RUN_ID/artifacts/model"
