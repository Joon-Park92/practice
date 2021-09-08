EXPERIMENT=1
RUN_ID="e7e263793b5a463191c72bba8a84d57e"

mlflow models serve -m "s3://pycon2021/mlflow/$EXPERIMENT/$RUN_ID/artifacts/model" -p 8080
