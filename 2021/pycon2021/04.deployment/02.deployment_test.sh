curl http://{enviroment-name}.{other_values}.ap-northeast-2.elasticbeanstalk.com/invocations -H 'Content-Type: application/json' -d '{
    "columns": ["a", "b", "c"],
    "data": [[1, 2, 3], [4, 5, 6]]
}'
