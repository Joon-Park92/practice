HOST="http://localhost:8080"


curl "$HOST/invocations" -H 'Content-Type: application/json' -d '{
    "columns": ["a", "b", "c"],
    "data": [[1, 2, 3], [4, 5, 6]]
}'
