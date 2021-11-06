from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

# Method 1
pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer()),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression()),
    ]
)

# Method 2
make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression())
