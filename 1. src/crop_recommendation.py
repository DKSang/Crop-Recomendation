import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from lazypredict.Supervised import LazyClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from ydata_profiling import ProfileReport

# Load data
data = pd.read_csv(r"C:\Users\LENOVO\PycharmProjects\data_machine\data_set\Crop_recommendation.csv")

# file=ProfileReport(data,title="crop_recomendation",explorative=True)
# file.to_file("crop_recomendation.html")

# Split features and target
X = data.drop(["label"], axis=1)
y = data["label"]

# Define numerical features
num_features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define preprocessing pipeline
num_preprocess = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Column transformer
transform = ColumnTransformer(transformers=[
    ("num", num_preprocess, num_features)
])

# Fit and transform the data
X_train_transformed = transform.fit_transform(X_train)
X_test_transformed = transform.transform(X_test)

# Step 1: Run LazyPredict to evaluate models
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train_transformed, X_test_transformed, y_train, y_test)

# Display top 5 models
print("\nTop 5 Models from LazyPredict:")
print(models.head(5))

# Step 2: Use a Pipeline for RandomizedSearchCV
best_model = Pipeline(steps=[
    ("preprocessor", transform),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Define parameter grid for RandomForestClassifier
param_grid = {
    "classifier__n_estimators": [100, 200, 300, 400, 500],
    "classifier__max_depth": [None, 10, 20, 30, 40],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__bootstrap": [True, False]
}

# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=best_model,
    param_distributions=param_grid,
    n_iter=20,
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Fit RandomizedSearchCV
random_search.fit(X_train, y_train)

# Predict and evaluate
y_pred = random_search.predict(X_test)
print("\nClassification Report for Best Model:")
print(classification_report(y_test, y_pred))

# Print best parameters and score
print("\nBest Parameters from RandomizedSearchCV:")
print(random_search.best_params_)
print("\nBest Cross-Validation Score:")
print(random_search.best_score_)
print("\nTest Set Score with Best Model:")
print(random_search.score(X_test, y_test))
