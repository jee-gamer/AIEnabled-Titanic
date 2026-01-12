import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("data/train_preprocessed.csv")

target = "Survived"
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "HasPrefix", "TicketNumber", "TicketIsLine", "TicketLength", "Fare", "Embarked"]
X = df[features].copy()
y = df[target].copy()

categorical = ["Sex", "Pclass", "Embarked", "HasPrefix", "TicketIsLine"]
numeric = ["Age", "SibSp", "Parch", "TicketNumber", "TicketLength", "Fare"]

cat_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

num_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=False)) # keeps sparse-friendly behavior
])

preprocess = ColumnTransformer(
    transformers=[
        ("cat", cat_tf, categorical),
        ("num", num_tf, numeric)
    ]
)

clf = LogisticRegression(max_iter=1000, n_jobs=None)

pipe = Pipeline(steps=[
    ("preprocessor", preprocess),
    ("model", clf)
])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pipe.fit(X_tr, y_tr)
pred = pipe.predict(X_te) # pred = predicted y
acc = accuracy_score(y_te, pred)
print(f"Accuracy: {acc:.3f}")
print(classification_report(y_te, pred))

joblib.dump(pipe, "titanic_model.pkl")
print("Saved model -> titanic_model.pkl")


