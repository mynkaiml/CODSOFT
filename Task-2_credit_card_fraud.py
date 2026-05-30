import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


train_df = pd.read_csv("fraudTrain.csv")
test_df = pd.read_csv("fraudTest.csv")

print("Training Dataset Shape:", train_df.shape)
print("Testing Dataset Shape:", test_df.shape)


features = [
    "amt",
    "category",
    "gender",
    "city_pop"
]

target = "is_fraud"


category_encoder = LabelEncoder()
gender_encoder = LabelEncoder()

train_df["category"] = category_encoder.fit_transform(
    train_df["category"]
)

test_df["category"] = category_encoder.transform(
    test_df["category"]
)

train_df["gender"] = gender_encoder.fit_transform(
    train_df["gender"]
)

test_df["gender"] = gender_encoder.transform(
    test_df["gender"]
)


X_train = train_df[features].copy()
y_train = train_df[target]

X_test = test_df[features].copy()
y_test = test_df[target]


scaler = StandardScaler()

X_train[["amt", "city_pop"]] = scaler.fit_transform(
    X_train[["amt", "city_pop"]]
)

X_test[["amt", "city_pop"]] = scaler.transform(
    X_test[["amt", "city_pop"]]
)


print("\nFraud Distribution:")
print(y_train.value_counts())


model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print("\nModel Performance")
print("-" * 40)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))