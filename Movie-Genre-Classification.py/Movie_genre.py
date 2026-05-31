import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report
)


df = pd.read_csv("train_data.txt", sep=":::", engine="python")


df.columns = [
    "ID",
    "TITLE",
    "GENRE",
    "DESCRIPTION"
]


df.dropna(inplace=True)

print("Dataset Shape:", df.shape)


X = df["DESCRIPTION"]
y = df["GENRE"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))