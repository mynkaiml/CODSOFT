import pandas as pd
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv("spam.csv", encoding="latin-1")

df = df[["v1", "v2"]]

df.columns = ["label", "message"]


df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})


def clean_text(text):

    text = text.lower()


    cleaned = ""

    for char in text:
        if char not in string.punctuation:
            cleaned += char

    return cleaned



df["message"] = df["message"].apply(clean_text)


X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


vectorizer = TfidfVectorizer(stop_words="english")

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


model = MultinomialNB()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(round(accuracy * 100, 2), "%")


print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
