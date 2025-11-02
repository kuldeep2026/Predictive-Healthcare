import pandas as pd

df = pd.read_csv("data/diseases.csv")  # Loads csv

df['Symptoms'] = df['Symptoms'].str.lower().str.split(r',\s*')     # Convert symptoms into lowercase and create a list of symptoms

print(df.head())

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer() 

X = mlb.fit_transform(df['Symptoms'])
y = df['Disease']

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split,cross_val_score
import joblib 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
scores = cross_val_score(model, X_train, y_train, cv=5)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)
print("Cross-Validation Scores:", scores)
print("Cross-Validation Accuracy:", scores.mean())

joblib.dump(model, "models/disease_model.joblib")
joblib.dump(mlb, "models/symptom_encoder.joblib")