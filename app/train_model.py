import pandas as pd

df = pd.read_csv("data/diseases.csv")  # Loads csv

df['Symptoms'] = df['Symptoms'].str.lower().str.split(r',\s*')     # Convert symptoms into lowercase and create a list of symptoms

print(df.head())

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer() 

X = mlb.fit_transform(df['Symptoms'])
y = df['Disease']

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
import joblib 

model = MultinomialNB()
score = cross_val_score(model, X, y, cv=5)

print("Cross Validated Accuracy : ", score.mean()) 

model.fit(X, y)

joblib.dump(model, "models/disease_model.joblib")
joblib.dump(mlb, "models/symptom_encoder.joblib")