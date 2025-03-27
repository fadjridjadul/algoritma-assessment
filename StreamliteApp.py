import os
os.system("pip install streamlit numpy pandas tensorflow joblib Pillow scikit-learn")

import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.title("Algoritma Analytics Data Analysis")
st.write(f"**I use chatgpt help to answer questions and make codes**")


# Load and clean dataset
df = pd.read_csv("analytics.csv", skiprows=6).iloc[:11]
df['Language'] = df['Language'].str.split('-').str[0].apply(lambda x: x if x in ['en', 'id', 'th'] else 'missing')
df['Goal Conversion Rate'] = df['Goal Conversion Rate'].str.rstrip('%').astype(float)

st.write("### Dataset Preview:")
st.dataframe(df)

# Find the language with the highest "Pages / Session" average
highest_language = df.groupby('Language')['Pages / Session'].mean().idxmax()
st.write(f"**Language with the highest average Pages / Session:** {highest_language}")

# Model A: Predict "Goal Conversion Rate" using "Pages / Session"
X_A, y = df[['Pages / Session']], df['Goal Conversion Rate']
model_A = LinearRegression().fit(X_A, y)
r2_A = r2_score(y, model_A.predict(X_A))
beta0 = model_A.intercept_

st.write(f"**Model A - R²:** {round(r2_A, 3)}")
st.write(f"**Intercept (β0):** {round(beta0, 3)}")

# Model B: Add "Language" as an additional predictor
X_B = pd.concat([X_A, pd.get_dummies(df['Language'], drop_first=True)], axis=1)
model_B = LinearRegression().fit(X_B, y)
r2_B = r2_score(y, model_B.predict(X_B))

# Adjusted R² calculation
adj_r2_A = 1 - (1 - r2_A) * (len(y) - 1) / max((len(y) - X_A.shape[1] - 1), 1)
adj_r2_B = 1 - (1 - r2_B) * (len(y) - 1) / max((len(y) - X_B.shape[1] - 1), 1)

# Determine the best model
if r2_B > r2_A and adj_r2_B > adj_r2_A:
    result = "Model B is better (Higher R² and Adjusted R²)."
elif r2_A > r2_B and adj_r2_A > adj_r2_B:
    result = "Model A is better (Higher R² and Adjusted R²)."
elif r2_A > r2_B:
    result = "Model A has a higher R² but lower Adjusted R²."
else:
    result = "Model B has a higher R² but lower Adjusted R²."

st.write(f"**Model A - R²:** {round(r2_A, 3)}, **Adjusted R²:** {round(adj_r2_A, 3)}")
st.write(f"**Model B - R²:** {round(r2_B, 3)}, **Adjusted R²:** {round(adj_r2_B, 3)}")
st.write(f"**Conclusion:** {result}")
