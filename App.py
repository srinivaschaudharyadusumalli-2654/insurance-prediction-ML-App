import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import streamlit as st


# =========================================================
# WEB PAGE CODE
# =========================================================

st.title("HEALTH INSURANCE PREDICTION")

img_url = "https://navi.com/blog/wp-content/uploads/2021/12/best-health-insurance-plans.jpg"

st.image(img_url)


# =========================================================
# STEP 2: LOAD INSURANCE DATA
# =========================================================

url = "https://raw.githubusercontent.com/ankitmisk/UIT-data/refs/heads/main/Insurance.csv"

df = pd.read_csv(url)


# =========================================================
# STEP 3: DATA PREPROCESSING
# =========================================================

# Remove Customer_ID column
df = df.drop("Customer_ID", axis=1)

# Convert Yes/No into 0/1
df["Previous_Insurance"] = df["Previous_Insurance"].map({
    "No": 0,
    "Yes": 1
})

df["Insurance_Bought"] = df["Insurance_Bought"].map({
    "No": 0,
    "Yes": 1
})


# =========================================================
# STEP 4: DIVIDE FEATURES AND TARGET
# =========================================================

X = df.iloc[:, :-1]

y = df.iloc[:, -1]


# =========================================================
# STEP 5: TRAINING AND TESTING DATA
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42,
    test_size=0.3
)


# =========================================================
# STEP 6: TRAIN THE MODEL
# =========================================================

model = LogisticRegression()

model.fit(X_train, y_train)


# =========================================================
# STEP 7: SHOW DATA SAMPLE
# =========================================================

st.write("Insurance Data Sample")

st.write(df.head())


# =========================================================
# SIDEBAR FOR USER INPUT
# =========================================================

st.sidebar.title("Fill Customer Details")

st.sidebar.image(img_url)


# =========================================================
# GET USER INPUT
# =========================================================

all_ans = []

for col_name in X.columns:

    min_v = int(X[col_name].min())
    max_v = int(X[col_name].max())

    if col_name != "Previous_Insurance":

        value = st.sidebar.slider(
            f"Select value for {col_name}",
            min_value=min_v,
            max_value=max_v,
            value=min_v
        )

    else:

        value = st.sidebar.number_input(
            f"Select value for {col_name} (0: No, 1: Yes)",
            min_value=0,
            max_value=1,
            value=0,
            step=1
        )

    all_ans.append(value)


# =========================================================
# CREATE USER DATAFRAME
# =========================================================

user_df = pd.DataFrame(
    [all_ans],
    columns=X.columns
)

st.write("Customer Details")

st.write(user_df)


# =========================================================
# STEP 8: MODEL EVALUATION
# =========================================================

training_score = model.score(X_train, y_train)

testing_score = model.score(X_test, y_test)

st.write("Training Score:", training_score)

st.write("Testing Score:", testing_score)


# =========================================================
# STEP 9: PREDICTION
# =========================================================

if st.button("Click to Predict"):

    with st.spinner("Predicting..."):

        time.sleep(2)

        final_ans = model.predict(user_df)[0]

        if final_ans == 0:

            st.info("Customer will NOT buy the Insurance")

        else:

            st.success("Customer will BUY the Insurance")
