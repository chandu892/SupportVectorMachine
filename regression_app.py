import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None

if "df_clean" not in st.session_state:
    st.session_state.df_clean = None

if "df_model" not in st.session_state:
    st.session_state.df_model = None

# =====================================================
# FOLDER SETUP
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEANED_DIR = os.path.join(BASE_DIR, "data", "cleaned")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEANED_DIR, exist_ok=True)

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="End-to-End SVR", layout="wide")
st.title("📈 End-to-End Support Vector Regression Platform")

# =====================================================
# SIDEBAR – SVR HYPERPARAMETERS
# =====================================================
st.sidebar.header("SVR Hyperparameters")

kernel = st.sidebar.selectbox(
    "Kernel", ["linear", "poly", "rbf", "sigmoid"]
)

C = st.sidebar.slider(
    "C (Regularization)", 0.01, 100.0, 1.0
)

gamma = st.sidebar.selectbox(
    "Gamma", ["scale", "auto"]
)

epsilon = st.sidebar.slider(
    "Epsilon (ε-tube)", 0.01, 1.0, 0.1
)

# =====================================================
# STEP 1: DATA INGESTION
# =====================================================
st.header("Step 1: Data Ingestion")

option = st.radio(
    "Choose data source",
    ["Download Boston Housing Dataset", "Upload CSV"]
)

if option == "Download Boston Housing Dataset":
    if st.button("Download Dataset"):
        url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
        df = pd.read_csv(url)
        path = os.path.join(RAW_DIR, "boston_housing.csv")
        df.to_csv(path, index=False)
        st.session_state.df_raw = df
        st.success("Boston Housing dataset downloaded")

if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        path = os.path.join(RAW_DIR, uploaded_file.name)
        df.to_csv(path, index=False)
        st.session_state.df_raw = df
        st.success("File uploaded successfully")

# =====================================================
# STEP 2: EDA
# =====================================================
if st.session_state.df_raw is not None:
    st.header("Step 2: Exploratory Data Analysis")

    df = st.session_state.df_raw

    st.dataframe(df.head())
    st.write("Shape:", df.shape)
    st.write("Missing Values:", df.isnull().sum())

    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# =====================================================
# STEP 3: DATA CLEANING
# =====================================================
if st.session_state.df_raw is not None:
    st.header("Step 3: Data Cleaning")

    strategy = st.selectbox(
        "Missing Value Strategy",
        ["Mean", "Median", "Drop Rows"]
    )

    df_clean = st.session_state.df_raw.copy()

    if strategy == "Drop Rows":
        df_clean.dropna(inplace=True)
    else:
        for col in df_clean.select_dtypes(include=np.number).columns:
            if strategy == "Mean":
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)

    st.session_state.df_clean = df_clean
    st.success("Data cleaning completed")

# =====================================================
# STEP 4: SAVE CLEANED DATA
# =====================================================
st.header("Step 4: Save Cleaned Dataset")

if st.button("Save Cleaned Data"):
    if st.session_state.df_clean is None:
        st.error("No cleaned data available")
    else:
        filename = f"cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = os.path.join(CLEANED_DIR, filename)
        st.session_state.df_clean.to_csv(path, index=False)
        st.success("Cleaned dataset saved")
        st.info(path)

# =====================================================
# STEP 5: LOAD CLEANED DATASET
# =====================================================
st.header("Step 5: Load Cleaned Dataset")

clean_files = os.listdir(CLEANED_DIR)

if clean_files:
    selected = st.selectbox("Select Dataset", clean_files)
    if st.button("Load Dataset"):
        st.session_state.df_model = pd.read_csv(
            os.path.join(CLEANED_DIR, selected)
        )
        st.success(f"Loaded dataset: {selected}")

if st.session_state.df_model is not None:
    st.dataframe(st.session_state.df_model.head())

# =====================================================
# STEP 6: TRAIN SVR MODEL
# =====================================================
if st.session_state.df_model is not None:
    st.header("Step 6: Train SVR Model")

    df_model = st.session_state.df_model

    target = st.selectbox(
        "Select Target Column",
        df_model.select_dtypes(include=np.number).columns
    )

    y = df_model[target]
    X = df_model.drop(columns=[target])
    X = X.select_dtypes(include=np.number)

    if X.empty:
        st.error("No numeric features available")
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.25, random_state=42
        )

        model = SVR(
            kernel=kernel,
            C=C,
            gamma=gamma,
            epsilon=epsilon
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        st.success(f"R² Score: {r2:.3f}")
        st.success(f"RMSE: {rmse:.3f}")
        st.success(f"MAE: {mae:.3f}")

        # ========================
        # VISUALIZATIONS
        # ========================
        fig1, ax1 = plt.subplots()
        ax1.scatter(y_test, y_pred)
        ax1.set_xlabel("Actual Values")
        ax1.set_ylabel("Predicted Values")
        ax1.set_title("Actual vs Predicted")
        st.pyplot(fig1)

        residuals = y_test - y_pred
        fig2, ax2 = plt.subplots()
        ax2.scatter(y_pred, residuals)
        ax2.axhline(0)
        ax2.set_xlabel("Predicted Values")
        ax2.set_ylabel("Residuals")
        ax2.set_title("Residual Plot")
        st.pyplot(fig2)

else:
    st.info("Load a cleaned dataset in Step 5 to enable training.")
