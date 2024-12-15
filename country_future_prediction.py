import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Set up the Streamlit app with a colorful theme
st.set_page_config(page_title="Analytics and Prediction of Country", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for styling the app
st.markdown(
    """
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background: linear-gradient(to right, #4e54c8, #8f94fb); /* Gradient background */
    }
    .title {
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0;
        color: #fff;
    }
    .subtitle {
        text-align: center;
        font-size: 1.5rem;
        font-style: italic;
        margin-top: 0;
        color: #f4f4f4;
    }
    .sidebar .sidebar-content {
        background: #2c3e50;
        color: white;
        padding: 20px;
        border-radius: 5px;
    }
    .stDataFrame {
        background: white;
        border: 1px solid #ddd;
        border-radius: 5px;
        margin: 20px 0;
        padding: 10px;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        font-size: 1rem;
        padding: 10px 20px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #2980b9; /* Hover effect */
    }
    .metric-container {
        background: white;
        padding: 20px;
        margin-bottom: 20px;
        border-radius: 5px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-container h4 {
        color: #2c3e50;
    }
    .icon-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
        width: 100px;
        border-radius: 50%;
        background-color: #e74c3c;
        color: white;
        font-size: 2rem;
        margin: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main title
st.markdown("<div class='title'>📊 Future Analytics and Prediction for Country</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Upload your dataset, analyze indicators, and visualize predictions.</div>",
    unsafe_allow_html=True,

)

# Sidebar for page selection
page = st.sidebar.selectbox("Choose an Analysis", ["Technology Analytics", "Socio-Cultural Analytics", "Economic Analytics", "CO2 Emissions Prediction", "Overall Development Index Prediction"])

# Function to calculate the Development Index (DI)
def calculate_development_index(mae, mse, r2):
    max_mae = max(mae, 1)  # Prevent division by zero
    max_mse = max(mse, 1)  # Prevent division by zero
    normalized_mae = 1 - (mae / max_mae)
    normalized_mse = 1 - (mse / max_mse)
    DI = (normalized_mae * 0.3) + (normalized_mse * 0.3) + (r2 * 0.4)
    return DI

# Generic machine learning model function
def run_machine_learning_analysis(page_name):
    st.subheader(f"🔍 {page_name} Analysis")

    data_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    if data_file:
        dataset = pd.read_csv(data_file)
        st.subheader("📜 Dataset Preview")
        st.dataframe(dataset.head(), use_container_width=True)

        all_columns = dataset.columns.tolist()

        # Selecting target and features manually
        target_column = st.sidebar.selectbox("Select Target Column", options=all_columns, index=0)  # No default, first is selected
        feature_columns = st.sidebar.multiselect("Select Feature Columns", options=all_columns)  # No default features

        if target_column in feature_columns:
            feature_columns.remove(target_column)  # Ensure target column is not in features

        if target_column and feature_columns:
            missing_columns = [col for col in [target_column] + feature_columns if col not in dataset.columns]
            if missing_columns:
                st.error(f"Missing columns: {', '.join(missing_columns)}")
            else:
                X = dataset[feature_columns]
                y = dataset[target_column]

                # Split data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Standardize the data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Train the Random Forest model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train_scaled, y_train)

                # Model evaluation
                y_pred = model.predict(X_test_scaled)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # Calculate the Development Index (DI)
                development_index = calculate_development_index(mae, mse, r2)

                col1, col2, col3 = st.columns(3)
                col1.metric(label="MAE", value=f"{mae:.2f}")
                col2.metric(label="MSE", value=f"{mse:.2f}")
                col3.metric(label="R²", value=f"{r2:.2f}")

                st.subheader("📈 Development Index")
                st.metric(label="Development Index (DI)", value=f"{development_index:.2f}", delta=f"{development_index - 1:.2f}")

                # Feature Importance Calculation
                st.subheader("📊 Feature Importance")
                feature_importances = model.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': feature_columns,
                    'Importance': feature_importances
                })

                # Sort features by importance
                importance_df = importance_df.sort_values(by='Importance', ascending=False)

                # Plot Feature Importances
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax, palette='viridis')
                ax.set_title(f"Feature Importance for {page_name}")
                st.pyplot(fig)

                # Interactive Predictions vs Actual (using Plotly)
                st.subheader("📊 Predictions vs Actual Values")
                fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual Values", "y": "Predicted Values"}, title=f"Actual vs Predicted {page_name} Values")
                fig.update_traces(marker=dict(color='purple', opacity=0.6))
                st.plotly_chart(fig)

                predictions = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
                csv = predictions.to_csv(index=False)
                st.download_button(f"Download {page_name} Predictions", csv, f"{page_name.lower().replace(' ', '_')}_predictions.csv", "text/csv")

# Run the selected analysis
if page == "Technology Analytics":
    run_machine_learning_analysis("Technology Analytics")
elif page == "Socio-Cultural Analytics":
    run_machine_learning_analysis("Socio-Cultural Analytics")
elif page == "Economic Analytics":
    run_machine_learning_analysis("Economic Analytics")
elif page == "CO2 Emissions Prediction":
    run_machine_learning_analysis("CO2 Emissions Prediction")
elif page == "Overall Development Index Prediction":
    run_machine_learning_analysis("Overall Development Index Prediction")
