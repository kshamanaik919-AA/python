import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# --------------------------------
# PAGE SETTINGS
# --------------------------------

st.set_page_config(page_title="FinGram", layout="wide")

st.title("💰 FinGram")
st.subheader("Smart Financial Behavior Prediction System")

# --------------------------------
# LOAD DATASET
# --------------------------------

data = pd.read_csv("expenses.csv")

# --------------------------------
# MACHINE LEARNING MODELS
# --------------------------------

# Linear Regression Model
X = data[["Month"]]
y = data["Expense"]

lr_model = LinearRegression()
lr_model.fit(X, y)

# KMeans Model
cluster_data = data[["Income", "Expense"]]

kmeans = KMeans(n_clusters=3, random_state=42)
data["Cluster"] = kmeans.fit_predict(cluster_data)

# --------------------------------
# USER INPUT SECTION
# --------------------------------

st.write("## Enter Financial Details")

income = st.number_input("Monthly Income", min_value=1000)

food = st.number_input("Food Expense", min_value=0)

travel = st.number_input("Travel Expense", min_value=0)

shopping = st.number_input("Shopping Expense", min_value=0)

bills = st.number_input("Bills Expense", min_value=0)

entertainment = st.number_input("Entertainment Expense", min_value=0)

future_month = st.number_input(
    "Predict Expense for Future Month",
    min_value=13,
    max_value=24
)

# --------------------------------
# PREDICT BUTTON
# --------------------------------

if st.button("Analyze Financial Behavior"):

    # Total Expense
    total_expense = (
        food +
        travel +
        shopping +
        bills +
        entertainment
    )

    # Savings
    savings = income - total_expense

    # Future Prediction
    future_prediction = lr_model.predict([[future_month]])
    # Spending Ratio
    expense_ratio = (total_expense / income) * 100

    # Spending Type Logic
    if expense_ratio < 40:
        spending_type = "Low Spender"

    elif expense_ratio < 70:
        spending_type = "Medium Spender"

    else:
        spending_type = "High Spender"

    # --------------------------------
    # RESULTS
    # --------------------------------

    st.success("Financial Analysis Completed")

    st.write("## Prediction Results")

    st.metric("Total Expense", f"₹{total_expense}")

    st.metric("Estimated Savings", f"₹{savings}")

    st.metric(
        "Predicted Future Expense",
        f"₹{future_prediction[0]:.2f}"
    )

    st.metric("Spending Type", spending_type)

    # --------------------------------
    # SMART SUGGESTIONS
    # --------------------------------

    st.write("## Financial Suggestions")

    if savings < 5000:
        st.warning(
            "Your savings are low. Try reducing shopping or entertainment expenses."
        )

    else:
        st.success(
            "Good financial management! Your savings are healthy."
        )

    # --------------------------------
    # VISUALIZATION
    # --------------------------------

    expense_data = pd.DataFrame({
        "Category": [
            "Food",
            "Travel",
            "Shopping",
            "Bills",
            "Entertainment"
        ],
        "Expense": [
            food,
            travel,
            shopping,
            bills,
            entertainment
        ]
    })

    fig = px.pie(
        expense_data,
        names="Category",
        values="Expense",
        title="Expense Distribution"
    )

    st.plotly_chart(fig)

# --------------------------------
# FOOTER
# --------------------------------

st.info(
    "FinGram uses Machine Learning algorithms like Linear Regression "
    "and K-Means Clustering to analyze user financial behavior."
)