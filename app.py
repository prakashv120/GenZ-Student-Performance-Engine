import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="Gen-Z Student Engine", layout="wide")

# ---------- LOAD MODEL ----------
model = joblib.load("student_model.pkl")

# ---------- SIDEBAR ----------
st.sidebar.header("Step 1: Upload File")
file = st.sidebar.file_uploader("Upload student_behavior.csv", type=["csv"])
st.sidebar.info("Please use student_behavior.csv")

# ---------- HEADER ----------
st.title("🎓 Gen-Z Student Performance Intelligence Engine")
st.write(
    "AI-powered dashboard to identify at-risk students using attendance, study habits, "
    "internal scores, and lifestyle behavior."
)

# ---------- AFTER UPLOAD ----------
if file is not None:
    df = pd.read_csv(file)

    X = df.drop("result", axis=1, errors="ignore")
    preds = model.predict(X)
    df["Predicted_Result"] = preds

    def risk(x):
        return {
            "Excellent": "Low Risk",
            "Average": "Medium Risk",
            "Poor": "High Risk",
        }[x]

    df["Risk_Level"] = df["Predicted_Result"].apply(risk)

    # ---------- KPI CARDS ----------
    st.subheader("📊 Class Overview")

    total = len(df)
    high = (df["Risk_Level"] == "High Risk").sum()
    medium = (df["Risk_Level"] == "Medium Risk").sum()
    low = (df["Risk_Level"] == "Low Risk").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", total)
    c2.metric("High Risk", high)
    c3.metric("Medium Risk", medium)
    c4.metric("Low Risk", low)

    st.markdown("---")

    # ---------- GAUGE ----------
    st.subheader("🚦 Overall Class Risk Gauge")

    risk_score = int((high / total) * 100)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={'text': "Class Risk Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 40], 'color': "green"},
                {'range': [40, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"},
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ---------- DISTRIBUTION ----------
    st.subheader("📈 Performance Distribution")

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        sns.countplot(x="Predicted_Result", data=df, ax=ax1)
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        sns.countplot(x="Risk_Level", data=df, ax=ax2)
        st.pyplot(fig2)

    st.markdown("---")

    # ---------- BEHAVIOR ANALYSIS ----------
    st.subheader("📉 Behavior Analysis")

    col3, col4 = st.columns(2)

    with col3:
        fig3, ax3 = plt.subplots()
        sns.boxplot(x="Predicted_Result", y="attendance", data=df, ax=ax3)
        st.pyplot(fig3)

    with col4:
        fig4, ax4 = plt.subplots()
        sns.boxplot(x="Predicted_Result", y="phone_usage", data=df, ax=ax4)
        st.pyplot(fig4)

    st.markdown("---")

    # ---------- TABLE ----------
    st.subheader("🧾 Student Prediction Report")
    st.dataframe(df)

    st.success("✅ Full AI analysis completed!")
