import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Gen-Z Student Engine", layout="wide")

# ---------- LOAD MODEL ----------
model = joblib.load("student_model.pkl")


# ---------- DONUT GAUGE FUNCTION ----------
def donut_gauge(score):
    fig, ax = plt.subplots(figsize=(2.6, 2.6))

    color = "#2ecc71"
    if score > 70:
        color = "#e74c3c"
    elif score > 40:
        color = "#f39c12"

    ax.pie(
        [score, 100 - score],
        startangle=90,
        colors=[color, "#ecf0f1"],
        wedgeprops={'width': 0.35}
    )

    ax.text(0, 0, f"{score}%",
            ha='center', va='center',
            fontsize=20, fontweight='bold')

    ax.axis('equal')
    st.pyplot(fig, use_container_width=False)


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

    # ---------- DONUT GAUGE ----------
    st.subheader("🚦 Overall Class Risk Gauge")

    risk_score = int((high / total) * 100)

    colg1, colg2 = st.columns([1, 3])

    with colg1:
        donut_gauge(risk_score)

    with colg2:
        st.markdown(f"""
        ### Class Risk Score: **{risk_score}%**

        - 🟢 Low Risk: Good class performance  
        - 🟡 Medium Risk: Needs monitoring  
        - 🔴 High Risk: Immediate attention required
        """)

    st.markdown("---")

    # ---------- PERFORMANCE DISTRIBUTION ----------
    st.subheader("📈 Performance Distribution")

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(3.5, 2.2))
        sns.countplot(x="Predicted_Result", data=df, ax=ax1)
        ax1.set_title("Result Distribution")
        st.pyplot(fig1, use_container_width=False)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(3.5, 2.2))
        sns.countplot(x="Risk_Level", data=df, ax=ax2)
        ax2.set_title("Risk Distribution")
        st.pyplot(fig2, use_container_width=False)

    st.markdown("---")

    # ---------- BEHAVIOR ANALYSIS ----------
    st.subheader("📉 Behavior Analysis")

    col3, col4 = st.columns(2)

    with col3:
        fig3, ax3 = plt.subplots(figsize=(3.5, 2.2))
        sns.boxplot(x="Predicted_Result", y="attendance", data=df, ax=ax3)
        ax3.set_title("Attendance vs Result")
        st.pyplot(fig3, use_container_width=False)

    with col4:
        fig4, ax4 = plt.subplots(figsize=(3.5, 2.2))
        sns.boxplot(x="Predicted_Result", y="phone_usage", data=df, ax=ax4)
        ax4.set_title("Phone Usage vs Result")
        st.pyplot(fig4, use_container_width=False)

    st.markdown("---")

    # ---------- TABLE ----------
    st.subheader("🧾 Student Prediction Report")
    st.dataframe(df, use_container_width=True)

    st.success("✅ Full AI analysis completed!")
