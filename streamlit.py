import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
from fpdf import FPDF
import plotly.graph_objects as go
from PIL import Image
import io

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Credit Risk AI",
    page_icon="🏦",
    layout="wide"
)

# ================= UTILITIES =================
def clean_text(text):
    return (
        str(text)
        .replace("–", "-")
        .replace("—", "-")
        .replace("₹", "Rs.")
        .replace("“", '"')
        .replace("”", '"')
    )

def get_pdf_safe_image(uploaded_image=None):
    path = "pdf_profile.jpg"
    if uploaded_image:
        img = Image.open(uploaded_image).convert("RGB")
    else:
        img = Image.open("default_profile.png").convert("RGB")
    img.save(path, "JPEG")
    return path

# ================= MODEL =================
class CreditRiskANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

@st.cache_resource
def load_model():
    model = CreditRiskANN()
    model.load_state_dict(torch.load("credit_model.pt", map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

model = load_model()
scaler = load_scaler()

# ================= HEADER =================
st.title("🏦 Credit Risk Prediction System")
st.markdown(
    "AI-powered **Credit Default Prediction** using ANN + PyTorch "
    "with professional UI, risk visualization, and PDF reporting."
)
st.divider()

# ================= INPUT FORM =================
with st.form("credit_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        name = st.text_input("Customer Full Name", placeholder="e.g. Aman Verma")
        utilization = st.slider("Credit Usage (%)", 0, 100, 30)
        age = st.number_input("Age (Years)", 18, 100, 35)
        late_30 = st.number_input("Late Payments (30–59 days)", 0, 20, 0)

    with c2:
        debt_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.05)
        income = st.number_input("Monthly Income (₹)", 0, 300000, 50000)
        open_loans = st.number_input("Active Credit Accounts", 0, 50, 5)
        late_90 = st.number_input("Late Payments (90+ days)", 0, 20, 0)

    with c3:
        real_estate = st.number_input("Home / Property Loans", 0, 10, 1)
        late_60 = st.number_input("Late Payments (60–89 days)", 0, 20, 0)
        dependents = st.number_input("Number of Financial Dependents", 0, 10, 0)
        profile_img = st.file_uploader(
            "Upload Profile Image (Optional)",
            type=["jpg", "jpeg", "png"]
        )

    predict_btn = st.form_submit_button("🔍 Predict Credit Risk")
    
# ================= PREDICTION =================
if predict_btn:
    X = np.array([[
        utilization / 100,
        age,
        late_30,
        debt_ratio,
        income,
        open_loans,
        late_90,
        real_estate,
        late_60,
        dependents
    ]])

    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        prob = torch.sigmoid(model(X_tensor)).item()

    # ========= PREDICTION + GAUGE SIDE BY SIDE =========
    st.divider()
    st.subheader("🔎 Credit Risk Assessment")

    gauge_col, pred_col = st.columns([1, 1.2])

    with gauge_col:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 25], "color": "#2ecc71"},
                    {"range": [25, 50], "color": "#f1c40f"},
                    {"range": [50, 100], "color": "#e74c3c"},
                ],
                "bar": {"color": "#c0392b"}
            }
        ))
        fig.update_layout(height=250, margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with pred_col:
        st.markdown("### Prediction Result")
        if prob < 0.25:
            decision = "Low Risk - Loan Approved"
            st.success(f"{decision} ({prob:.2%})")
        elif prob < 0.50:
            decision = "Medium Risk - Manual Review Required"
            st.warning(f"{decision} ({prob:.2%})")
        else:
            decision = "High Risk - Loan Rejected"
            st.error(f"{decision} ({prob:.2%})")

    # ========= CUSTOMER PROFILE =========
    st.divider()
    st.subheader("👤 Customer Profile")

    img_col, info_col = st.columns([1, 3])

    if profile_img:
        img_col.image(profile_img, width=100)
    else:
        img_col.image("default_profile.png", width=100)

    info_col.markdown(f"""
    ### {name if name else "Unnamed Customer"}
    **Age:** {age}  
    **Monthly Income:** ₹{income:,.0f}  
    **Dependents:** {dependents}  
    **Debt-to-Income Ratio:** {debt_ratio:.2f}  
    **Credit Usage:** {utilization}%  
    **Active Accounts:** {open_loans}
    """)

    # ========= PDF GENERATION =========
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Credit Risk Assessment Report", ln=True, align="C")
    pdf.ln(8)

    # ================== CUSTOMER NAME + IMAGE ==================
    safe_name = clean_text(name if name else "Unnamed Customer")
    pdf_img = get_pdf_safe_image(profile_img)

    # Place image at current y
    start_y = pdf.get_y()
    pdf.image(pdf_img, x=10, y=start_y, w=30)

    # Text beside image
    pdf.set_xy(45, start_y)
    pdf.set_font("Arial", "B", 14)
    pdf.multi_cell(0, 8, f"Customer Name: {safe_name}\nRisk Probability: {prob:.2%}\nDecision: {decision}")

    # ================== MOVE BELOW IMAGE ==================
    # Calculate space to move cursor below image (image height + padding)
    pdf.set_y(start_y + 35)  # 30 for image + 5 for padding

    # ================== OTHER DETAILS ==================
    pdf.set_font("Arial", "", 12)
    for k, v in {
        "Age": age,
        "Monthly Income": f"Rs. {income}",
        "Dependents": dependents,
        "Debt Ratio": debt_ratio,
        "Credit Usage (%)": utilization,
        "Active Accounts": open_loans
    }.items():
        pdf.cell(0, 8, clean_text(f"{k}: {v}"), ln=True)

    # ================== CONVERT TO BYTES ==================
    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    buffer = io.BytesIO(pdf_bytes)

    # ================== STREAMLIT DOWNLOAD BUTTON ==================
    st.divider()
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.download_button(
            "📄 Download Credit Risk Report (PDF)",
            buffer,
            "credit_risk_report.pdf",
            "application/pdf",
            use_container_width=True
        )

st.caption("⚠️ Educational project. Not financial advice.")