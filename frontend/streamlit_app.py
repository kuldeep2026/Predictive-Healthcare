import streamlit as st
import joblib


# 1.Load trained model + encoder
model = joblib.load("models/disease_model.joblib")
mlb = joblib.load("models/symptom_encoder.joblib")

# 2.Get all possible symptoms from encoder
all_symptoms = list(mlb.classes_)

# 3.Page config

st.set_page_config(
    page_title="Predictive Healthcare",
    page_icon="frontend/assets/logo.png",
    layout="wide"
)

# 4.Background & card styling

st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1588776814546-86f328c62de1");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
.card {
    background-color: rgba(255, 255, 255, 0.95);
    padding: 20px;
    border-radius: 20px;
    box-shadow: 4px 4px 20px rgba(0,0,0,0.3);
    margin: 20px auto;
    max-width: 600px;
    font-family: 'Arial', sans-serif;
}
.tips {
    color: #2b7a78;
    font-weight: bold;
    font-size: 16px;
}
.disclaimer {
    background-color: rgba(255, 220, 220, 0.9);
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 14px;
    color: #800000;
    margin-top: 40px;
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
}
.sidebar .sidebar-content {
    background-color: #B3E5FC;
    color: #000000;
    padding: 20px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

# 5.App title

col1, col2 = st.columns([1, 15])  # Adjust ratio for less gap

with col1:
    st.image("frontend/assets/logo.png", width=60)  # Logo

with col2:
    # Use markdown with <h1> and margin/line-height tweaks for vertical alignment
    st.markdown(
        "<h1 style='margin:0; line-height:0.6;'>Predictive Healthcare</h1>",
        unsafe_allow_html=True
    )
st.write("Select your symptoms to predict the disease.")

# 6.Sidebar content

st.sidebar.markdown("<h2 style='color:#007F5F'>💡 Instructions</h2>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style='background-color:#B3E5FC;color:#000000;padding:15px;border-radius:10px'>
<ul>
<li>📝 Select your symptoms from the list</li>
<li>▶ Click <b>Predict</b> button</li>
<li>📊 Predicted disease will appear in a card</li>
<li>ℹ️ See <b>About</b> section for more info</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<h2 style='color:#007F5F'>🧪 How it works</h2>", unsafe_allow_html=True)
st.sidebar.info("""
1. Symptoms converted to binary vector using encoder  
2. ML model predicts the disease  
3. Prediction + tips shown in a card
""")

st.sidebar.markdown("<h2 style='color:#007F5F'>ℹ️ About</h2>", unsafe_allow_html=True)
st.sidebar.info("""
This is a **Predictive Healthcare Prototype**.  
This app predicts a disease based on the symptoms you enter.
It uses a trained ML model with a small dataset of 10 diseases for educational purposes.
\n⚠️ It is not a substitute for professional medical advice.
Future versions will include more diseases and a top 3 probable disease feature.
""")

st.sidebar.markdown("<h2 style='color:#007F5F'>📌 Example Symptoms</h2>", unsafe_allow_html=True)
st.sidebar.info("fever, cough, fatigue, headache, nausea, chest pain, shortness of breath, etc.")

# 7.Multiselect for symptoms
symptoms_selected = st.multiselect(
    "Select your symptoms:",
    options=all_symptoms
)

# 8. Predict button
if st.button("Predict"):
    if not symptoms_selected:
        st.warning("Please select at least one symptom.")
    else:
        # Transform input
        X_input = mlb.transform([symptoms_selected])

        # Predict
        prediction = model.predict(X_input)[0]

        # Disease tips
        disease_info = {
            "Common Cold": "🤧 Rest, drink warm fluids",
            "Influenza (Flu)": "🤒 See a doctor, antiviral medicines",
            "Malaria": "🩸 Consult a physician immediately",
            "Dengue": "💧 Hydrate, see doctor for platelet count",
            "Typhoid": "💊 Consult a doctor, proper antibiotics",
            "Diabetes": "🍎 Monitor blood sugar, see endocrinologist",
            "Hypertension (High BP)": "🏃‍♂️ Monitor BP, lifestyle changes, consult doctor",
            "Asthma": "💨 Use inhaler, avoid triggers, consult doctor",
            "Migraine": "🌙 Rest, avoid bright lights, painkillers",
            "Tuberculosis (TB)": "🚨 Immediate medical treatment required"
        }

        # Show prediction in professional card
        st.markdown(
            f"""
            <div class="card">
                <h3 style="color:#007F5F;font-size:24px;">✅ Predicted Disease: {prediction}</h3>
                <p class="tips">💡 Tips: {disease_info.get(prediction, "No info available")}</p>
            </div>
            """, unsafe_allow_html=True
        )

# 9. Disclaimer at bottom
st.markdown(
    """
    <div class="disclaimer">
    ⚠️ Disclaimer: This is a <b>prototype for educational purposes only</b>.  
    It is <b>not a substitute</b> for professional medical advice.
    </div>
    """, unsafe_allow_html=True
)
