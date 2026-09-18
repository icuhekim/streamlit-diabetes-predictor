from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# This must be the first Streamlit command in the application.
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered",
)


# Build an absolute model path that works locally and on Render.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "diabetes_pipeline.pkl"


@st.cache_resource
def load_model():
    """
    Load the trained pipeline once and keep it in memory.

    Streamlit reruns this script whenever the user interacts with a widget.
    Caching prevents the model from being reloaded on every interaction.
    """
    return joblib.load(MODEL_PATH)


model = load_model()


st.title("Diabetes Risk Predictor")

st.write(
    """
    Enter the patient characteristics below. The trained machine-learning
    model will estimate the probability of the positive diabetes outcome
    represented in the original dataset.
    """
)

st.info(
    "This application is an educational machine-learning demonstration "
    "and is not intended for diagnosis or clinical decision-making."
)


with st.form("prediction_form"):
    st.subheader("Patient information")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=1,
            step=1,
        )

        glucose = st.number_input(
            "Glucose",
            min_value=0.0,
            max_value=250.0,
            value=120.0,
            step=1.0,
        )

        blood_pressure = st.number_input(
            "Blood pressure",
            min_value=0.0,
            max_value=150.0,
            value=70.0,
            step=1.0,
        )

        skin_thickness = st.number_input(
            "Skin thickness",
            min_value=0.0,
            max_value=100.0,
            value=20.0,
            step=1.0,
        )

    with col2:
        insulin = st.number_input(
            "Insulin",
            min_value=0.0,
            max_value=900.0,
            value=80.0,
            step=1.0,
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=70.0,
            value=25.0,
            step=0.1,
        )

        diabetes_pedigree = st.number_input(
            "Diabetes pedigree function",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            step=0.01,
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=33,
            step=1,
        )

    submitted = st.form_submit_button(
        "Estimate risk",
        type="primary",
        use_container_width=True,
    )


if submitted:
    patient_data = pd.DataFrame(
        [
            {
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": diabetes_pedigree,
                "Age": age,
            }
        ]
    )

    predicted_class = int(model.predict(patient_data)[0])
    probability = float(model.predict_proba(patient_data)[0, 1])

    st.divider()
    st.subheader("Model result")

    st.metric(
        label="Estimated probability",
        value=f"{probability:.1%}",
    )

    st.progress(probability)

    if predicted_class == 1:
        st.warning("Higher model-estimated likelihood of a positive outcome.")
    else:
        st.success("Lower model-estimated likelihood of a positive outcome.")

    with st.expander("View submitted values"):
        st.dataframe(patient_data, hide_index=True, use_container_width=True)