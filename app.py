import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Prediction System")
st.caption("Classroom prototype for exploring possible academic risk using historical student data.")
st.info(
    "Responsible use: This is an educational prototype based on historical public data. "
    "Predictions are not diagnoses and must not be the sole basis for decisions about students."
)

@st.cache_resource
def load_model():
    return joblib.load("student_model.pkl")

@st.cache_data
def load_dataset():
    return pd.read_csv("student_data.csv")

# Check that training has been completed first.
if not os.path.exists("student_model.pkl") or not os.path.exists("student_data.csv"):
    st.error("Required files are missing. Run `python train_model.py` first.")
    st.stop()

try:
    model = load_model()
    data = load_dataset()
except Exception as error:
    st.error(f"Could not load the model or dataset: {error}")
    st.stop()

if "G3" not in data.columns:
    st.error("The dataset does not contain the G3 column.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Total student records", f"{len(data):,}")
c2.metric("Average final grade (G3)", f"{data['G3'].mean():.2f}")
c3.metric("Historical at-risk records", int((data["G3"] < 10).sum()))

st.divider()
st.subheader("📊 Historical dataset preview")
st.dataframe(data.head(10), use_container_width=True)

with st.expander("About the prediction"):
    st.write("""
    - **AT RISK:** predicted final Mathematics grade category is below 10.
    - **NOT AT RISK:** predicted final Mathematics grade category is 10 or higher.
    - G3 is used to create the training target and is excluded from prediction inputs.
    - The model uses a Random Forest classifier.
    """)

st.divider()
st.subheader("📁 Predict risk from an uploaded CSV")
st.write("Upload a CSV containing the same predictor columns used during training. Do not include G3.")

feature_columns = list(model.feature_names_in_)
sample_template = data.drop(columns=["G3"], errors="ignore").head(5)
st.download_button(
    "⬇️ Download sample input CSV",
    data=sample_template.to_csv(index=False).encode("utf-8"),
    file_name="sample_prediction_input.csv",
    mime="text/csv"
)

uploaded_file = st.file_uploader("Choose student CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        new_data = pd.read_csv(uploaded_file)
        missing_columns = [col for col in feature_columns if col not in new_data.columns]

        if missing_columns:
            st.error("Missing required columns: " + ", ".join(missing_columns))
        elif new_data.empty:
            st.warning("The uploaded CSV has no student records.")
        else:
            # Ignore extra columns and preserve the exact training feature order.
            prediction_input = new_data[feature_columns]
            predictions = model.predict(prediction_input)

            results = new_data.copy()
            results["Prediction"] = [
                "AT RISK" if value == 1 else "NOT AT RISK"
                for value in predictions
            ]

            st.success("Prediction completed!")
            left, right = st.columns(2)
            left.metric("Students analyzed", len(results))
            right.metric("Predicted AT RISK", int((predictions == 1).sum()))

            st.subheader("Prediction results")
            st.dataframe(results, use_container_width=True)

            st.download_button(
                "⬇️ Download prediction results",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="student_predictions.csv",
                mime="text/csv"
            )
    except Exception as error:
        st.error(f"Unable to process this CSV: {error}")

st.divider()
st.caption("Dataset: UCI Student Performance Dataset (Mathematics). For academic demonstration only.")
