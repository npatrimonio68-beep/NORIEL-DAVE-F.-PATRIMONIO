# Student Performance Prediction System

A classroom prototype that estimates whether a student is **at risk of failing** the final Mathematics grade, built for the Data Mining laboratory exercise.

## Problem
Academic risk is often noticed only after poor results. This prototype explores early identification of students who may need additional academic support, using available student indicators such as study time, absences, past failures, and the first- and second-period grades (G1, G2).

## Dataset
UCI Machine Learning Repository, *Student Performance* dataset (ID 320):
https://archive.ics.uci.edu/dataset/320/student+performance

The dataset is downloaded automatically by `train_model.py` through the `ucimlrepo` package and saved as `student_data.csv`.

## Prediction Target
- `G3 < 10` → **AT RISK** (1)
- `G3 >= 10` → **NOT AT RISK** (0)

`G3` is used only to create the target and is **never** used as an input feature (to avoid target leakage).

## Technologies
- Python 3.10+
- pandas, scikit-learn (Random Forest classifier), joblib
- Streamlit (user interface)
- ucimlrepo (dataset retrieval)
- Git / GitHub

## Project Structure
```
student-performance-prediction/
├── train_model.py
├── app.py
├── requirements.txt
└── README.md
```

## Installation and Run Steps
1. Clone the repository and open the project folder:
   ```bash
   git clone https://github.com/YOUR-USERNAME/student-performance-prediction.git
   cd student-performance-prediction
   ```
2. (Recommended) Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   source .venv/bin/activate     # macOS / Linux
   ```
3. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
4. Train the model (creates `student_data.csv` and `student_model.pkl`):
   ```bash
   python train_model.py
   ```
5. Run the app:
   ```bash
   streamlit run app.py
   ```
6. Open http://localhost:8501, download the sample input CSV from the app, and upload it to test predictions.

## How the App Works
1. The user uploads a CSV file.
2. The system checks that all required predictor columns are present.
3. The trained model predicts risk for each record.
4. The system displays the results and lets the user download them as a CSV.

## Limitations
- The model is trained on a single Portuguese secondary-school dataset and may not generalize to other institutions or populations.
- Predictions are **not diagnoses** and must not be used as the sole basis for real student decisions.
- Accuracy alone is not enough to judge the model; review precision, recall, and F1-score, especially for the "At risk" class.
- Using G1 and G2 makes predictions more accurate but later in the academic cycle.

## Optional Extensions
- Compare Random Forest with Logistic Regression.
- Add a confusion matrix and feature-importance discussion.
- Deploy on Streamlit Community Cloud.
- Repeat the experiment without G1 and G2.
