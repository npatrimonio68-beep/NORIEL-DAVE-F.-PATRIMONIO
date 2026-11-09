# Student Performance Prediction System

A Python, Random Forest, Streamlit, and GitHub classroom prototype based on the UCI Student Performance dataset (Mathematics).

## 1. Project files

- `train_model.py` — downloads the UCI dataset, preprocesses the data, trains and evaluates the Random Forest model.
- `app.py` — Streamlit dashboard for viewing the dataset and predicting uploaded student records.
- `requirements.txt` — Python package dependencies.

The training script creates `student_data.csv` and `student_model.pkl` when it runs successfully.

## 2. Prediction target

- `G3 < 10` = **AT RISK** (target value 1)
- `G3 >= 10` = **NOT AT RISK** (target value 0)

**Important:** G3 is excluded from the model input features to prevent target leakage.

## 3. Requirements

- Python 3.10 or later
- Internet connection for the initial UCI dataset download
- VS Code or another terminal
- Git and a GitHub account

## 4. Install and run locally

Open a terminal in this project folder:

```bash
python -m venv .venv
```

Activate the environment:

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install packages:
```bash
pip install -r requirements.txt
```

Train the model and download the dataset:
```bash
python train_model.py
```

Start Streamlit:
```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## 5. Test the app

1. Download the sample input CSV from the Streamlit app.
2. Upload that sample CSV to the app.
3. Confirm that the prediction results appear.
4. Download the results CSV.
5. Upload a CSV with a missing required column and confirm that an error appears.

## 6. Upload to GitHub

Create a GitHub repository named `student-performance-prediction`. In the terminal, run:

```bash
git init
git add train_model.py app.py requirements.txt README.md .gitignore
git commit -m "Build student performance prediction prototype"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/student-performance-prediction.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username.

Do not upload passwords, access tokens, or other secrets. The generated CSV and model files are excluded by `.gitignore`; your instructor may allow you to upload them if required.

## 7. SAD mini-analysis

- **Problem:** Academic risk may be noticed only after poor results. The prototype explores whether historical student indicators can help identify possible risk.
- **Users:** Faculty member or academic analyst.
- **Inputs:** Student attributes and academic indicators such as study time, absences, past failures, G1, and G2.
- **Process:** Validate data → preprocess → apply trained model → classify risk.
- **Outputs:** AT RISK / NOT AT RISK prediction and model evaluation summary.
- **Limitation:** The model uses historical data from Portuguese secondary schools and may not generalize to another institution or student population.

## 8. Responsible use

Predictions are not diagnoses. A faculty member should review the student's situation and provide appropriate support. Do not use the model as the sole basis for real student decisions.
