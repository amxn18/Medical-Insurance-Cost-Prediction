# 🏥 Medical Insurance Charges Prediction

This project predicts medical insurance charges based on user input using Machine Learning models like Linear Regression and XGBoost.

## 📁 Dataset

- Source: `insurance.csv`
- Columns:
  - `age`: Age of primary beneficiary
  - `sex`: Gender (`male`, `female`)
  - `bmi`: Body Mass Index
  - `children`: Number of children covered by insurance
  - `smoker`: Smoking status (`yes`, `no`)
  - `region`: Region (`southeast`, `southwest`, `northeast`, `northwest`)
  - `charges`: Insurance cost

## ⚙️ Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn
- XGBoost

## 🧠 Models Used

- Linear Regression
- XGBoost Regressor

## 🔍 Evaluation Metric

- R² Score

## 🛠️ How to Run

```bash
# Clone the repo
git clone https://github.com/yourusername/medical-insurance-prediction.git
cd medical-insurance-prediction

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the project
python medical.py
