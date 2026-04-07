# CrimeTime 🔍
### AI-Powered District-Wise Crime Safety Analyzer for India

**Minor Project | JECRC University | Session 2025–26**

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Clean data & train model (only needed once)
python data_cleaning.py
python train_model.py

# 3. Run the Flask web app
python app.py
```

Then open: http://localhost:5000

---

## Project Structure

```
crimetime/
├── app.py                  # Flask web application
├── data_cleaning.py        # Data cleaning + Safety Score computation
├── train_model.py          # Random Forest model training
├── requirements.txt
├── data/
│   ├── dstrIPC_1.csv       # Raw NCRB dataset
│   ├── cleaned.csv         # Cleaned dataset with Safety Score
│   └── eda_stats.json      # EDA statistics
├── model/
│   ├── rf_model.pkl        # Trained Random Forest model
│   ├── encoders.pkl        # Label encoders
│   └── metrics.json        # Model performance metrics
└── templates/
    ├── index.html          # Main analyzer UI
    └── eda.html            # EDA & insights page
```

---

## Features
- State → District → Year input
- Safety Score (0–100) with color-coded labels: Safe / Moderate / Unsafe / Dangerous
- Crime type breakdown chart
- Year-wise trend charts
- District comparison table within same state
- EDA page with national insights & model performance
- Random Forest classifier with 100% accuracy on test set

## Dataset
NCRB District-wise IPC Crime Data (2001–2012)
- 9,015 records | 35 States/UTs | 808 Districts | 30 crime types
