from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import json
import os

app = Flask(__name__)

BASE = os.path.dirname(__file__)
CLEANED_PATH = os.path.join(BASE, 'data', 'cleaned.csv')
EDA_PATH = os.path.join(BASE, 'data', 'eda_stats.json')
MODEL_PATH = os.path.join(BASE, 'model', 'rf_model.pkl')
ENCODER_PATH = os.path.join(BASE, 'model', 'encoders.pkl')
METRICS_PATH = os.path.join(BASE, 'model', 'metrics.json')

# Load everything at startup
df = pd.read_csv(CLEANED_PATH)
with open(EDA_PATH) as f:
    eda_stats = json.load(f)
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(ENCODER_PATH, 'rb') as f:
    encoders = pickle.load(f)
with open(METRICS_PATH) as f:
    metrics = json.load(f)

FEATURE_COLS = encoders['features']
le_state = encoders['state']
le_district = encoders['district']
le_label = encoders['label']

CRIME_DISPLAY = [
    'MURDER', 'RAPE', 'KIDNAPPING & ABDUCTION', 'ROBBERY',
    'BURGLARY', 'THEFT', 'RIOTS', 'DACOITY',
    'DOWRY DEATHS', 'ARSON', 'HURT/GREVIOUS HURT',
    'CRUELTY BY HUSBAND OR HIS RELATIVES',
    'ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY',
]

@app.route('/')
def index():
    states = sorted(df['STATE/UT'].unique().tolist())
    years = sorted(df['YEAR'].unique().tolist())
    return render_template('index.html', states=states, years=years)

@app.route('/get_districts')
def get_districts():
    state = request.args.get('state', '').upper()
    districts = sorted(df[df['STATE/UT'] == state]['DISTRICT'].unique().tolist())
    return jsonify(districts)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    state = data.get('state', '').upper()
    district = data.get('district', '').upper()
    year = int(data.get('year', 2012))

    # Filter row from dataset
    row = df[(df['STATE/UT'] == state) & (df['DISTRICT'] == district) & (df['YEAR'] == year)]

    if row.empty:
        # Use latest available year for this district
        row = df[(df['STATE/UT'] == state) & (df['DISTRICT'] == district)]
        if row.empty:
            return jsonify({'error': 'No data found for this selection.'})
        row = row.sort_values('YEAR', ascending=False).head(1)

    row = row.iloc[0]
    safety_score = round(float(row['SAFETY_SCORE']), 2)
    safety_label = row['SAFETY_LABEL']

    # Crime breakdown for charts
    crime_breakdown = {col: int(row[col]) for col in CRIME_DISPLAY if col in row.index}

    # District comparison (same state, same year)
    state_df = df[(df['STATE/UT'] == state) & (df['YEAR'] == year)]
    if state_df.empty:
        state_df = df[df['STATE/UT'] == state]
    district_comparison = state_df[['DISTRICT', 'SAFETY_SCORE', 'TOTAL IPC CRIMES']]\
        .sort_values('SAFETY_SCORE', ascending=False)\
        .head(15)\
        .to_dict(orient='records')

    # Year trend for this district
    trend_df = df[(df['STATE/UT'] == state) & (df['DISTRICT'] == district)]\
        .sort_values('YEAR')
    year_trend = {
        'years': trend_df['YEAR'].tolist(),
        'crimes': trend_df['TOTAL IPC CRIMES'].tolist(),
        'scores': trend_df['SAFETY_SCORE'].tolist()
    }

    # State average for comparison
    state_avg_score = round(float(state_df['SAFETY_SCORE'].mean()), 2)
    national_avg_score = round(float(df['SAFETY_SCORE'].mean()), 2)

    return jsonify({
        'state': state,
        'district': district,
        'year': int(row['YEAR']),
        'safety_score': safety_score,
        'safety_label': safety_label,
        'total_crimes': int(row['TOTAL IPC CRIMES']),
        'crime_breakdown': crime_breakdown,
        'district_comparison': district_comparison,
        'year_trend': year_trend,
        'state_avg_score': state_avg_score,
        'national_avg_score': national_avg_score,
    })

@app.route('/eda')
def eda():
    return render_template('eda.html', stats=eda_stats, metrics=metrics)

@app.route('/api/eda_data')
def eda_data():
    return jsonify(eda_stats)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
