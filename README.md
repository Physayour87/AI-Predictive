# AI Predictive Maintenance in Telecom — Research Prototype

Based on the uploaded MIT project methodology, this application trains Logistic Regression, Random Forest and Gradient Boosting classifiers on **newly generated synthetic** telecom base-station observations. It predicts a 30-day equipment-failure risk and compares maintenance strategies.

## Install and run

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local address printed by Streamlit (normally http://localhost:8501).

## Research design
- Ten features: age, temperature, load, battery health, VSWR, signal, supply voltage, alarm count, fan speed, days since preventive maintenance.
- 75% development / 25% hold-out test split; a validation split within development selects the economic threshold.
- Five-fold stratified ROC-AUC cross-validation is performed only on development data.
- Metrics: precision, recall, F1, ROC-AUC, confusion matrix.
- Illustrative costs: $12,000/unplanned failure, $1,500/planned visit, 6 hours/unplanned outage, 1 hour/prevented failure.

## Important limitations
The synthetic generator is an **independent illustrative reconstruction**, not the dissertation's original source code or dataset. Its outcomes must not be represented as the dissertation's previously reported experimental results. The model's 30-day label is simulated, not a measured forward-looking failure event. Random splitting is acceptable for the synthetic demonstration but real repeated-site telemetry requires chronological/site-aware validation. The economics assume successful prevention after intervention, and do not model partial effectiveness, staffing capacity or uncertainty. Real deployment needs operator data, calibrated risks, data governance, authentication and monitoring.
