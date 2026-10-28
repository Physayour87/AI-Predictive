"""Reproducible illustrative telecom predictive-maintenance experiment.
Synthetic observations are NOT real operator telemetry and do not reproduce
published results unless independently verified.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

FEATURES = ['age', 'temp', 'load', 'batt', 'vswr', 'sig', 'pwr', 'alarms', 'fan', 'days_pm']
LABELS = {'age':'Equipment age (years)', 'temp':'Cabinet temperature (°C)', 'load':'Load (%)', 'batt':'Battery health (%)', 'vswr':'Antenna VSWR', 'sig':'Signal (dBm)', 'pwr':'Supply voltage (V)', 'alarms':'Alarms (30 days)', 'fan':'Fan speed (RPM)', 'days_pm':'Days since preventive maintenance'}

def generate_data(n=20000, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.uniform(0, 10, n)
    temp = np.clip(rng.normal(39, 7.4, n), 15, 65)
    load = np.clip(rng.normal(60, 15, n), 0, 100)
    batt = np.clip(90 - 4 * age + rng.normal(0, 14, n), 5, 100)
    vswr = np.clip(rng.normal(1.45, .17, n), 1.0, 3.0)
    sig = np.clip(rng.normal(-77, 6.2, n), -110, -40)
    pwr = np.clip(rng.normal(47, 1.6, n), 38, 55)
    alarms = rng.poisson(np.clip(1.5 + .18 * age + .018 * np.maximum(temp-39, 0), .1, 15))
    fan = np.clip(5200 - 45 * age + rng.normal(0, 280, n), 2500, 6000)
    days_pm = rng.uniform(0, 365, n)
    risk = (.44*(age-5) + .052*(temp-39) + .018*(load-60)
            -.040*(batt-70) + 1.0*(vswr-1.45) + .20*(alarms-2.5)
            -.001*(fan-4900) + .002*(days_pm-182) + .05*(-77-sig)
            + .09*(47-pwr))
    # Intercept calibrated to ~23.4% prevalence, not to published confusion matrices.
    lo, hi = -12., 12.
    for _ in range(50):
        mid = (lo+hi)/2
        prob = 1/(1+np.exp(-np.clip(risk+mid, -40, 40)))
        if prob.mean() > .234: hi = mid
        else: lo = mid
    prob = 1/(1+np.exp(-np.clip(risk+(lo+hi)/2, -40, 40)))
    df = pd.DataFrame(dict(age=age,temp=temp,load=load,batt=batt,vswr=vswr,sig=sig,pwr=pwr,alarms=alarms,fan=fan,days_pm=days_pm))
    df['fail'] = rng.binomial(1, prob)
    df.insert(0, 'site_id', [f'SITE-{i+1:05d}' for i in range(n)])
    return df

def get_models(seed=42):
    return {
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced', random_state=seed)),
        'Random Forest': RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=seed, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(random_state=seed)
    }

def evaluate(df, seed=42):
    X, y = df[FEATURES], df['fail'].astype(int)
    # Keep final test untouched during threshold choice.
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=.25, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=.20, stratify=y_dev, random_state=seed)
    models = get_models(seed)
    rows, results = [], {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for name, model in models.items():
        model.fit(X_train, y_train)
        pv = model.predict_proba(X_val)[:,1]
        # Select threshold on validation only, minimising planned-visit + missed-failure costs.
        thresholds = np.linspace(.05, .95, 91)
        costs = [1500*(pv>=t).sum()+12000*((pv<t)&(y_val.to_numpy()==1)).sum() for t in thresholds]
        threshold = float(thresholds[int(np.argmin(costs))])
        model.fit(X_dev, y_dev)
        pt = model.predict_proba(X_test)[:,1]
        pred = (pt>=threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0,1]).ravel()
        scores = cross_val_score(get_models(seed)[name], X_dev, y_dev, cv=cv, scoring='roc_auc', n_jobs=1)
        rows.append({'Model':name, 'Threshold':round(threshold,3),'Precision':precision_score(y_test,pred,zero_division=0), 'Recall':recall_score(y_test,pred,zero_division=0),'F1':f1_score(y_test,pred,zero_division=0),'ROC-AUC':roc_auc_score(y_test,pt),'CV AUC mean':scores.mean(),'CV AUC std':scores.std(),'TN':tn,'FP':fp,'FN':fn,'TP':tp})
        results[name] = {'model':model, 'threshold':threshold, 'probabilities':pt, 'predictions':pred}
    return pd.DataFrame(rows), results, X_test, y_test

def economics(tn, fp, fn, tp, n, failures, failure_cost=12000, visit_cost=1500, failure_hours=6, planned_hours=1):
    visits = int(tp+fp)
    return pd.DataFrame([
        {'Strategy':'Reactive', 'Cost (USD)':failures*failure_cost,'Downtime (hours)':failures*failure_hours,'Visits':0},
        {'Strategy':'Time-based', 'Cost (USD)':n*visit_cost,'Downtime (hours)':np.nan,'Visits':n},
        {'Strategy':'Predictive', 'Cost (USD)':visits*visit_cost+fn*failure_cost,'Downtime (hours)':tp*planned_hours+fn*failure_hours,'Visits':visits},
    ])
