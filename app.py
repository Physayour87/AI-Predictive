import streamlit as st
import pandas as pd
import numpy as np
from core import FEATURES, LABELS, generate_data, evaluate, economics

st.set_page_config(page_title='Telecom AI Predictive Maintenance', page_icon='📡', layout='wide')
st.title('📡 AI Predictive Maintenance for Telecom Networks')
st.caption('Research prototype • Synthetic data • Illustrative 30-day equipment failure risk')
@st.cache_data(show_spinner='Generating synthetic data and training three models...')
def experiment(n, seed):
    df = generate_data(n, seed)
    metrics, models, X_test, y_test = evaluate(df, seed)
    return df, metrics, models, X_test, y_test

with st.sidebar:
    st.header('Experiment settings')
    n = st.selectbox('Synthetic records', [2000, 5000, 20000], index=0)
    seed = st.number_input('Random seed', value=42, step=1)
    st.info('20,000 records is closest to the dissertation design but trains more slowly.')
df, metrics, results, X_test, y_test = experiment(n, int(seed))
tab1, tab2, tab3, tab4 = st.tabs(['Overview', 'Model comparison', 'Predict site failure', 'Maintenance economics'])
with tab1:
    a,b,c = st.columns(3)
    a.metric('Synthetic records', f'{len(df):,}')
    b.metric('Simulated failure rate', f'{df.fail.mean():.1%}')
    c.metric('Prediction horizon', '30 days')
    st.subheader('Sample equipment observations')
    st.dataframe(df.head(15), use_container_width=True)
    st.download_button('Download synthetic dataset (CSV)', df.to_csv(index=False), 'telecom_synthetic_data.csv', 'text/csv')
with tab2:
    st.subheader('Held-out test performance')
    st.dataframe(metrics.round(3), use_container_width=True)
    st.caption('Thresholds selected using a separate validation partition; cross-validation runs on development data only. No results are guaranteed to match the document.')
    selected = st.selectbox('Inspect model', metrics['Model'].tolist())
    r = metrics.set_index('Model').loc[selected]
    st.write(f"Confusion matrix: TN={int(r.TN)}, FP={int(r.FP)}, FN={int(r.FN)}, TP={int(r.TP)}")
    st.bar_chart(metrics.set_index('Model')[['Precision','Recall','F1','ROC-AUC']])
    model = results[selected]['model']
    if hasattr(model, 'feature_importances_'):
        st.subheader('Feature importance (model-derived)')
        importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
        st.bar_chart(importance)
with tab3:
    st.subheader('Assess a telecom base station')
    name = st.selectbox('Prediction model', metrics['Model'].tolist(), index=2)
    defaults = {'age':5.,'temp':39.,'load':60.,'batt':70.,'vswr':1.45,'sig':-77.,'pwr':47.,'alarms':2.,'fan':4900.,'days_pm':182.}
    with st.form('site_form'):
        cols = st.columns(2)
        values = {}
        for i, feature in enumerate(FEATURES):
            with cols[i%2]:
                values[feature] = st.number_input(LABELS[feature], value=float(defaults[feature]))
        submitted = st.form_submit_button('Predict 30-day failure risk')
    if submitted:
        record = pd.DataFrame([values], columns=FEATURES)
        p = float(results[name]['model'].predict_proba(record)[0,1])
        t = results[name]['threshold']
        st.metric('Estimated 30-day failure probability', f'{p:.1%}')
        st.write('**Recommendation:** ' + ('Schedule remote diagnostics and prioritise a planned maintenance visit.' if p>=t else 'Continue monitoring; no immediate model-triggered visit.'))
        st.caption(f'Operational threshold: {t:.2f}. This is a synthetic-data demonstration, not a certified safety decision.')
with tab4:
    st.subheader('Illustrative maintenance strategy comparison')
    name = st.selectbox('Choose model for cost analysis', metrics['Model'].tolist(), index=2)
    col1,col2 = st.columns(2)
    with col1:
        failure_cost = st.number_input('Unplanned failure cost (USD)', min_value=1., value=12000., step=500.)
        failure_hours = st.number_input('Unplanned failure downtime (hours)', min_value=0., value=6.)
    with col2:
        visit_cost = st.number_input('Planned visit cost (USD)', min_value=1., value=1500., step=100.)
        planned_hours = st.number_input('Downtime for prevented failure (hours)', min_value=0., value=1.)
    r = metrics.set_index('Model').loc[name]
    econ = economics(int(r.TN),int(r.FP),int(r.FN),int(r.TP),len(y_test),int(y_test.sum()),failure_cost,visit_cost,failure_hours,planned_hours)
    st.dataframe(econ, use_container_width=True)
    reactive = econ.iloc[0]['Cost (USD)']; predictive = econ.iloc[2]['Cost (USD)']
    st.metric('Modelled cost saving vs reactive', f'{(1-predictive/reactive):.1%}')
    st.bar_chart(econ.set_index('Strategy')['Cost (USD)'])
    st.caption('Assumes every predicted true positive is successfully prevented by one planned visit. Time-based downtime is not modelled, matching the dissertation. Results are scenario estimates, not measured savings.')
