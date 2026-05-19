# ============================================================
# AML TRANSACTION MONITORING SYSTEM — DISSERTATION VERSION
# SAML-D Dataset | Random Forest (Best Model) | 5-Classifier Comparison
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AML Monitoring System",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — PROFESSIONAL DISSERTATION THEME
# ============================================================

st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1f2e, #252d3d);
        border: 1px solid #2e3a50;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1a1f2e;
        border-radius: 10px;
        padding: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #a0aec0;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: white !important;
    }
    .section-header {
        background: linear-gradient(90deg, #1e3a5f, #1a1f2e);
        border-left: 4px solid #2563eb;
        padding: 10px 16px;
        border-radius: 0 8px 8px 0;
        margin: 20px 0 12px 0;
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
    }
    .info-box {
        background: linear-gradient(135deg, #1a2744, #1e2d4a);
        border: 1px solid #2563eb44;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 10px 0;
        color: #cbd5e0;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .alert-high {
        background: linear-gradient(135deg, #3b1414, #4a1c1c);
        border: 1px solid #ef444455;
        border-radius: 8px;
        padding: 12px 16px;
        color: #fca5a5;
        font-weight: 500;
    }
    .alert-medium {
        background: linear-gradient(135deg, #3b2e14, #4a391c);
        border: 1px solid #f59e0b55;
        border-radius: 8px;
        padding: 12px 16px;
        color: #fcd34d;
        font-weight: 500;
    }
    .alert-low {
        background: linear-gradient(135deg, #14311a, #1c3d24);
        border: 1px solid #22c55e55;
        border-radius: 8px;
        padding: 12px 16px;
        color: #86efac;
        font-weight: 500;
    }
    .perf-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
        color: #cbd5e0;
    }
    .perf-table th {
        background: #1e3a5f;
        padding: 10px 14px;
        text-align: left;
        border-bottom: 2px solid #2563eb;
        font-weight: 600;
        color: #93c5fd;
    }
    .perf-table td {
        padding: 9px 14px;
        border-bottom: 1px solid #2e3a50;
    }
    .perf-table tr:nth-child(even) td { background: #1a1f2e; }
    .perf-table tr:hover td { background: #1e3a5f44; }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-blue   { background: #1e40af44; color: #93c5fd; border: 1px solid #3b82f655; }
    .badge-green  { background: #14532d44; color: #86efac; border: 1px solid #22c55e55; }
    .badge-red    { background: #7f1d1d44; color: #fca5a5; border: 1px solid #ef444455; }
    .badge-yellow { background: #78350f44; color: #fcd34d; border: 1px solid #f59e0b55; }
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<h1 style='margin-bottom:4px'>🛡️ Anti-Money Laundering Transaction Monitoring System</h1>
<p style='color:#64748b; font-size:0.95rem; margin-top:0'>
    SAML-D Synthetic Dataset &nbsp;|&nbsp; Random Forest (Best Model) &nbsp;|&nbsp;
    SMOTE Resampling &nbsp;|&nbsp; Hybrid ML + Rule-Based Detection
</p>
""", unsafe_allow_html=True)
st.divider()

# ============================================================
# LOAD MODEL ARTEFACTS
# ============================================================

model_loaded = False
try:
    model  = joblib.load("best_aml_model.pkl")
    scaler = joblib.load("aml_scaler.pkl")
    EXPECTED_FEATURES = joblib.load("feature_columns.pkl")
    model_loaded = True
except Exception:
    EXPECTED_FEATURES = [
        'Amount','log_amount','currency_mismatch','cross_border',
        'high_amount','high_frequency','sender_txn_count',
        'receiver_txn_count','fan_out','fan_in',
        'hour','day_of_week','is_weekend',
        'amount_vs_avg','pair_frequency'
    ]

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### 🗂️ System Status")
    if model_loaded:
        st.success("✅ Model loaded successfully")
    else:
        st.error("⚠️ Model not found")

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    <div class='info-box'>
    Dissertation dashboard for <b>ML-based AML detection</b> using the
    SAML-D synthetic dataset (1,048,575 transactions).<br><br>
    Combines a trained ML classifier with expert rule-based scoring
    to flag suspicious financial activity.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎨 Risk Legend")
    st.markdown("""
    <div class='alert-high'>🔴 High — ML Prob &gt; 0.30</div><br>
    <div class='alert-medium'>🟡 Medium — ML Prob 0.10–0.30</div><br>
    <div class='alert-low'>🟢 Low — ML Prob &lt; 0.10</div>
    """, unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "📊  Dashboard & Analytics",
    "🧠  Model & Methodology",
    "🔮  Single Transaction Prediction"
])

# ============================================================
# TAB 1: DASHBOARD
# ============================================================

with tab1:

    uploaded_file = st.file_uploader(
        "Upload SAML-D formatted CSV dataset",
        type=["csv"],
        help="Expected columns: Time, Date, Sender_account, Receiver_account, Amount, "
             "Payment_currency, Received_currency, Sender_bank_location, "
             "Receiver_bank_location, Payment_type, Is_laundering"
    )

    if not uploaded_file:
        st.markdown("""
        <div class='info-box'>
        📂 <b>Upload your dataset above to begin.</b><br><br>
        The system expects CSV files following the <b>SAML-D schema</b>:<br>
        • <b>Amount</b> — Transaction value<br>
        • <b>Payment_currency / Received_currency</b> — Currency codes<br>
        • <b>Sender_bank_location / Receiver_bank_location</b> — Country codes<br>
        • <b>Payment_type</b> — ACH, Cross-border, Cheque, Cash Deposit, etc.<br>
        • <b>Is_laundering</b> — Ground-truth label (0 = Normal, 1 = Suspicious)<br><br>
        ⏱️ <b>Note:</b> Large datasets may take 30–60 seconds to process.
        </div>
        """, unsafe_allow_html=True)

    else:
        # Load and clean
        df = pd.read_csv(uploaded_file)
        df = df.loc[:, ~df.columns.duplicated()]
        df.columns = df.columns.str.strip()

        if "Amount" not in df.columns:
            num = df.select_dtypes(include='number').columns
            df["Amount"] = df[num[0]] if len(num) else 0

        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        df = df.dropna(subset=['Amount'])
        if df.empty:
            st.error("Dataset empty after cleaning.")
            st.stop()

        # ── ROW-LEVEL FEATURES ──────────────────────────────────────
        df['log_amount']  = np.log1p(df['Amount'])
        df['high_amount'] = (df['Amount'] > 10000).astype(int)

        df['currency_mismatch'] = 0
        if 'Payment_currency' in df.columns and 'Received_currency' in df.columns:
            df['currency_mismatch'] = (df['Payment_currency'] != df['Received_currency']).astype(int)

        df['cross_border'] = 0
        if 'Sender_bank_location' in df.columns and 'Receiver_bank_location' in df.columns:
            df['cross_border'] = (df['Sender_bank_location'] != df['Receiver_bank_location']).astype(int)

        if 'Date' in df.columns and 'Time' in df.columns:
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce', dayfirst=True)
        elif 'Date' in df.columns:
            df['DateTime'] = pd.to_datetime(df['Date'], errors='coerce')
        else:
            df['DateTime'] = pd.NaT

        df['hour']        = df['DateTime'].dt.hour.fillna(0)
        df['day_of_week'] = df['DateTime'].dt.dayofweek.fillna(0)
        df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)

        # ── SORT CHRONOLOGICALLY ─────────────────────────────────────
        # Must match the notebook training pipeline exactly
        if df['DateTime'].notna().any():
            df = df.sort_values('DateTime').reset_index(drop=True)

        # Aggregation features
        from collections import defaultdict

        sender_receivers_map  = defaultdict(set)
        receiver_senders_map  = defaultdict(set)
        sender_counts_map_    = defaultdict(int)
        receiver_counts_map_  = defaultdict(int)
        sender_sum_map_       = defaultdict(float)
        pair_count_map_       = defaultdict(int)

        fan_out_list_       = []
        fan_in_list_        = []
        s_count_list_       = []
        r_count_list_       = []
        amt_vs_avg_list_    = []
        pair_freq_list_     = []

        has_accounts = ('Sender_account' in df.columns and
                        'Receiver_account' in df.columns)

        if has_accounts:
            with st.spinner("Processing dataset features..."):
                for _, row in df.iterrows():
                    s   = row['Sender_account']
                    r   = row['Receiver_account']
                    amt = row['Amount']
                    key = (s, r)

                    fan_out_list_.append(len(sender_receivers_map[s]))
                    fan_in_list_.append(len(receiver_senders_map[r]))
                    s_count_list_.append(sender_counts_map_[s])
                    r_count_list_.append(receiver_counts_map_[r])
                    pair_freq_list_.append(pair_count_map_[key])

                    if sender_counts_map_[s] > 0:
                        avg_val = sender_sum_map_[s] / sender_counts_map_[s]
                        amt_vs_avg_list_.append(amt / avg_val if avg_val > 0 else 1.0)
                    else:
                        amt_vs_avg_list_.append(1.0)

                    sender_receivers_map[s].add(r)
                    receiver_senders_map[r].add(s)
                    sender_counts_map_[s]   += 1
                    receiver_counts_map_[r] += 1
                    sender_sum_map_[s]      += amt
                    pair_count_map_[key]    += 1

            df['fan_out']           = fan_out_list_
            df['fan_in']            = fan_in_list_
            df['sender_txn_count']  = s_count_list_
            df['receiver_txn_count']= r_count_list_
            df['amount_vs_avg']     = amt_vs_avg_list_
            df['pair_frequency']    = pair_freq_list_
            df['high_frequency']    = (df['sender_txn_count'] > 10).astype(int)
        else:
            df['fan_out'] = df['fan_in'] = 0
            df['sender_txn_count'] = df['receiver_txn_count'] = 0
            df['high_frequency'] = 0
            df['amount_vs_avg'] = 1
            df['pair_frequency'] = 0

        # ML prediction
        if model_loaded:
            for col in EXPECTED_FEATURES:
                if col not in df.columns:
                    df[col] = 0
            X = df[EXPECTED_FEATURES].fillna(0)
            try:
                X_sc = scaler.transform(X)
                df['ML_Prediction']  = model.predict(X_sc)
                df['ML_Probability'] = model.predict_proba(X_sc)[:, 1]
            except Exception:
                df['ML_Prediction']  = model.predict(X)
                df['ML_Probability'] = model.predict_proba(X)[:, 1]
        else:
            df['ML_Prediction']  = 0
            df['ML_Probability'] = 0.0

        # Rule-based scoring
        df['Risk_Score'] = (
            df['high_amount']   * 30 +
            df['cross_border']  * 20 +
            df['currency_mismatch'] * 15 +
            df['high_frequency']* 25 +
            (df['fan_out'] > 10).astype(int) * 20 +
            (df['fan_in']  > 10).astype(int) * 20 +
            (df['amount_vs_avg'] > 3).astype(int) * 20
        )

        df['Final_Flag'] = ((df['ML_Probability'] > 0.30) | (df['Risk_Score'] > 70)).astype(int)
        df['Risk_Level'] = np.where(
            df['ML_Probability'] > 0.30, "High",
            np.where(df['ML_Probability'] > 0.10, "Medium", "Low")
        )

        has_label = 'Is_laundering' in df.columns

        # KPI metrics
        st.markdown("<div class='section-header'>📌 Key Performance Indicators</div>", unsafe_allow_html=True)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total Transactions", f"{len(df):,}")
        c2.metric("ML Flagged",         f"{int(df['ML_Prediction'].sum()):,}")
        c3.metric("Final Flagged",      f"{int(df['Final_Flag'].sum()):,}")
        c4.metric("High Risk",          f"{int((df['Risk_Level']=='High').sum()):,}")
        c5.metric("Avg Amount",         f"${df['Amount'].mean():,.2f}")
        c6.metric("Max Amount",         f"${df['Amount'].max():,.2f}")

        if has_label:
            st.markdown("<div class='section-header'>🎯 Ground-Truth Performance</div>", unsafe_allow_html=True)
            g1, g2, g3, g4 = st.columns(4)
            tp = int(((df['ML_Prediction'] == 1) & (df['Is_laundering'] == 1)).sum())
            fp = int(((df['ML_Prediction'] == 1) & (df['Is_laundering'] == 0)).sum())
            fn = int(((df['ML_Prediction'] == 0) & (df['Is_laundering'] == 1)).sum())
            prec = tp / (tp + fp + 1e-9)
            rec  = tp / (tp + fn + 1e-9)
            f1   = 2 * prec * rec / (prec + rec + 1e-9)
            g1.metric("True Positives",  f"{tp:,}")
            g2.metric("False Positives", f"{fp:,}")
            g3.metric("Precision",       f"{prec:.3f}")
            g4.metric("F1 Score",        f"{f1:.3f}")

        # Charts row 1
        st.markdown("<div class='section-header'>📊 Transaction Distribution</div>", unsafe_allow_html=True)
        r1c1, r1c2 = st.columns(2)

        CHART_BG = dict(plot_bgcolor='#1a1f2e', paper_bgcolor='#1a1f2e',
                        font_color='#a0aec0', title_font_color='#e2e8f0')
        GRID = dict(gridcolor='#2e3a50')

        with r1c1:
            fig = px.histogram(df, y='Amount', nbins=80,
                               title="Transaction Amount Distribution",
                               color_discrete_sequence=['#3b82f6'],
                               labels={'Amount': 'Amount', 'count': 'Frequency'})
            fig.update_layout(**CHART_BG, xaxis={**GRID}, 
                              yaxis=dict(gridcolor='#2e3a50', range=[0, 100000]))
            st.plotly_chart(fig, use_container_width=True)

        with r1c2:
            fig = px.pie(df, names='Risk_Level', title="Risk Level Distribution",
                         color='Risk_Level',
                         color_discrete_map={'High':'#ef4444','Medium':'#f59e0b','Low':'#22c55e'},
                         hole=0.4)
            fig.update_layout(**CHART_BG)
            st.plotly_chart(fig, use_container_width=True)

        # Charts row 2
        r2c1, r2c2 = st.columns(2)

        with r2c1:
            fig = px.histogram(df, x='ML_Probability', nbins=60,
                               title="ML Fraud Probability Distribution",
                               color_discrete_sequence=['#8b5cf6'],
                               labels={'ML_Probability': 'Fraud Probability'})
            fig.add_vline(x=0.10, line_dash="dash", line_color="#f59e0b",
                          annotation_text="Medium (0.10)", annotation_font_color="#f59e0b")
            fig.add_vline(x=0.30, line_dash="dash", line_color="#ef4444",
                          annotation_text="High (0.30)", annotation_font_color="#ef4444")
            fig.update_layout(**CHART_BG, xaxis={**GRID}, yaxis={**GRID})
            st.plotly_chart(fig, use_container_width=True)

        with r2c2:
            hourly = df.groupby('hour').agg(
                Total=('Amount','count'), Flagged=('Final_Flag','sum')).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=hourly['hour'], y=hourly['Total'],
                                 name='All', marker_color='#3b82f6', opacity=0.6))
            fig.add_trace(go.Bar(x=hourly['hour'], y=hourly['Flagged'],
                                 name='Flagged', marker_color='#ef4444'))
            fig.update_layout(title='Transaction Volume by Hour of Day', barmode='overlay',
                               **CHART_BG, xaxis=dict(title='Hour', **GRID),
                               yaxis=dict(title='Count', **GRID), legend=dict(bgcolor='#1a1f2e'))
            st.plotly_chart(fig, use_container_width=True)

        # Charts row 3
        r3c1, r3c2 = st.columns(2)

        with r3c1:
            sample = df.sample(min(5000, len(df)), random_state=42)
            fig = px.scatter(sample, x='Amount', y='ML_Probability', color='Risk_Level',
                             color_discrete_map={'High':'#ef4444','Medium':'#f59e0b','Low':'#22c55e'},
                             title='Amount vs ML Fraud Probability', opacity=0.6,
                             labels={'ML_Probability': 'Fraud Probability'})
            fig.update_layout(**CHART_BG, xaxis={**GRID}, yaxis={**GRID}, legend=dict(bgcolor='#1a1f2e'))
            st.plotly_chart(fig, use_container_width=True)

        with r3c2:
            if 'Payment_type' in df.columns:
                pt = df.groupby('Payment_type').agg(
                    FlagRate=('Final_Flag','mean')).reset_index().sort_values('FlagRate')
                fig = px.bar(pt, y='Payment_type', x='FlagRate', orientation='h',
                             title='Flag Rate by Payment Type',
                             color='FlagRate', color_continuous_scale='Reds',
                             labels={'FlagRate':'Proportion Flagged','Payment_type':''})
                fig.update_layout(**CHART_BG, xaxis={**GRID}, yaxis={**GRID}, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

        # Charts row 4
        r4c1, r4c2 = st.columns(2)

        with r4c1:
            s2 = df.sample(min(3000, len(df)), random_state=1)
            fig = px.scatter(s2, x='fan_out', y='fan_in', color='Risk_Level',
                             color_discrete_map={'High':'#ef4444','Medium':'#f59e0b','Low':'#22c55e'},
                             title='Network Behaviour: Fan-Out vs Fan-In', opacity=0.5,
                             labels={'fan_out':'Sender Fan-Out','fan_in':'Receiver Fan-In'})
            fig.update_layout(**CHART_BG, xaxis={**GRID}, yaxis={**GRID}, legend=dict(bgcolor='#1a1f2e'))
            st.plotly_chart(fig, use_container_width=True)

        with r4c2:
            risk_cross = df.groupby(['cross_border','currency_mismatch'])['Final_Flag'].mean().reset_index()
            risk_cross['cross_border']      = risk_cross['cross_border'].map({0:'Domestic',1:'Cross-border'})
            risk_cross['currency_mismatch'] = risk_cross['currency_mismatch'].map(
                {0:'Same Currency',1:'Currency Mismatch'})
            fig = px.bar(risk_cross, x='cross_border', y='Final_Flag',
                         color='currency_mismatch', barmode='group',
                         title='Flag Rate: Cross-Border × Currency Mismatch',
                         labels={'Final_Flag':'Proportion Flagged','cross_border':''},
                         color_discrete_map={'Same Currency':'#3b82f6','Currency Mismatch':'#ef4444'})
            fig.update_layout(**CHART_BG, xaxis={**GRID}, yaxis={**GRID}, legend=dict(bgcolor='#1a1f2e'))
            st.plotly_chart(fig, use_container_width=True)

        # Charts row 5
        r5c1, r5c2 = st.columns(2)

        with r5c1:
            fig = px.box(df[df['amount_vs_avg'] < 20], x='Risk_Level', y='amount_vs_avg',
                         color='Risk_Level',
                         color_discrete_map={'High':'#ef4444','Medium':'#f59e0b','Low':'#22c55e'},
                         title='Amount vs Sender Average — by Risk Level',
                         labels={'amount_vs_avg':'Amount / Sender Average','Risk_Level':''})
            fig.update_layout(**CHART_BG, xaxis={**GRID}, yaxis={**GRID}, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with r5c2:
            day_map = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
            df['Day'] = df['day_of_week'].map(day_map)
            day_flag = df.groupby('Day').agg(FlagRate=('Final_Flag','mean')).reindex(
                ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']).reset_index()
            fig = go.Figure(go.Bar(
                x=day_flag['Day'], y=day_flag['FlagRate'],
                marker_color=['#ef4444' if d in ['Sat','Sun'] else '#3b82f6' for d in day_flag['Day']]
            ))
            fig.update_layout(title='Flag Rate by Day of Week', **CHART_BG,
                               xaxis={**GRID}, yaxis=dict(title='Proportion Flagged', **GRID))
            st.plotly_chart(fig, use_container_width=True)

        # Risk score histogram
        st.markdown("<div class='section-header'>⚖️ Rule-Based Risk Score Distribution</div>", unsafe_allow_html=True)
        fig = px.histogram(df, x='Risk_Score', nbins=30, color='Risk_Level',
                           color_discrete_map={'High':'#ef4444','Medium':'#f59e0b','Low':'#22c55e'},
                           title='Composite Rule-Based Risk Score Distribution',
                           labels={'Risk_Score':'Risk Score (0–150)'}, barmode='stack')
        fig.add_vline(x=70, line_dash="dash", line_color="white",
                      annotation_text="Flag threshold = 70", annotation_font_color="white")
        fig.update_layout(**CHART_BG, xaxis={**GRID}, yaxis={**GRID}, legend=dict(bgcolor='#1a1f2e'))
        st.plotly_chart(fig, use_container_width=True)

        # Flagged transactions table
        st.markdown("<div class='section-header'>🚨 Top Flagged Transactions</div>", unsafe_allow_html=True)
        display_cols = ['Sender_account','Receiver_account','Amount','Payment_type',
                        'Sender_bank_location','Receiver_bank_location',
                        'ML_Probability','Risk_Score','Risk_Level','Final_Flag']
        disp = [c for c in display_cols if c in df.columns]
        flagged_df = df[df['Final_Flag']==1][disp].sort_values('ML_Probability', ascending=False)
        st.dataframe(flagged_df.head(200).style.background_gradient(
            subset=['ML_Probability'], cmap='Reds'),
            use_container_width=True, height=380)
        st.caption(f"Showing top 200 of {len(flagged_df):,} flagged transactions")

        with st.expander("📋 Full Dataset with Predictions"):
            st.dataframe(df[disp].head(5000), use_container_width=True)
            st.caption("Showing first 5,000 rows")


# ============================================================
# TAB 2: MODEL & METHODOLOGY
# ============================================================

with tab2:

    st.markdown("<div class='section-header'>📂 Dataset: SAML-D (Synthetic AML Dataset)</div>", unsafe_allow_html=True)
    col_ds1, col_ds2 = st.columns([3, 2])

    with col_ds1:
        st.markdown("""
        <div class='info-box'>
        The <b>SAML-D</b> dataset is a synthetic Anti-Money Laundering dataset designed
        for academic research and ML benchmarking. It contains <b>1,048,575 transactions</b>
        with severe class imbalance — only <b>0.091%</b> are labelled suspicious (956 cases).<br><br>
        Each record captures sender/receiver identifiers, transaction amounts, currencies,
        bank locations, payment types, and ground-truth laundering labels.
        </div>
        """, unsafe_allow_html=True)

    with col_ds2:
        st.markdown("""
        <table class='perf-table'>
        <tr><th>Property</th><th>Value</th></tr>
        <tr><td>Total Rows</td><td>1,048,575</td></tr>
        <tr><td>Raw Columns</td><td>12</td></tr>
        <tr><td>Legitimate (Class 0)</td><td>1,047,618</td></tr>
        <tr><td>Suspicious (Class 1)</td><td>956</td></tr>
        <tr><td>Fraud Rate</td><td>0.091%</td></tr>
        <tr><td>Missing Values</td><td>None</td></tr>
        </table>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>🗃️ Dataset Schema</div>", unsafe_allow_html=True)
    st.markdown("""
    <table class='perf-table'>
    <tr><th>Column</th><th>Type</th><th>Description</th></tr>
    <tr><td>Time</td><td>String</td><td>Transaction time (HH:MM:SS)</td></tr>
    <tr><td>Date</td><td>String</td><td>Transaction date (DD-MM-YYYY)</td></tr>
    <tr><td>Sender_account</td><td>Integer</td><td>10-digit sender account ID</td></tr>
    <tr><td>Receiver_account</td><td>Integer</td><td>10-digit receiver account ID</td></tr>
    <tr><td>Amount</td><td>Float</td><td>Transaction value</td></tr>
    <tr><td>Payment_currency</td><td>String</td><td>Currency sent (UK pounds, Dirham, etc.)</td></tr>
    <tr><td>Received_currency</td><td>String</td><td>Currency received</td></tr>
    <tr><td>Sender_bank_location</td><td>String</td><td>Country of sending bank</td></tr>
    <tr><td>Receiver_bank_location</td><td>String</td><td>Country of receiving bank</td></tr>
    <tr><td>Payment_type</td><td>String</td><td>ACH, Cross-border, Cheque, Cash Deposit, Wire</td></tr>
    <tr><td>Is_laundering</td><td>Binary</td><td>0 = Normal, 1 = Suspicious</td></tr>
    <tr><td>Laundering_type</td><td>String</td><td>Subtype (Fan-Out, Fan-In, Cycle, etc.)</td></tr>
    </table>
    """, unsafe_allow_html=True)

    if EXPECTED_FEATURES:
        st.markdown("<div class='section-header'>📋 Model Feature Vector</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({'Index': range(1, len(EXPECTED_FEATURES)+1),
                                   'Feature': EXPECTED_FEATURES}),
                     use_container_width=True, hide_index=True)

    st.markdown("<div class='section-header'>📊 Model Performance Comparison</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    Five classifiers were trained and evaluated on the held-out test set (20% of SAML-D).
    All metrics below are for <b>Class 1 (Fraud)</b> — the class that matters.
    The Random Forest was selected as the best model based on its highest F1 Score.
    Overall accuracy is not shown here because it is misleading under extreme class imbalance
    (a model predicting everything as normal scores 99.9% accuracy while catching zero fraud).
    </div>
    """, unsafe_allow_html=True)

    model_data = {
        'Model':             ['Random Forest (Tuned)', 'Decision Tree', 'XGBoost', 'LightGBM', 'Logistic Regression'],
        'Precision (Fraud)': [0.3348,                   0.2981,          0.2754,    0.2512,     0.1840],
        'Recall (Fraud)':    [0.3989,                   0.3617,          0.3245,    0.2980,     0.2109],
        'F1 Score':          [0.3641,                   0.3267,          0.2980,    0.2721,     0.1963],
    }
    CHART_BG2 = CHART_BG if 'CHART_BG' in dir() else dict(plot_bgcolor='#1a1f2e', paper_bgcolor='#1a1f2e',
                     font_color='#a0aec0', title_font_color='#e2e8f0')

    fig_cmp = go.Figure()

    bar_colours = {
        'Precision (Fraud)': '#3b82f6',
        'Recall (Fraud)':    '#8b5cf6',
        'F1 Score':          '#22c55e',
    }

    for metric, colour in bar_colours.items():
        fig_cmp.add_trace(go.Bar(
            name=metric,
            x=model_data['Model'],
            y=model_data[metric],
            marker_color=colour,
            marker_line_color='#2e3a50',
            marker_line_width=0.8,
            text=[f'{v:.4f}' for v in model_data[metric]],
            textposition='outside',
            textfont=dict(color='#e2e8f0', size=10),
        ))



    fig_cmp.update_layout(
        **CHART_BG2,
        barmode='group',
        title=dict(text='Classifier Comparison — Fraud Class Metrics (Test Set)', font=dict(size=14)),
        xaxis=dict(gridcolor='#2e3a50', tickfont=dict(color='#e2e8f0', size=11)),
        yaxis=dict(gridcolor='#2e3a50', range=[0, 0.50],
                   title='Score', tickformat='.3f'),
        legend=dict(bgcolor='#1a1f2e', bordercolor='#2e3a50', borderwidth=1,
                    font=dict(color='#e2e8f0')),
        margin=dict(t=60, b=20),
        height=420,
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown("""
    <table class='perf-table'>
    <tr><th>Model</th><th>Precision (Fraud)</th><th>Recall (Fraud)</th>
        <th>F1 Score</th><th>Selected?</th></tr>
    <tr><td><b>Random Forest (Tuned)</b></td><td>0.3348</td><td>0.3989</td>
        <td><b>0.3641</b></td><td>✅ Best Model</td></tr>
    <tr><td>Decision Tree</td><td>0.2981</td><td>0.3617</td><td>0.3267</td><td>—</td></tr>
    <tr><td>XGBoost</td><td>0.2754</td><td>0.3245</td><td>0.2980</td><td>—</td></tr>
    <tr><td>LightGBM</td><td>0.2512</td><td>0.2980</td><td>0.2721</td><td>—</td></tr>
    <tr><td>Logistic Regression</td><td>0.1840</td><td>0.2109</td><td>0.1963</td><td>—</td></tr>
    </table>

    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>🎯 What the ML Probability Score Means</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    The ML model outputs a number between <b>0.00 and 0.50</b> for every transaction.
    This is the model's confidence that the transaction is fraudulent.
    It is <b>not</b> a percentage of money at risk — it is a probability score.
    The higher the number, the more suspicious the transaction.<br><br>
    In this system, <b>0.50 is treated as the maximum meaningful score</b> because the Random Forest
    rarely exceeds this value even for confirmed fraud cases in the SAML-D dataset — this is
    normal behaviour for ensemble models on imbalanced data.
    </div>
    """, unsafe_allow_html=True)

    prob_examples = [
        (0.02,  '🟢 Low',    'alert-low',
         'Very likely a normal transaction. No action needed.',
         'Domestic payment, same currency, typical amount for this sender.'),
        (0.10,  '🟢 Low',    'alert-low',
         'Low suspicion. Worth logging but no immediate action.',
         'Slightly unusual hour but all other indicators are normal.'),
        (0.15,  '🟡 Medium', 'alert-medium',
         'Some risk signals present. Monitor this account.',
         'Cross-border transaction with currency mismatch but low amount.'),
        (0.25,  '🟡 Medium', 'alert-medium',
         'Multiple risk indicators. Analyst should review.',
         'High-frequency sender making a cross-border transfer.'),
        (0.32,  '🔴 High',   'alert-high',
         'Strong fraud signal. Escalate for investigation.',
         'Large cross-border transfer with currency mismatch at 2 AM.'),
        (0.42,  '🔴 High',   'alert-high',
         'Very strong fraud signal. Likely suspicious activity.',
         'Fan-out pattern, amount 8× sender average, weekend transaction.'),
        (0.50,  '🔴 High',   'alert-high',
         'Maximum fraud confidence. Immediate escalation required.',
         'All rule conditions fired. Classic layering pattern detected.'),
    ]

    for prob, level, css_class, meaning, example in prob_examples:
        gauge_pct = int(prob * 200)
        bar_color = '#ef4444' if prob > 0.30 else '#f59e0b' if prob > 0.10 else '#22c55e'
        st.markdown(f"""
        <div class='{css_class}' style='margin-bottom:10px; padding:14px 18px;'>
            <div style='display:flex; align-items:center; gap:16px; flex-wrap:wrap;'>
                <div style='min-width:90px'>
                    <span style='font-size:1.4rem; font-weight:700; color:#e2e8f0'>{prob:.2f}</span>
                    <span style='font-size:0.8rem; color:#a0aec0; margin-left:4px'>/ 0.50</span>
                </div>
                <div style='flex:1; min-width:160px'>
                    <div style='background:#0e1117; border-radius:6px; height:10px; overflow:hidden;'>
                        <div style='width:{gauge_pct}%; background:{bar_color}; height:100%; border-radius:6px;'></div>
                    </div>
                    <div style='font-size:0.75rem; color:#64748b; margin-top:3px'>{gauge_pct}% of max</div>
                </div>
                <div style='min-width:80px'>
                    <span style='font-weight:600'>{level}</span>
                </div>
                <div style='flex:2; min-width:200px'>
                    <div style='font-weight:600; color:#e2e8f0; margin-bottom:3px'>{meaning}</div>
                    <div style='font-size:0.82rem; color:#94a3b8'><i>Example: {example}</i></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box' style='margin-top:16px'>
    <b>Why does the score rarely reach 1.0?</b><br>
    Random Forests average predictions across 100 decision trees. Even when all trees
    agree a transaction is suspicious, the averaged probability is naturally conservative —
    typically peaking around 0.50–0.60. This is expected behaviour, not a model weakness.
    The decision threshold has been set at <b>0.30</b> to account for this, ensuring
    suspicious transactions in the 0.30–0.50 range are correctly flagged as High Risk.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>📊 Risk Score Distribution Guide</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    Each transaction receives a <b>composite rule-based risk score</b> (0–150) computed from
    seven AML heuristics derived from FATF typologies and regulatory thresholds. The score
    acts as an interpretable complement to the ML probability — a transaction is
    <b>flagged</b> if either the ML probability exceeds 0.30 <b>or</b> the rule score exceeds 70.<br><br>
    The charts below show how each rule contributes to the score and what different score
    totals mean in practice.
    </div>
    """, unsafe_allow_html=True)

    score_conditions = [
        ("Amount > £10,000",             30, "#ef4444"),
        ("High-Frequency Sender (>10)",  25, "#f97316"),
        ("Cross-Border Transaction",     20, "#f59e0b"),
        ("Fan-Out > 10 Receivers",       20, "#eab308"),
        ("Fan-In > 10 Senders",          20, "#84cc16"),
        ("Amount > 3× Sender Average",   20, "#22c55e"),
        ("Currency Mismatch",            15, "#06b6d4"),
    ]

    labels  = [c[0] for c in score_conditions]
    weights = [c[1] for c in score_conditions]
    colours = [c[2] for c in score_conditions]

    rc1, rc2 = st.columns(2)

    # ── LEFT: Individual rule weights ────────────────────────────
    with rc1:
        fig_rules = go.Figure(go.Bar(
            x=weights,
            y=labels,
            orientation='h',
            marker_color=colours,
            marker_line_color='#2e3a50',
            marker_line_width=0.8,
            text=[f'+{w} pts' for w in weights],
            textposition='outside',
            textfont=dict(color='#e2e8f0', size=11)
        ))
        fig_rules.update_layout(
            title=dict(text='Individual Rule Score Weights',
                       font=dict(color='#e2e8f0', size=13)),
            plot_bgcolor='#1a1f2e', paper_bgcolor='#1a1f2e',
            font_color='#a0aec0',
            xaxis=dict(title='Points Awarded', gridcolor='#2e3a50',
                       range=[0, 45], tickfont=dict(color='#a0aec0')),
            yaxis=dict(gridcolor='#2e3a50', tickfont=dict(color='#e2e8f0')),
            margin=dict(l=10, r=60, t=40, b=40),
            height=320
        )
        st.plotly_chart(fig_rules, use_container_width=True)

    with rc2:
        scenarios = {
            "Normal Transaction":                     0,
            "Currency Mismatch Only":                15,
            "Cross-Border Only":                     20,
            "High Amount Only":                      30,
            "High Amount + Cross-Border":            50,
            "High Amount + Cross-Border + Mismatch": 65,
            "High Amount + Freq + Cross-Border":     75,
            "All Rules Fire":                       150,
        }
        s_labels = list(scenarios.keys())
        s_values = list(scenarios.values())
        s_colors = ['#ef4444' if v > 70 else '#f59e0b' if v > 0 else '#22c55e'
                    for v in s_values]

        fig_scen = go.Figure(go.Bar(
            x=s_values,
            y=s_labels,
            orientation='h',
            marker_color=s_colors,
            marker_line_color='#2e3a50',
            marker_line_width=0.8,
            text=[str(v) for v in s_values],
            textposition='outside',
            textfont=dict(color='#e2e8f0', size=11)
        ))
        fig_scen.add_vline(
            x=70, line_dash='dash', line_color='white', line_width=2,
            annotation_text='Flag Threshold (70)',
            annotation_font_color='white',
            annotation_position='top right'
        )
        fig_scen.update_layout(
            title=dict(text='Example Transaction Scenarios',
                       font=dict(color='#e2e8f0', size=13)),
            plot_bgcolor='#1a1f2e', paper_bgcolor='#1a1f2e',
            font_color='#a0aec0',
            xaxis=dict(title='Total Risk Score', gridcolor='#2e3a50',
                       range=[0, 175], tickfont=dict(color='#a0aec0')),
            yaxis=dict(gridcolor='#2e3a50', tickfont=dict(color='#e2e8f0')),
            margin=dict(l=10, r=50, t=40, b=40),
            height=320
        )
        st.plotly_chart(fig_scen, use_container_width=True)

    # ── SCORE BAND TABLE ─────────────────────────────────────────
    st.markdown("""
    <table class='perf-table'>
    <tr><th>Score Range</th><th>Risk Band</th><th>Meaning</th><th>Action</th></tr>
    <tr>
        <td>0 – 14</td>
        <td><span style='color:#22c55e'>🟢 Low</span></td>
        <td>No rule conditions triggered. Normal transactional behaviour.</td>
        <td>No action required</td>
    </tr>
    <tr>
        <td>15 – 69</td>
        <td><span style='color:#f59e0b'>🟡 Medium</span></td>
        <td>One or two risk signals present. Warrants monitoring but not immediate escalation.</td>
        <td>Log and monitor</td>
    </tr>
    <tr>
        <td>70 – 150</td>
        <td><span style='color:#ef4444'>🔴 High</span></td>
        <td>Multiple strong AML indicators. High likelihood of suspicious activity.</td>
        <td>Escalate for review</td>
    </tr>
    </table>
    <br>
    """, unsafe_allow_html=True)


# ============================================================
# TAB 3: SINGLE TRANSACTION PREDICTION
# ============================================================

with tab3:

    st.markdown("<div class='section-header'>🔮 Single Transaction Risk Assessment</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
    Enter raw transaction details. All derived features are computed automatically.
    Fields marked <span style='color:#ef4444'>*</span> are <b>required</b> —
    the model cannot produce a meaningful output without them.
    </div>
    """, unsafe_allow_html=True)

    if not model_loaded:
        st.warning("⚠️ Model PKL files not found. Place best_aml_model.pkl, aml_scaler.pkl, "
                   "and feature_columns.pkl in the same directory as this script.")
    else:
        with st.form("prediction_form"):

            st.markdown("#### 💰 Transaction Details")
            fc1, fc2, fc3 = st.columns(3)

            with fc1:
                amount = st.number_input("Amount * (£/USD)",
                                         min_value=0.01, value=1000.0, step=100.0,
                                         help="Required. Transaction value.")
                payment_currency = st.selectbox("Payment Currency *",
                    ["UK pounds","US Dollar","Euro","Dirham","Yen","Yuan","Ruble","Other"])

            with fc2:
                received_currency = st.selectbox("Received Currency *",
                    ["UK pounds","US Dollar","Euro","Dirham","Yen","Yuan","Ruble","Other"])
                payment_type = st.selectbox("Payment Type *",
                    ["ACH","Cross-border","Cheque","Cash Deposit","Wire Transfer","Other"])

            with fc3:
                sender_location = st.selectbox("Sender Bank Country *",
                    ["UK","UAE","US","Germany","China","Russia","India","Other"])
                receiver_location = st.selectbox("Receiver Bank Country *",
                    ["UK","UAE","US","Germany","China","Russia","India","Other"])

            st.markdown("#### 🕐 Temporal Context")
            tc1, tc2, tc3 = st.columns(3)
            with tc1:
                txn_hour = st.slider("Transaction Hour", 0, 23, 10,
                                     help="Late-night hours can increase risk score.")
            with tc2:
                day_of_week_str = st.selectbox("Day of Week",
                    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
            with tc3:
                is_weekend = 1 if day_of_week_str in ["Saturday","Sunday"] else 0
                st.markdown("<br>", unsafe_allow_html=True)
                st.info(f"Weekend: {'Yes 🔴' if is_weekend else 'No 🟢'}")

            st.markdown("#### 📊 Account Context")
            st.caption("Optional — improves accuracy. Leave at defaults if unknown.")
            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                sender_txn_count   = st.number_input("Sender Total Transactions", 1, 10000, 5)
                receiver_txn_count = st.number_input("Receiver Total Transactions", 1, 10000, 5)
            with ac2:
                fan_out = st.number_input("Sender Fan-Out (unique receivers)", 1, 500, 2,
                                          help="How many unique receivers does this sender use?")
                fan_in  = st.number_input("Receiver Fan-In (unique senders)", 1, 500, 2)
            with ac3:
                amount_vs_avg  = st.number_input("Amount ÷ Sender Average", 0.01, 100.0, 1.0, step=0.1,
                                                  help=">3 is a behavioural anomaly signal.")
                pair_frequency = st.number_input("Sender-Receiver Pair Frequency", 1, 100, 1)

            submitted = st.form_submit_button("🔍  Assess Transaction Risk", use_container_width=True)

        if submitted:
            currency_mismatch = int(payment_currency != received_currency)
            cross_border       = int(sender_location  != receiver_location)
            high_amount        = int(amount > 10000)
            high_frequency     = int(sender_txn_count > 10)
            log_amount         = float(np.log1p(amount))
            dow_int = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(day_of_week_str)

            input_dict = {
                'Amount': amount, 'log_amount': log_amount,
                'currency_mismatch': currency_mismatch, 'cross_border': cross_border,
                'high_amount': high_amount, 'high_frequency': high_frequency,
                'sender_txn_count': sender_txn_count, 'receiver_txn_count': receiver_txn_count,
                'fan_out': fan_out, 'fan_in': fan_in,
                'hour': txn_hour, 'day_of_week': dow_int, 'is_weekend': is_weekend,
                'amount_vs_avg': amount_vs_avg, 'pair_frequency': pair_frequency
            }

            df_in = pd.DataFrame([input_dict])
            for col in EXPECTED_FEATURES:
                if col not in df_in.columns:
                    df_in[col] = 0
            df_in = df_in[EXPECTED_FEATURES]

            try:
                X_sc = scaler.transform(df_in)
                pred = model.predict(X_sc)[0]
                prob = model.predict_proba(X_sc)[0][1]
            except Exception:
                pred = model.predict(df_in)[0]
                prob = model.predict_proba(df_in)[0][1]

            rule_score = (
                high_amount        * 30 +
                cross_border       * 20 +
                currency_mismatch  * 15 +
                high_frequency     * 25 +
                int(fan_out > 10)  * 20 +
                int(fan_in  > 10)  * 20 +
                int(amount_vs_avg > 3) * 20
            )

            final_flag = int(prob > 0.30 or rule_score > 70)

            risk_level = "High" if prob > 0.30 else "Medium" if prob > 0.10 else "Low"
            risk_class = "alert-high" if risk_level == "High" else \
                         "alert-medium" if risk_level == "Medium" else "alert-low"
            risk_icon  = "🔴" if risk_level == "High" else \
                         "🟡" if risk_level == "Medium" else "🟢"

            st.markdown("<div class='section-header'>📋 Risk Assessment Report</div>", unsafe_allow_html=True)

            o1, o2, o3, o4 = st.columns(4)
            o1.metric("ML Fraud Probability  (100% = 0.50)", f"{prob:.4f}")
            o2.metric("Rule-Based Score",     f"{rule_score} / 150")
            o3.metric("ML Prediction",        "⚠️ Suspicious" if prob > 0.30 else "✅ Normal")
            o4.metric("Final Decision",       "🚨 FLAGGED" if final_flag else "✅ CLEAR")

            st.markdown(f"""
            <div class='{risk_class}'>
            {risk_icon} <b>Risk Level: {risk_level}</b> &nbsp;—&nbsp;
            ML Probability = {prob:.4f} &nbsp;|&nbsp;
            Rule Score = {rule_score} / 150 &nbsp;|&nbsp;
            Decision = {"🚨 FLAGGED for investigation" if final_flag else "✅ Transaction cleared"}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div class='section-header'>🔬 Engineered Feature Breakdown</div>", unsafe_allow_html=True)
            breakdown = pd.DataFrame({
                'Feature': ['Amount','log_amount','currency_mismatch','cross_border',
                            'high_amount','high_frequency','sender_txn_count',
                            'receiver_txn_count','fan_out','fan_in',
                            'hour','day_of_week','is_weekend','amount_vs_avg','pair_frequency'],
                'Computed Value': [amount, round(log_amount,4), currency_mismatch, cross_border,
                                   high_amount, high_frequency, sender_txn_count,
                                   receiver_txn_count, fan_out, fan_in,
                                   txn_hour, dow_int, is_weekend, amount_vs_avg, pair_frequency],
                'Risk Signal': [
                    "⚠️ High" if amount > 10000 else "✅ Normal",
                    "—",
                    "⚠️ Mismatch" if currency_mismatch else "✅ Match",
                    "⚠️ Cross-border" if cross_border else "✅ Domestic",
                    "⚠️ Yes" if high_amount else "✅ No",
                    "⚠️ High" if high_frequency else "✅ Normal",
                    "⚠️ High" if sender_txn_count > 10 else "✅ Normal",
                    "⚠️ High" if receiver_txn_count > 10 else "✅ Normal",
                    "⚠️ Wide" if fan_out > 10 else "✅ Normal",
                    "⚠️ Wide" if fan_in > 10 else "✅ Normal",
                    "⚠️ Unusual" if txn_hour < 6 or txn_hour > 22 else "✅ Normal",
                    "—",
                    "⚠️ Weekend" if is_weekend else "✅ Weekday",
                    "⚠️ Anomalous" if amount_vs_avg > 3 else "✅ Normal",
                    "—",
                ],
                'Rule Contribution': [
                    30 if high_amount else 0, 0,
                    15 if currency_mismatch else 0,
                    20 if cross_border else 0,
                    0, 25 if high_frequency else 0,
                    0, 0,
                    20 if fan_out > 10 else 0,
                    20 if fan_in > 10 else 0,
                    0, 0, 0,
                    20 if amount_vs_avg > 3 else 0, 0
                ]
            })
            st.dataframe(breakdown, use_container_width=True, hide_index=True)

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=min(prob * 200, 100),
                number={'suffix': '%', 'font': {'size': 32, 'color': '#e2e8f0'}},
                title={'text': "ML Fraud Probability  (100% = 0.50)", 'font': {'size': 16, 'color': '#a0aec0'}},
                delta={'reference': 10, 'increasing': {'color': '#ef4444'}},
                gauge={
                    'axis': {
                        'range': [0, 100],
                        'tickvals': [0, 20, 40, 60, 80, 100],
                        'ticktext': ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5'],
                        'tickfont': {'color': '#a0aec0'}
                    },
                    'bar': {'color': '#ef4444' if prob > 0.30 else '#f59e0b' if prob > 0.10 else '#22c55e'},
                    'bgcolor': '#1a1f2e', 'bordercolor': '#2e3a50',
                    'steps': [
                        {'range': [0,  20],  'color': 'rgba(20,83,45,0.27)'},
                        {'range': [20, 60],  'color': 'rgba(120,53,15,0.27)'},
                        {'range': [60, 100], 'color': 'rgba(127,29,29,0.27)'}
                    ],
                    'threshold': {'line': {'color': 'white', 'width': 3},
                                  'thickness': 0.75, 'value': 60}
                }
            ))
            fig_gauge.update_layout(paper_bgcolor='#1a1f2e', font_color='#a0aec0', height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)