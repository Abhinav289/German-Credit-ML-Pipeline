import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="🏦",
    layout="wide"
)

# ── Load model ────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('../models/calibrated_pipeline.pkl')

@st.cache_resource
def load_explainer():
    pipeline    = load_model()
    X_train     = pd.read_csv('../data/processed/X_train.csv')
    
    # Extract scaler + xgb from calibrated pipeline
    first_est   = pipeline.calibrated_classifiers_[0].estimator
    scaler      = first_est.named_steps['scaler']
    xgb_model   = first_est.named_steps['model']
    
    X_train_scaled = scaler.transform(X_train)
    explainer      = shap.TreeExplainer(
                        xgb_model,
                        data=X_train_scaled,
                        feature_perturbation='interventional'
                    )
    return explainer, scaler

model    = load_model()
explainer, scaler = load_explainer()

THRESHOLD = 0.612



# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title("🏦 Credit Risk Predictor")
    st.markdown("---")
    
    st.markdown("### 📋 Default Borrower Profile")
    st.markdown(
        """
        From EDA — a borrower **most likely to default** shows:
        
        🔴 Checking account: negative or absent  
        🔴 Credit history: no credits / all paid  
        🔴 Savings: less than 100 DM  
        🔴 No known property (no collateral)  
        🔴 Co-applicant on the loan  
        🔴 Loan duration: 45+ months  
        🔴 Age: 25–30 years  
        
        Compare your inputs against this profile.
        """
    )
    st.markdown("---")
    st.markdown("**Model:** XGBoost + Isotonic Calibration")
    st.markdown("**CV AUC:** 0.890 ± 0.019")
    st.markdown("**Threshold:** 0.612")


    # ── Main ──────────────────────────────────────────────────
st.title("Credit Risk Assessment")
st.markdown(
    "Fill in the borrower details below. "
    "Hover over **ⓘ** labels to understand each field."
)
st.markdown("---")

# ── Input Form ────────────────────────────────────────────
st.subheader("📝 Borrower Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Checking Account Status** ⓘ")
    st.caption("Current balance in borrower's checking account. "
               "Negative balance = existing financial stress.")
    checking_status = st.selectbox(
        "Checking Account",
        options=['no checking', '>=200', '0<=X<200', '<0'],
        label_visibility='collapsed'
    )

    st.markdown("**Loan Duration (months)** ⓘ")
    st.caption("How many months to repay. "
               "Longer duration = more uncertainty = higher risk.")
    duration = st.slider(
        "Duration", min_value=4, max_value=72,
        value=24, label_visibility='collapsed'
    )

    st.markdown("**Savings Account Status** ⓘ")
    st.caption("Balance in savings. "
               "Acts as a financial buffer against income shocks.")
    savings_status = st.selectbox(
        "Savings",
        options=['>=1000', '500<=X<1000', '100<=X<500',
                 '<100', 'no known savings'],
        label_visibility='collapsed'
    )

    st.markdown("**Employment Duration** ⓘ")
    st.caption("How long at current job. "
               "Longer employment = more stable income.")
    employment = st.selectbox(
        "Employment",
        options=['>=7', '4<=X<7', '1<=X<4', '<1', 'unemployed'],
        label_visibility='collapsed'
    )

with col2:
    st.markdown("**Credit Amount (DM)** ⓘ")
    st.caption("Total loan amount requested. "
               "Amount alone is not decisive — ability to repay matters more.")
    credit_amount = st.number_input(
        "Credit Amount", min_value=250,
        max_value=20000, value=3000,
        label_visibility='collapsed'
    )

    st.markdown("**Credit History** ⓘ")
    st.caption("Past repayment behavior. "
               "Strongest behavioral predictor of future default.")
    credit_history = st.selectbox(
        "Credit History",
        options=[
            'critical/other existing credit',
            'existing paid',
            'delayed previously',
            'no credits/all paid',
            'all paid'
        ],
        label_visibility='collapsed'
    )

    st.markdown("**Installment Rate (% of income)** ⓘ")
    st.caption("Monthly repayment as % of disposable income. "
               "Higher % = more financial burden.")
    installment_commitment = st.selectbox(
        "Installment Rate",
        options=[1, 2, 3, 4],
        index=2,
        label_visibility='collapsed'
    )

    st.markdown("**Age** ⓘ")
    st.caption("Borrower's age. "
               "Younger borrowers (25–30) show higher default rates in this dataset.")
    age = st.slider(
        "Age", min_value=19, max_value=75,
        value=35, label_visibility='collapsed'
    )

with col3:
    st.markdown("**Property / Collateral** ⓘ")
    st.caption("Most valuable asset. "
               "Collateral reduces bank's loss if borrower defaults.")
    property_magnitude = st.selectbox(
        "Property",
        options=['real estate', 'life insurance',
                 'car', 'no known property'],
        label_visibility='collapsed'
    )

    st.markdown("**Housing** ⓘ")
    st.caption("Living situation. "
               "Homeowners show more financial stability and commitment.")
    housing = st.selectbox(
        "Housing",
        options=['own', 'rent', 'for free'],
        label_visibility='collapsed'
    )

    st.markdown("**Other Parties** ⓘ")
    st.caption("Co-signer or guarantor on the loan. "
               "A guarantor significantly reduces default risk.")
    other_parties = st.selectbox(
        "Other Parties",
        options=['none', 'guarantor', 'co applicant'],
        label_visibility='collapsed'
    )

    st.markdown("**Personal Status** ⓘ")
    st.caption("Marital and gender status. "
               "Note: fairness audit confirms model treats groups equitably.")
    personal_status = st.selectbox(
        "Personal Status",
        options=['male single', 'female div/dep/mar',
                 'male div/sep', 'male mar/wid'],
        label_visibility='collapsed'
    )

    st.markdown("**Other Payment Plans** ⓘ")
    st.caption("Active installments elsewhere. "
               "Existing EMIs reduce available income for repayment.")
    other_payment_plans = st.selectbox(
        "Other Payment Plans",
        options=['none', 'bank', 'stores'],
        label_visibility='collapsed'
    )

    st.markdown("**Foreign Worker** ⓘ")
    st.caption("Employment as foreign national. "
               "Note: low variance feature — minimal impact on prediction.")
    foreign_worker = st.selectbox(
        "Foreign Worker",
        options=['yes', 'no'],
        label_visibility='collapsed'
    )


   


# ── Initialize session state ──────────────────────────────
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'proba' not in st.session_state:
    st.session_state.proba = None
if 'input_df' not in st.session_state:
    st.session_state.input_df = None
if 'financial_stress_score' not in st.session_state:
    st.session_state.financial_stress_score = None

# ── Predict Button ────────────────────────────────────────
st.markdown("---")
predict_clicked = st.button("🔍 Assess Credit Risk",
                             width='stretch',
                             type="primary")

if predict_clicked:
    # ── Feature engineering (same as before) ──────────────
    checking_map   = {'no checking': 0, '>=200': 1,
                      '0<=X<200': 2, '<0': 3}
    savings_map    = {'>=1000': 0, '500<=X<1000': 1,
                      '100<=X<500': 2, '<100': 3,
                      'no known savings': 4}
    employment_map = {'>=7': 0, '4<=X<7': 1,
                      '1<=X<4': 2, '<1': 3, 'unemployed': 4}

    checking_enc   = checking_map[checking_status]
    savings_enc    = savings_map[savings_status]
    employment_enc = employment_map[employment]

    credit_amount_log = np.log1p(credit_amount)
    monthly_burden    = credit_amount / duration

    checking_stress = {'<0': 3, '0<=X<200': 2,
                       '>=200': 1, 'no checking': 0}
    savings_stress  = {'<100': 3, '100<=X<500': 2,
                       '500<=X<1000': 1, '>=1000': 0,
                       'no known savings': 0}
    financial_stress_score = (
        checking_stress[checking_status] +
        savings_stress[savings_status]
    )

    high_risk_interaction = int(
        checking_status in ['<0', '0<=X<200'] and
        credit_history in ['all paid', 'no credits/all paid']
    )

    purpose_risk = 'low_risk'

    feature_cols = [
        'checking_status', 'duration', 'savings_status',
        'employment', 'installment_commitment', 'age',
        'credit_amount_log', 'monthly_burden',
        'financial_stress_score', 'high_risk_interaction',
        'credit_history_critical/other existing credit',
        'credit_history_delayed previously',
        'credit_history_existing paid',
        'credit_history_no credits/all paid',
        'personal_status_male div/sep',
        'personal_status_male mar/wid',
        'personal_status_male single',
        'other_parties_guarantor',
        'other_parties_none',
        'property_magnitude_life insurance',
        'property_magnitude_no known property',
        'property_magnitude_real estate',
        'other_payment_plans_none',
        'other_payment_plans_stores',
        'housing_own',
        'housing_rent',
        'foreign_worker_yes',
        'purpose_risk_low_risk'
    ]

    input_dict = {col: 0 for col in feature_cols}
    input_dict['checking_status']        = checking_enc
    input_dict['duration']               = duration
    input_dict['savings_status']         = savings_enc
    input_dict['employment']             = employment_enc
    input_dict['installment_commitment'] = installment_commitment
    input_dict['age']                    = age
    input_dict['credit_amount_log']      = credit_amount_log
    input_dict['monthly_burden']         = monthly_burden
    input_dict['financial_stress_score'] = financial_stress_score
    input_dict['high_risk_interaction']  = high_risk_interaction

    for col, val in [
        (f'credit_history_{credit_history}',         1),
        (f'personal_status_{personal_status}',       1),
        (f'other_parties_{other_parties}',           1),
        (f'property_magnitude_{property_magnitude}', 1),
        (f'other_payment_plans_{other_payment_plans}', 1),
        (f'housing_{housing}',                       1),
    ]:
        if col in input_dict:
            input_dict[col] = val

    if foreign_worker == 'yes':
        input_dict['foreign_worker_yes'] = 1
    if purpose_risk == 'low_risk':
        input_dict['purpose_risk_low_risk'] = 1

    input_df = pd.DataFrame([input_dict])
    proba    = model.predict_proba(input_df)[0][1]

    # ── Store everything in session_state ─────────────────
    st.session_state.prediction_made      = True
    st.session_state.proba                = proba
    st.session_state.input_df             = input_df
    st.session_state.financial_stress_score = financial_stress_score
    st.session_state.feature_cols         = feature_cols


# ── Show results if prediction exists in session_state ────
if st.session_state.prediction_made:

    proba                  = st.session_state.proba
    input_df               = st.session_state.input_df
    financial_stress_score = st.session_state.financial_stress_score
    feature_cols           = st.session_state.feature_cols
    risk_pct               = round(proba * 100, 1)

    # ── Traffic Light Verdict ─────────────────────────────
    st.markdown("---")
    st.subheader("📊 Assessment Result")

    if proba < THRESHOLD:
        verdict       = "✅ LOW RISK"
        verdict_color = "green"
        advice        = "Borrower profile suggests acceptable credit risk."
    elif proba < 0.75:
        verdict       = "⚠️ MODERATE RISK"
        verdict_color = "orange"
        advice        = "Borderline profile — recommend additional review."
    else:
        verdict       = "🔴 HIGH RISK"
        verdict_color = "red"
        advice        = "Borrower profile suggests high likelihood of default."

    r1, r2, r3 = st.columns([1, 1, 1])
    with r2:
        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = risk_pct,
            title = {'text': "Default Probability (%)"},
            gauge = {
                'axis':  {'range': [0, 100]},
                'bar':   {'color': verdict_color},
                'steps': [
                    {'range': [0,  40],  'color': '#d4edda'},
                    {'range': [40, 65],  'color': '#fff3cd'},
                    {'range': [65, 100], 'color': '#f8d7da'},
                ],
                'threshold': {
                    'line':      {'color': 'black', 'width': 4},
                    'thickness': 0.75,
                    'value':     THRESHOLD * 100
                }
            }
        ))
        fig_gauge.update_layout(height=300,
                                 margin=dict(t=50, b=0))
        st.plotly_chart(fig_gauge,width='stretch')

        st.markdown(
            f"<h2 style='text-align:center; color:{verdict_color}'>"
            f"{verdict}</h2>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='text-align:center'>{advice}</p>",
            unsafe_allow_html=True
        )

    # ── Financial Stress Score ────────────────────────────
    st.markdown("---")
    st.subheader("💰 Financial Stress Score")
    stress_col1, stress_col2 = st.columns([1, 2])
    with stress_col1:
        st.metric(
            label="Your Financial Stress Score",
            value=f"{financial_stress_score} / 6",
            delta=(f"Default rate at this score: "
                   f"{[8.87,13.21,15.69,18.18,37.50,43.90,52.05][min(financial_stress_score,6)]}%"),
            delta_color="inverse"
        )
    with stress_col2:
        st.progress(financial_stress_score / 6)
        st.caption(
            "Score 0 = lowest stress (8.87% default rate) | "
            "Score 6 = highest stress (52.05% default rate)"
        )

    # ── SHAP Explanation ──────────────────────────────────
    st.markdown("---")
    if st.button("🔍 Explain this prediction"):

        with st.spinner("Computing SHAP explanation..."):
            input_scaled       = scaler.transform(input_df)
            shap_vals          = explainer(input_scaled)
            shap_values_single = shap_vals.values[0]

            shap_df = pd.DataFrame({
                'feature':    feature_cols,
                'shap_value': shap_values_single
            }).sort_values('shap_value', key=abs,
                           ascending=False).head(10)

            colors = ['crimson' if v > 0 else 'steelblue'
                      for v in shap_df['shap_value']]

            fig_shap = go.Figure(go.Bar(
                x=shap_df['shap_value'],
                y=shap_df['feature'],
                orientation='h',
                marker_color=colors
            ))
            fig_shap.update_layout(
                title="Top 10 Features Driving This Prediction",
                xaxis_title=("SHAP Value "
                             "(🔴 pushes toward default | "
                             "🔵 pushes toward safe)"),
                template='plotly_dark',
                height=450
            )
            st.plotly_chart(fig_shap, width='stretch')

            # Plain English
            st.subheader("📖 Plain English Explanation")
            top_risk    = shap_df[shap_df['shap_value'] > 0].head(3)
            top_protect = shap_df[shap_df['shap_value'] < 0].head(3)

            if not top_risk.empty:
                st.markdown("**Factors increasing default risk:**")
                for _, row in top_risk.iterrows():
                    st.markdown(
                        f"🔴 `{row['feature']}` "
                        f"(impact: +{row['shap_value']:.3f})"
                    )
            if not top_protect.empty:
                st.markdown("**Factors reducing default risk:**")
                for _, row in top_protect.iterrows():
                    st.markdown(
                        f"🔵 `{row['feature']}` "
                        f"(impact: {row['shap_value']:.3f})"
                    )
    

    



