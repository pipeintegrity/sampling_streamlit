import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Sampling Statistics Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styles mimicking the original Tailwind template
st.markdown("""
    <style>
    .main { background-color: #F8FAFC; }
    div[data-testid="stSidebar"] { background-color: #F1F5F9; }
    h1, h2, h3 { color: #0F172A; font-family: 'Helvetica Neue', Arial, sans-serif; }
    </style>
""", unsafe_allow_html=True)

# --- Shared UI Components ---
def result_card(title, value, unit="", description="", variant="default"):
    styles = {
        "default": {"bg": "#EEF2FF", "border": "#C7D2FE", "text": "#3730A3"},
        "warning": {"bg": "#FFFBEB", "border": "#FDE68A", "text": "#92400E"},
        "danger": {"bg": "#FFF1F2", "border": "#FECDD3", "text": "#BE123C"},
        "success": {"bg": "#ECFDF5", "border": "#A7F3D0", "text": "#047857"}
    }
    s = styles.get(variant, styles["default"])
    html = f"""
    <div style="background-color: {s['bg']}; border: 1px solid {s['border']}; color: {s['text']}; padding: 20px; border-radius: 12px; margin-bottom: 15px;">
        <h3 style="font-size: 11px; font-weight: bold; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; opacity: 0.8; color: {s['text']};">{title}</h3>
        <div style="display: flex; align-items: baseline; gap: 6px;">
            <span style="font-size: 28px; font-weight: 900;">{value}</span>
            <span style="font-size: 14px; font-weight: 500; opacity: 0.8;">{unit}</span>
        </div>
        {f'<p style="margin-top: 12px; color: #475569; font-size: 13px; line-height: 1.5;">{description}</p>' if description else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def insight_box(title, content_html):
    html = f"""
    <div style="background-color: #FFFBEB; border-left: 4px solid #F59E0B; padding: 20px; border-radius: 0 12px 12px 0; margin-top: 20px; margin-bottom: 20px;">
        <div style="display: flex; align-items: center; gap: 8px; color: #78350F; font-weight: bold; margin-bottom: 12px;">
            <span>ℹ️</span> <strong>{title}</strong>
        </div>
        <div style="color: #334155; font-size: 13.5px; line-height: 1.6;">
            {content_html}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# Initialize data context to capture export strings
if "report_data" not in st.session_state:
    st.session_state.report_data = {}

# --- NAVIGATION SIDEBAR ---
st.sidebar.markdown(
    '<div style="background: linear-gradient(135deg, #4338CA, #1E1B4B); padding: 20px; margin: -20px -20px 20px -20px; color: white; border-radius: 0 0 12px 12px;">'
    '<h2 style="color: white; margin: 0; font-size: 18px; font-weight: 900;">Sampling Design</h2>'
    '<p style="color: #C7D2FE; margin: 0; font-size: 12px;">Statistical Tools</p>'
    '</div>', 
    unsafe_allow_html=True
)

category = st.sidebar.selectbox(
    "Select Category",
    ["Hypothesis Testing", "Sample Planning", "Intervals", "Proportions", "Distributions"]
)

if category == "Hypothesis Testing":
    module = st.sidebar.radio("Select Tool", ["T-Test", "Power Analysis", "ANOVA / F-Test", "Chi-Square"])
elif category == "Sample Planning":
    module = st.sidebar.radio("Select Tool", ["Sample Size Determination", "OC Curves"])
elif category == "Intervals":
    module = st.sidebar.radio("Select Tool", ["Confidence Intervals", "Tolerance Intervals"])
elif category == "Proportions":
    module = st.sidebar.radio("Select Tool", ["Binomial Confidence Int.", "Bayesian Binomial", "Bayes' Theorem"])
else:
    module = st.sidebar.radio("Select Tool", ["Poisson", "Normal", "Binomial"])

st.sidebar.markdown("---")

# --- MODULE MODULE LOGIC ---

if module == "T-Test":
    st.title("T-Test Calculation")
    st.caption("Determine if group means differ significantly.")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        mode = st.radio("Test Type", ["One-Sample", "Two-Sample (Welch)"])
        tail = st.radio("Tails", ["Two-Tailed", "One-Tailed"])
        tailDir = "greater"
        if tail == "One-Tailed":
            tailDir = st.selectbox("Direction", ["Greater Than (μ₁ > μ₂)", "Less Than (μ₁ < μ₂)"])
            tailDir = "greater" if "Greater" in tailDir else "less"
            
        alpha = st.number_input("Alpha (α)", value=0.05, min_value=0.001, max_value=0.5, step=0.01)
        
        st.markdown("**Sample 1**")
        m1 = st.number_input("Mean (Sample 1)", value=105.0)
        sd1 = st.number_input("Std Dev (Sample 1)", value=15.0, min_value=0.001)
        n1 = st.number_input("n (Sample 1)", value=30, min_value=2, step=1)
        
        if mode == "One-Sample":
            st.markdown("**Target Population**")
            m2 = st.number_input("Population Mean (μ₀)", value=100.0)
            sd2, n2 = 0, 0
        else:
            st.markdown("**Sample 2**")
            m2 = st.number_input("Mean (Sample 2)", value=100.0)
            sd2 = st.number_input("Std Dev (Sample 2)", value=15.0, min_value=0.001)
            n2 = st.number_input("n (Sample 2)", value=30, min_value=2, step=1)
            
    # Computations
    if mode == "One-Sample":
        df = max(n1 - 1, 1)
        t_stat = (m1 - m2) / (sd1 / np.sqrt(n1))
    else:
        v1 = (sd1 ** 2) / n1
        v2 = (sd2 ** 2) / n2
        df = ((v1 + v2) ** 2) / ((v1 ** 2) / (n1 - 1) + (v2 ** 2) / (n2 - 1))
        t_stat = (m1 - m2) / np.sqrt(v1 + v2)
        
    p_two = 2 * stats.t.sf(np.abs(t_stat), df)
    p_greater = stats.t.sf(t_stat, df)
    p_less = stats.t.cdf(t_stat, df)
    
    if tail == "Two-Tailed":
        p_val = p_two
        hyp_label = "H₁: μ₁ ≠ μ₂" if mode == "Two-Sample (Welch)" else "H₁: μ₁ ≠ μ₀"
    elif tailDir == "greater":
        p_val = p_greater
        hyp_label = "H₁: μ₁ > μ₂" if mode == "Two-Sample (Welch)" else "H₁: μ₁ > μ₀"
    else:
        p_val = p_less
        hyp_label = "H₁: μ₁ < μ₂" if mode == "Two-Sample (Welch)" else "H₁: μ₁ < μ₀"
        
    significant = p_val < alpha
    
    with col2:
        # Visualizer
        x_axis = np.linspace(-4, 4, 200)
        y_axis = stats.t.pdf(x_axis, df)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_axis, y=y_axis, mode='lines', name='t-dist', line=dict(color='#6366f1', width=2)))
        
        # Shade rejection region
        if tail == "Two-Tailed":
            crit = stats.t.ppf(1 - alpha/2, df)
            fig.add_vrect(x0=-4, x1=-crit, fillcolor="#fee2e2", opacity=0.5, layer="below", line_width=0)
            fig.add_vrect(x0=crit, x1=4, fillcolor="#fee2e2", opacity=0.5, layer="below", line_width=0)
        elif tailDir == "greater":
            crit = stats.t.ppf(1 - alpha, df)
            fig.add_vrect(x0=crit, x1=4, fillcolor="#fee2e2", opacity=0.5, layer="below", line_width=0)
        else:
            crit = stats.t.ppf(alpha, df)
            fig.add_vrect(x0=-4, x1=crit, fillcolor="#fee2e2", opacity=0.5, layer="below", line_width=0)
            
        if -4 <= t_stat <= 4:
            fig.add_vline(x=t_stat, line_dash="dash", line_color="#ef4444", line_width=2)
            fig.add_trace(go.Scatter(x=[t_stat], y=[stats.t.pdf(t_stat, df)], mode='markers', marker=dict(color='#ef4444', size=8), showlegend=False))
            
        fig.update_layout(title="Distribution Visualization", height=250, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            result_card("T-Statistic", f"{t_stat:.3f}", description=hyp_label)
        with sub_col2:
            result_card("P-Value", f"{p_val:.4f}", variant="success" if significant else "default", description="Identified tail risk outcome.")
            
        result_card("H₀ Decision", "Reject H₀" if significant else "Fail to Reject H₀", variant="success" if significant else "warning", description="Difference is statistically significant." if significant else "Difference is likely due to chance.")
        result_card("Degrees of Freedom", f"{df:.1f}")
        
        insight_box("Understanding the T-Test", """
            A T-test evaluates whether the difference between means is statistically significant or merely due to random chance.
            <ul>
                <li><strong>One-Sample:</strong> Compares your sample mean against a known target mean.</li>
                <li><strong>Two-Sample (Welch):</strong> Compares two independent groups without assuming equal variances.</li>
            </ul>
        """)

    st.session_state.report_data = {
        "Module": "T-Test", "Mode": mode, "T-Stat": f"{t_stat:.3f}", "P-Value": f"{p_val:.4f}", "Decision": "Reject H0" if significant else "Fail to Reject H0"
    }

elif module == "Sample Size Determination":
    st.title("Sample Size Determination")
    st.caption("Minimum sample to detect a true effect with specified power and confidence.")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        conf = st.number_input("Confidence Level (1-α)", value=0.95, min_value=0.5, max_value=0.999)
        power = st.number_input("Target Power (1-β)", value=0.80, min_value=0.5, max_value=0.999)
        error = st.number_input("Detectable Difference (E)", value=0.05, min_value=0.001)
        p = st.number_input("Estimated Rate (p)", value=0.50, min_value=0.001, max_value=0.999)
        pop = st.number_input("Population (N)", value=0, min_value=0, step=1, help="Enter 0 for an infinite / very large population.")
        
    z_alpha = np.abs(stats.norm.ppf((1 - conf) / 2))
    z_beta = np.abs(stats.norm.ppf(power))
    n0 = ((z_alpha + z_beta) ** 2 * p * (1 - p)) / (error ** 2)
    
    if pop > 0:
        n_result = int(np.ceil(n0 / (1 + (n0 - 1) / pop)))
    else:
        n_result = int(np.ceil(n0))
        
    with col2:
        result_card("Required n", f"{n_result:,}", unit="Units")
        insight_box("Strategic Design", """
            This calculation considers both Type I error (Confidence) and Type II error (Power) to ensure your study is not "underpowered."<br>
            If the population is at least 10 times larger than the sample size, you can safely use the infinite population formula.
        """)
        
    st.session_state.report_data = {
        "Module": "Sample Size Determination", "Confidence": conf, "Power": power, "Margin of Error": error, "Required Sample Size": n_result
    }

elif module == "OC Curves":
    st.title("OC Curves")
    st.caption("Probability of lot acceptance versus defect rate.")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        n = st.number_input("Sample Size (n)", value=50, min_value=1, step=1)
        c = st.number_input("Acceptance Number (c)", value=2, min_value=0, step=1)
        maxDefect = st.number_input("LTPD / RQL", value=0.08, min_value=0.0, max_value=1.0)
        
        beta_risk = stats.binom.cdf(c, n, maxDefect)
        result_card("Consumer's Risk (β)", f"{beta_risk*100:.2f}", unit="%", variant="danger")
        
    with col2:
        p_vals = np.linspace(0, 0.25, 100)
        pa_vals = stats.binom.cdf(c, n, p_vals)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=p_vals, y=pa_vals, mode='lines', name='Performance', line=dict(color='#6366f1', width=3)))
        fig.add_trace(go.Scatter(x=[maxDefect], y=[beta_risk], mode='markers+text', text=["β Risk"], textposition="top right", marker=dict(color='#f43f5e', size=10)))
        fig.update_layout(title="Acceptance Probability P(a) vs Defect Rate (p)", height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        insight_box("Operating Characteristics", "The OC curve shows how well a sampling plan $(n, c)$ discriminates between good and bad lots. The Consumer's Risk ($\\beta$) evaluates probability of acceptance at LTPD.")

    st.session_state.report_data = {
        "Module": "OC Curves", "Sample Size n": n, "Acceptance c": c, "LTPD": maxDefect, "Beta Risk": f"{beta_risk*100:.2f}%"
    }

elif module == "Confidence Intervals":
    st.title("Confidence Intervals")
    st.caption("Range likely to contain the true population mean.")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        mean = st.number_input("Mean", value=100.0)
        sd = st.number_input("Std Dev", value=15.0, min_value=0.001)
        n = st.number_input("n", value=30, min_value=1, step=1)
        conf = st.number_input("Confidence Level", value=0.95, min_value=0.01, max_value=0.999)
        
    moe = np.abs(stats.norm.ppf((1 - conf) / 2)) * (sd / np.sqrt(n))
    
    with col2:
        sub1, sub2 = st.columns(2)
        with sub1:
            result_card("Lower bound", f"{mean - moe:.2f}")
        with sub2:
            result_card("Upper bound", f"{mean + moe:.2f}")
        result_card("Margin of Error", f"{moe:.3f}")
        
        insight_box("Interpreting a Confidence Interval", "Correct approach: If you repeated this procedure many times, 95% of those intervals would capture the true mean parameter.")

    st.session_state.report_data = {
        "Module": "Confidence Intervals", "Sample Mean": mean, "Lower CI": f"{mean - moe:.2f}", "Upper CI": f"{mean + moe:.2f}"
    }

elif module == "Power Analysis":
    st.title("Power Analysis")
    st.caption("Probability of detecting a true difference between proportions.")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        mode = st.radio("Test Type", ["One-Sample", "Two-Sample"])
        p1 = st.number_input("p1 (Null Proportion)", value=0.50, min_value=0.0, max_value=1.0)
        p2 = st.number_input("p2 (Alternative Proportion)", value=0.55, min_value=0.0, max_value=1.0)
        n = st.number_input("Sample Size (n per group)", value=100, min_value=1, step=1)
        alpha = st.number_input("Alpha (α)", value=0.05, min_value=0.001, max_value=0.5)
        
    h = np.abs(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))
    factor = np.sqrt(n / 2) if mode == "Two-Sample" else np.sqrt(n)
    current_power = stats.norm.cdf((h * factor) - np.abs(stats.norm.ppf(alpha / 2)))
    
    with col2:
        result_card("Statistical Power", f"{current_power:.3f}", variant="success" if current_power >= 0.8 else "warning")
        result_card("Effect Size (Cohen's h)", f"{h:.4f}")
        insight_box("Power Dynamics", "Low sample sizes result in underpowered studies, increasing the risk of the Winner's Curse where observed effects appear artificially inflated.")

    st.session_state.report_data = {
        "Module": "Power Analysis", "Mode": mode, "Calculated Power": f"{current_power:.3f}", "Effect Size h": f"{h:.4f}"
    }

elif module == "Binomial Confidence Int.":
    st.title("Binomial Confidence Interval")
    st.caption("Confidence interval for a proportion metric.")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        x = st.number_input("Successes (x)", value=15, min_value=0, step=1)
        n = st.number_input("Trials (n)", value=100, min_value=1, step=1)
        conf = st.number_input("Confidence Level", value=0.95, min_value=0.01, max_value=0.999)
        
    p = x / n
    moe = np.abs(stats.norm.ppf((1 - conf) / 2)) * np.sqrt((p * (1 - p)) / n)
    
    with col2:
        result_card("Observed Proportion", f"{p*100:.1f}", unit="%")
        sub1, sub2 = st.columns(2)
        with sub1:
            result_card("Lower Bound (Wald)", f"{max(0.0, p - moe)*100:.1f}", unit="%")
        with sub2:
            result_card("Upper Bound (Wald)", f"{min(1.0, p + moe)*100:.1f}", unit="%")
        insight_box("Proportion Estimation", "The classical normal Wald method may fail near boundaries (0 or 1). Consider switching rules if proportions are highly skewed.")

    st.session_state.report_data = {
        "Module": "Binomial Confidence Interval", "Observed Proportion": f"{p*100:.1f}%", "Lower Bound": f"{max(0.0, p - moe)*100:.1f}%", "Upper Bound": f"{min(1.0, p + moe)*100:.1f}%"
    }

elif module == "Bayesian Binomial":
    st.title("Bayesian Credible Intervals")
    st.caption("Update Beta prior with observed attribute data stream.")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        priorA = st.number_input("Prior α", value=0.5, min_value=0.001)
        priorB = st.number_input("Prior β", value=0.5, min_value=0.001)
        x = st.number_input("Observed Successes (x)", value=15, min_value=0, step=1)
        n = st.number_input("Total Trials (n)", value=100, min_value=1, step=1)
        
    postA = priorA + x
    postB = priorB + (n - x)
    lo = stats.beta.ppf(0.025, postA, postB)
    hi = stats.beta.ppf(0.975, postA, postB)
    postMean = postA / (postA + postB)
    
    with col2:
        x_axis = np.linspace(0, 1, 200)
        y_prior = stats.beta.pdf(x_axis, priorA, priorB)
        y_post = stats.beta.pdf(x_axis, postA, postB)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_axis, y=y_prior, mode='lines', name='Prior', line=dict(color='#94a3b8', dash='dash')))
        fig.add_trace(go.Scatter(x=x_axis, y=y_post, mode='lines', name='Posterior', line=dict(color='#6366f1', width=3.5)))
        fig.update_layout(title="Prior vs Posterior Distribution", height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        sub1, sub2 = st.columns(2)
        with sub1:
            result_card("95% Credible Lower", f"{lo*100:.2f}", unit="%")
        with sub2:
            result_card("95% Credible Upper", f"{hi*100:.2f}", unit="%")
        result_card("Posterior Mean", f"{postMean*100:.2f}", unit="%")

    st.session_state.report_data = {
        "Module": "Bayesian Binomial", "Posterior Alpha": postA, "Posterior Beta": postB, "Posterior Mean": f"{postMean*100:.2f}%"
    }

elif module == "Bayes' Theorem":
    st.title("Bayes' Theorem")
    st.caption("Posterior probability calculation updating base rates.")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        prior = st.number_input("Prior P(A)", value=0.01, min_value=0.0, max_value=1.0, step=0.005)
        sens = st.number_input("Sensitivity P(B|A)", value=0.95, min_value=0.0, max_value=1.0, step=0.01)
        fpr = st.number_input("False Positive Rate P(B|¬A)", value=0.05, min_value=0.0, max_value=1.0, step=0.01)
        
    post = (sens * prior) / ((sens * prior) + (fpr * (1 - prior)))
    
    with col2:
        st.latex(r"P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B|A) \cdot P(A) + P(B|\neg A) \cdot P(\neg A)}")
        result_card("Posterior P(A|B)", f"{post*100:.2f}", unit="%")
        insight_box("The Base Rate Fallacy", "When base rates are extremely low, even tests with high sensitivity yield surprisingly high quantities of false positives.")

    st.session_state.report_data = {
        "Module": "Bayes Theorem", "Prior": prior, "Sensitivity": sens, "Posterior P(A|B)": f"{post*100:.2f}%"
    }

elif module == "Tolerance Intervals":
    st.title("Tolerance Intervals")
    st.caption("Statistical bounds capturing a fixed population proportion.")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        mean = st.number_input("Sample Mean", value=10.0)
        sd = st.number_input("Sample Std Dev", value=0.5, min_value=0.001)
        confLevel = st.number_input("Confidence Level (1-α)", value=0.95, min_value=0.01, max_value=0.999)
        popProp = st.number_input("Population Proportion (p)", value=0.99, min_value=0.01, max_value=0.999)
        n = st.number_input("Sample Size (n)", value=50, min_value=2, step=1)
        
    zp = np.abs(stats.norm.ppf((1 + popProp) / 2))
    zg = np.abs(stats.norm.ppf(1 - confLevel))
    k = zp * np.sqrt(1 + 1 / n) * (1 + (zg ** 2) / (2 * n - 2))
    
    with col2:
        result_card("Tolerance Factor k", f"{k:.4f}")
        sub1, sub2 = st.columns(2)
        with sub1:
            result_card("Lower Bound", f"{mean - k * sd:.3f}")
        with sub2:
            result_card("Upper Bound", f"{mean + k * sd:.3f}")
        insight_box("Tolerance vs Confidence Bounds", "Confidence intervals target process <em>parameters</em> (the mean), while Tolerance Intervals isolate the location of <em>individual pieces</em>.")

    st.session_state.report_data = {
        "Module": "Tolerance Intervals", "Tolerance Factor k": f"{k:.4f}", "Lower Bound": f"{mean - k * sd:.3f}", "Upper Bound": f"{mean + k * sd:.3f}"
    }

elif module == "ANOVA / F-Test":
    st.title("One-Way ANOVA")
    st.caption("Compare means across three or more independent groups.")
    st.markdown("---")
    
    st.markdown("##### Input Group Data Settings")
    
    # Initialize session state dataframe for ANOVA input if not present
    if "anova_df" not in st.session_state:
        st.session_state.anova_df = pd.DataFrame([
            {"Group": "Group 1", "Mean": 12.5, "Std Dev": 2.1, "n": 20},
            {"Group": "Group 2", "Mean": 14.8, "Std Dev": 2.4, "n": 20},
            {"Group": "Group 3", "Mean": 13.2, "Std Dev": 1.9, "n": 20}
        ])
        
    edited_df = st.data_editor(st.session_state.anova_df, num_rows="dynamic", use_container_width=True)
    st.session_state.anova_df = edited_df
    
    alpha = st.number_input("Alpha (α)", value=0.05, min_value=0.001, max_value=0.5)
    
    if len(edited_df) >= 2:
        try:
            k = len(edited_df)
            ns = edited_df['n'].astype(float).values
            means = edited_df['Mean'].astype(float).values
            sds = edited_df['Std Dev'].astype(float).values
            
            N = np.sum(ns)
            grandMean = np.sum(ns * means) / N
            ssBetween = np.sum(ns * (means - grandMean) ** 2)
            ssWithin = np.sum((ns - 1) * (sds ** 2))
            
            dfBetween = k - 1
            dfWithin = int(N - k)
            
            msBetween = ssBetween / dfBetween
            msWithin = ssWithin / dfWithin
            
            fStat = msBetween / msWithin
            pVal = stats.f.sf(fStat, dfBetween, dfWithin)
            etaSq = ssBetween / (ssBetween + ssWithin)
            significant = pVal < alpha
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### ANOVA Table Summary")
                anova_table = pd.DataFrame({
                    "Source": ["Between Groups", "Within Groups"],
                    "SS": [f"{ssBetween:.3f}", f"{ssWithin:.3f}"],
                    "df": [dfBetween, dfWithin],
                    "MS": [f"{msBetween:.3f}", f"{msWithin:.3f}"],
                    "F": [f"{fStat:.3f}", "—"]
                })
                st.table(anova_table)
                
            with col2:
                sub1, sub2 = st.columns(2)
                with sub1:
                    result_card("F-Statistic", f"{fStat:.3f}", description=f"df1={dfBetween}, df2={dfWithin}")
                with sub2:
                    result_card("P-Value", f"{pVal:.4f}", variant="success" if significant else "default")
                result_card("H₀ Decision", "Reject H₀" if significant else "Fail to Reject H₀", variant="success" if significant else "warning")
                result_card("Effect Size (η²)", f"{etaSq:.3f}", description="Proportion of variance explained by groups.")
        except Exception as e:
            st.error("Please verify all data inputs are complete numbers greater than zero.")
            
    st.session_state.report_data = {"Module": "ANOVA", "F-Stat": f"{fStat:.3f}" if 'fStat' in locals() else "Error"}

elif module == "Chi-Square":
    st.title("Chi-Square Test")
    st.caption("Test for goodness of fit or independence across categorical distributions.")
    st.markdown("---")
    
    mode = st.radio("Test Variant Type", ["Goodness of Fit", "Test of Independence"])
    alpha = st.number_input("Alpha (α)", value=0.05, min_value=0.001, max_value=0.5)
    
    if mode == "Goodness of Fit":
        if "gof_df" not in st.session_state:
            st.session_state.gof_df = pd.DataFrame([
                {"Category": "Category A", "Observed": 30, "Expected": 25},
                {"Category": "Category B", "Observed": 20, "Expected": 25},
                {"Category": "Category C", "Observed": 28, "Expected": 25},
                {"Category": "Category D", "Observed": 22, "Expected": 25}
            ])
        gof_ed = st.data_editor(st.session_state.gof_df, num_rows="dynamic", use_container_width=True)
        st.session_state.gof_df = gof_ed
        
        obs = gof_ed['Observed'].astype(float).values
        exp = gof_ed['Expected'].astype(float).values
        
        if len(obs) > 1 and np.all(exp > 0):
            chi2, p_val = stats.chisquare(f_obs=obs, f_exp=exp)
            df = len(obs) - 1
            significant = p_val < alpha
            
            col1, col2 = st.columns(2)
            with col1:
                result_card("χ² Statistic", f"{chi2:.3f}", description=f"Degrees of freedom: {df}")
            with col2:
                result_card("P-Value", f"{p_val:.4f}", variant="success" if significant else "default")
                result_card("H₀ Decision", "Reject H₀" if significant else "Fail to Reject H₀", variant="success" if significant else "warning")

    else:
        st.markdown("##### Contingency Table Matrix Editor")
        if "indep_df" not in st.session_state:
            st.session_state.indep_df = pd.DataFrame([[40, 20], [15, 35]], columns=["Col A", "Col B"], index=["Row A", "Row B"])
            
        indep_ed = st.data_editor(st.session_state.indep_df, num_rows="dynamic", use_container_width=True)
        st.session_state.indep_df = indep_ed
        
        try:
            matrix = indep_ed.astype(float).values
            if matrix.shape[0] >= 2 and matrix.shape[1] >= 2:
                chi2, p_val, df, expected_matrix = stats.chi2_contingency(matrix)
                n_total = np.sum(matrix)
                min_dim = min(matrix.shape) - 1
                cramers_v = np.sqrt(chi2 / (n_total * min_dim)) if n_total > 0 and min_dim > 0 else 0
                significant = p_val < alpha
                
                col1, col2 = st.columns(2)
                with col1:
                    result_card("χ² Statistic", f"{chi2:.3f}", description=f"Degrees of freedom: {df}")
                    result_card("Cramér's V (Effect Size)", f"{cramers_v:.3f}", description="Strength: 0.1 small, 0.3 medium, 0.5+ strong.")
                with col2:
                    result_card("P-Value", f"{p_val:.4f}", variant="success" if significant else "default")
                    result_card("H₀ Decision", "Reject H₀" if significant else "Fail to Reject H₀", variant="success" if significant else "warning")
        except Exception as e:
            st.warning("Ensure the table matrix is populated strictly with integers.")

    st.session_state.report_data = {"Module": "Chi-Square", "Variant": mode}

elif module == "Poisson":
    st.title("Poisson Distribution")
    st.caption("Probability of event counts observed over a fixed benchmark interval.")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        lam = st.number_input("Rate λ (lambda)", value=4.0, min_value=0.001)
        k = st.number_input("k (Number of Events)", value=3, min_value=0, step=1)
        
        pmf = stats.poisson.pmf(k, lam)
        cdf = stats.poisson.cdf(k, lam)
        
        result_card("P(X = k)", f"{pmf*100:.3f}", unit="%")
        result_card("P(X ≤ k)", f"{cdf*100:.3f}", unit="%")
        result_card("P(X ≥ k)", f"{(1 - stats.poisson.cdf(k - 1, lam))*100:.3f}", unit="%")
        
    with col2:
        max_k = max(20, int(np.ceil(lam * 3)))
        ks = np.arange(0, max_k + 1)
        ps = stats.poisson.pmf(ks, lam)
        
        fig = px.bar(x=ks, y=ps, labels={'x': 'k Events', 'y': 'Probability Mass'}, title="Probability Mass Function")
        fig.update_traces(marker_color=['#6366f1' if x == k else '#c7d2fe' for x in ks])
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.session_state.report_data = {"Module": "Poisson", "Lambda": lam, "k": k, "PMF": f"{pmf*100:.3f}%"}

elif module == "Normal":
    st.title("Normal Distribution")
    st.caption("Probabilities extracted under a continuous Gaussian curve.")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        mu = st.number_input("Mean (μ)", value=0.0)
        sigma = st.number_input("Std Dev (σ)", value=1.0, min_value=0.001)
        x = st.number_input("x (Value of Interest)", value=1.0)
        
        z = (x - mu) / sigma
        ple = stats.norm.cdf(z)
        pge = 1 - ple
        pdf_val = stats.norm.pdf(x, mu, sigma)
        
        sub1, sub2 = st.columns(2)
        with sub1:
            result_card("P(X ≤ x)", f"{ple*100:.3f}", unit="%")
        with sub2:
            result_card("P(X > x)", f"{pge*100:.3f}", unit="%")
        result_card("z-score", f"{z:.3f}", description=f"x is {abs(z):.2f} standard deviations away from the mean.")
        result_card("PDF Density f(x)", f"{pdf_val:.4f}")
        
    with col2:
        x_vals = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
        y_vals = stats.norm.pdf(x_vals, mu, sigma)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='PDF', line=dict(color='#6366f1', width=2.5)))
        
        # Shade area under curve
        shade_x = x_vals[x_vals <= x]
        shade_y = y_vals[x_vals <= x]
        fig.add_trace(go.Scatter(x=np.concatenate(([shade_x[0]], shade_x, [shade_x[-1]])), y=np.concatenate(([0], shade_y, [0])), fill='toself', fillcolor='rgba(99, 102, 241, 0.2)', line=dict(color='rgba(255,255,255,0)'), showlegend=False))
        
        fig.update_layout(title="Probability Density Function Area Shading", height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.session_state.report_data = {"Module": "Normal", "Mean": mu, "Sigma": sigma, "Value x": x, "Z-Score": f"{z:.3f}"}

elif module == "Binomial":
    st.title("Binomial Distribution")
    st.caption("Probability of tracking k successes in n independent Bernoulli trials.")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        n = st.number_input("Trials (n)", value=20, min_value=1, step=1)
        p = st.number_input("Success Probability (p)", value=0.30, min_value=0.0, max_value=1.0)
        k = st.number_input("k (Successes of Interest)", value=5, min_value=0, max_value=int(n), step=1)
        
        pmf = stats.binom.pmf(k, n, p)
        cdf = stats.binom.cdf(k, n, p)
        mean = n * p
        sd = np.sqrt(n * p * (1 - p))
        
        result_card("P(X = k)", f"{pmf*100:.3f}", unit="%")
        result_card("P(X ≤ k)", f"{cdf*100:.3f}", unit="%")
        result_card("Mean / Std Dev", f"{mean:.2f} / {sd:.2f}")
        
    with col2:
        ks = np.arange(0, n + 1)
        ps = stats.binom.pmf(ks, n, p)
        
        fig = px.bar(x=ks, y=ps, labels={'x': 'k Successes', 'y': 'Probability'}, title="Probability Mass Function")
        fig.update_traces(marker_color=['#6366f1' if x == k else '#c7d2fe' for x in ks])
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.session_state.report_data = {"Module": "Binomial", "n": n, "p": p, "k": k, "PMF": f"{pmf*100:.3f}%"}

# --- SYSTEM WIDE REPORT EXPORTER BUTTON ---
st.sidebar.markdown("### Output Export Panel")
report_string = "\n".join([f"**{key}**: {val}" for key, val in st.session_state.report_data.items()])
st.sidebar.download_button(
    label="📥 Export Report Summary",
    data=f"Sampling Statistics Explorer Summary Report\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n{report_string}",
    file_name=f"sampling_report_{module.lower().replace(' ', '_')}.txt",
    mime="text/plain"
)