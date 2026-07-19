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

# Custom styles matching the original Tailwind dashboard
st.markdown("""
    <style>
    .main { background-color: #F8FAFC; }
    div[data-testid="stSidebar"] { background-color: #F1F5F9; }
    h1, h2, h3, h4 { color: #0F172A; font-family: 'Helvetica Neue', Arial, sans-serif; }
    </style>
""", unsafe_allow_html=True)

# --- Enhanced UI Components (Guarantees Correct HTML Rendering) ---
def result_card(title, value, unit="", description="", variant="default"):
    styles = {
        "default": {"bg": "#EEF2FF", "border": "#C7D2FE", "text": "#3730A3"},
        "warning": {"bg": "#FFFBEB", "border": "#FDE68A", "text": "#92400E"},
        "danger": {"bg": "#FFF1F2", "border": "#FECDD3", "text": "#BE123C"},
        "success": {"bg": "#ECFDF5", "border": "#A7F3D0", "text": "#047857"}
    }
    s = styles.get(variant, styles["default"])
    desc_html = f'<p style="margin-top: 12px; color: #475569; font-size: 13px; line-height: 1.5;">{description}</p>' if description else ''
    
    html = (
        f'<div style="background-color: {s["bg"]}; border: 1px solid {s["border"]}; color: {s["text"]}; '
        f'padding: 20px; border-radius: 12px; margin-bottom: 15px;">'
        f'<h3 style="font-size: 11px; font-weight: bold; text-transform: uppercase; letter-spacing: 0.05em; '
        f'margin-bottom: 8px; opacity: 0.8; color: {s["text"]};">{title}</h3>'
        f'<div style="display: flex; align-items: baseline; gap: 6px;">'
        f'<span style="font-size: 28px; font-weight: 900;">{value}</span>'
        f'<span style="font-size: 14px; font-weight: 500; opacity: 0.8;">{unit}</span>'
        f'</div>{desc_html}</div>'
    )
    st.html(html)

def insight_box(title, content_html):
    html = (
        f'<div style="background-color: #FFFBEB; border-left: 4px solid #F59E0B; padding: 20px; '
        f'border-radius: 0 12px 12px 0; margin-top: 20px; margin-bottom: 20px;">'
        f'<div style="display: flex; align-items: center; gap: 8px; color: #78350F; font-weight: bold; margin-bottom: 12px;">'
        f'<span>ℹ️</span> <strong>{title}</strong>'
        f'</div>'
        f'<div style="color: #334155; font-size: 13.5px; line-height: 1.6;">{content_html}</div>'
        f'</div>'
    )
    st.html(html)

# Initialize global state context mapping
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

# --- APP ROUTING & MODULE EXECUTION ---

if module == "T-Test":
    st.title("T-Test Calculation")
    st.caption("Determine if group means differ significantly.")  #[cite: 1]
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        mode = st.radio("Test Type", ["One-Sample", "Two-Sample"])
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
            
    # Executing computational backend
    if mode == "One-Sample":
        df = max(n1 - 1, 1)
        t_stat = (m1 - m2) / (sd1 / np.sqrt(n1))
        hyp_label = "H₁: μ₁ ≠ μ₀" if tail == "Two-Tailed" else ("H₁: μ₁ > μ₀" if tailDir == "greater" else "H₁: μ₁ < μ₀")
    else:
        v1 = (sd1 ** 2) / n1
        v2 = (sd2 ** 2) / n2
        df = ((v1 + v2) ** 2) / ((v1 ** 2) / (n1 - 1) + (v2 ** 2) / (n2 - 1))
        t_stat = (m1 - m2) / np.sqrt(v1 + v2)
        hyp_label = "H₁: μ₁ ≠ μ₂" if tail == "Two-Tailed" else ("H₁: μ₁ > μ₂" if tailDir == "greater" else "H₁: μ₁ < μ₂")
        
    p_two = 2 * stats.t.sf(np.abs(t_stat), df)
    p_greater = stats.t.sf(t_stat, df)
    p_less = stats.t.cdf(t_stat, df)
    p_val = p_two if tail == "Two-Tailed" else (p_greater if tailDir == "greater" else p_less)
    significant = p_val < alpha
    
    with col2:
        # Plotly Render Configuration
        x_axis = np.linspace(-4, 4, 200)
        y_axis = stats.t.pdf(x_axis, df)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_axis, y=y_axis, mode='lines', name='t-dist', line=dict(color='#6366f1', width=2)))
        
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
            fig.add_trace(go.Scatter(x=[t_stat], y=[stats.t.pdf(t_stat, df)], mode='markers', marker=dict(color='#ef4444', size=8)))
            
        fig.update_layout(title="Distribution Visualization", height=230, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            result_card("T-Statistic", f"{t_stat:.3f}", description=hyp_label)  #[cite: 1]
        with sub_col2:
            result_card("P-Value", f"{p_val:.4f}", variant="success" if significant else "default", description="Two-tailed" if tail == "two" else f"One-tailed ({tailDir})")  #[cite: 1]
            
        result_card("H₀ Decision", "Reject H₀" if significant else "Fail to Reject H₀", variant="success" if significant else "warning", description="Difference is statistically significant." if significant else "Difference is likely due to chance.")  #[cite: 1]
        result_card("Degrees of Freedom", f"{df:.1f}")  #[cite: 1]
        
        insight_box("Understanding the T-Test", """
            <p>A T-test evaluates whether the difference between means is statistically significant or merely due to random chance.</p>
            <ul style="padding-left: 20px;">
                <li><strong>One-Sample:</strong> Compares your sample mean against a known target mean.</li>
                <li><strong>Two-Sample (Welch):</strong> Compares two independent groups without assuming equal variances.</li>
                <li><strong>Interpretation:</strong> A P-value smaller than Alpha (α) indicates that the probability of seeing this result by chance is low.</li>
            </ul>
            <p style="font-weight: bold; font-style: italic; color: #78350F; margin-top: 10px;">One-tailed vs Two-tailed:</p>
            <p>A <strong>two-tailed</strong> test asks "is there any difference?" — the rejection region is split equally across both tails. Use this when you have no prior expectation about direction, which is the safer default.</p>
            <p>A <strong>one-tailed</strong> test asks a directional question — "is μ₁ greater than μ₂?" or "is μ₁ less than μ₂?" — concentrating the entire rejection region in one tail. This gives more power to detect an effect in the predicted direction, but will miss an effect in the opposite direction entirely. Only choose a one-tailed test when the direction is justified by theory or prior evidence <em>before</em> seeing the data.</p>
        """)  #[cite: 1]

elif module == "Sample Size Determination":
    st.title("Sample Size Determination")
    st.caption("Minimum sample to detect a true effect with specified power and confidence.")  #[cite: 1]
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        conf = st.number_input("Confidence Level (1-α)", value=0.95, min_value=0.5, max_value=0.999)  #[cite: 1]
        power = st.number_input("Target Power (1-β)", value=0.80, min_value=0.5, max_value=0.999)  #[cite: 1]
        error = st.number_input("Detectable Difference (E)", value=0.05, min_value=0.001)  #[cite: 1]
        p = st.number_input("Estimated Rate (p)", value=0.50, min_value=0.001, max_value=0.999)  #[cite: 1]
        pop = st.number_input("Population (N)", value=0, min_value=0, step=1, help="Enter 0 for an infinite / very large population.")  #[cite: 1]
        
    z_alpha = np.abs(stats.norm.ppf((1 - conf) / 2))
    z_beta = np.abs(stats.norm.ppf(power))
    n0 = ((z_alpha + z_beta) ** 2 * p * (1 - p)) / (error ** 2)
    n_result = int(np.ceil(n0 / (1 + (n0 - 1) / pop))) if pop > 0 else int(np.ceil(n0))
        
    with col2:
        result_card("Required n", f"{n_result:,}", unit="Units", description="Minimum number of samples needed to detect the specified difference with the given confidence and power.")  #[cite: 1]
        insight_box("Strategic Design", """
            <p>This calculation considers both Type I error (Confidence) and Type II error (Power) to ensure your study is not "underpowered."</p>
            <p>If the population is at least 10 times larger than the sample size, you can use the infinite population formula.</p>
            <p>The detectable difference has a direct impact on the required sample size. A smaller detectable difference requires a much larger sample size.</p>
        """)  #[cite: 1]

elif module == "OC Curves":
    st.title("OC Curves")
    st.caption("Probability of lot acceptance versus defect rate.")  #[cite: 1]
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        n = st.number_input("Sample Size (n)", value=50, min_value=1, step=1)  #[cite: 1]
        c = st.number_input("Acceptance Number (c)", value=2, min_value=0, step=1)  #[cite: 1]
        maxDefect = st.number_input("LTPD / RQL", value=0.08, min_value=0.0, max_value=1.0)  #[cite: 1]
        
        beta_risk = stats.binom.cdf(c, n, maxDefect)
        result_card("Consumer's Risk (β)", f"{beta_risk*100:.2f}", unit="%", variant="danger", description=f"Probability of accepting a lot with {(maxDefect * 100):.1f}% defect rate (the LTPD).")  #[cite: 1]
        
    with col2:
        p_vals = np.linspace(0, 0.25, 100)
        pa_vals = stats.binom.cdf(c, n, p_vals)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=p_vals, y=pa_vals, mode='lines', name='Performance', line=dict(color='#6366f1', width=3)))
        fig.add_trace(go.Scatter(x=[maxDefect], y=[beta_risk], mode='markers', marker=dict(color='#f43f5e', size=10)))
        fig.update_layout(title="Acceptance Probability P(a) vs Defect Rate (p)", height=260, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        insight_box("Operating Characteristics", """
            <p>The OC curve shows how well a sampling plan (n, c) discriminates between good and bad lots.</p>
            <p>The <strong>Consumer's Risk (β)</strong> is the probability of accepting a lot that has a defect rate equal to the LTPD.</p>
        """)  #[cite: 1]

elif module == "Confidence Intervals":
    st.title("Confidence Intervals")
    st.caption("Range likely to contain the true population mean.")  #[cite: 1]
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        mean = st.number_input("Mean", value=100.0)  #[cite: 1]
        sd = st.number_input("Std Dev", value=15.0, min_value=0.001)  #[cite: 1]
        n = st.number_input("n", value=30, min_value=1, step=1)  #[cite: 1]
        conf = st.number_input("Confidence Level", value=0.95, min_value=0.01, max_value=0.999)  #[cite: 1]
        
    moe = np.abs(stats.norm.ppf((1 - conf) / 2)) * (sd / np.sqrt(n))
    
    with col2:
        sub1, sub2 = st.columns(2)
        with sub1:
            result_card("Lower bound", f"{mean - moe:.2f}", description=f"{(conf * 100):.0f}% confidence interval lower bound.")  #[cite: 1]
        with sub2:
            result_card("Upper bound", f"{mean + moe:.2f}", description=f"{(conf * 100):.0f}% confidence interval upper bound.")  #[cite: 1]
        result_card("Margin of Error", f"{moe:.3f}", description="Half-width of the confidence interval.")  #[cite: 1]
        
        insight_box("Interpreting a Confidence Interval", """
            <p>The margin of error decreases as your sample size (n) increases, reflecting greater precision in your estimate.</p>
            <p><strong>Correct interpretation:</strong> If you repeated this study many times and constructed a 95% CI each time, approximately 95% of those intervals would contain the true population mean. The confidence refers to the <em>procedure</em>, not to any single interval.</p>
            <p style="font-weight: bold; font-style: italic; color: #78350F; margin-top: 10px;">Common misconception:</p>
            <p>A 95% CI does <strong>not</strong> mean "there is a 95% probability that the true mean lies within this interval." The true population mean is a fixed (unknown) value — it either is or isn't inside this particular interval. Probability statements about the true mean belong to Bayesian credible intervals, not frequentist confidence intervals.</p>
        """)  #[cite: 1]

elif module == "Power Analysis":
    st.title("Power Analysis")
    st.caption("Probability of detecting a true difference.")  #[cite: 1]
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        mode = st.radio("Test Type", ["One-Sample", "Two-Sample"])  #[cite: 1]
        p1 = st.number_input("p1", value=0.50, min_value=0.0, max_value=1.0)  #[cite: 1]
        p2 = st.number_input("p2", value=0.55, min_value=0.0, max_value=1.0)  #[cite: 1]
        n = st.number_input("n", value=100, min_value=1, step=1)  #[cite: 1]
        alpha = st.number_input("Alpha", value=0.05, min_value=0.001, max_value=0.5)  #[cite: 1]
        
    h = np.abs(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))
    factor = np.sqrt(n / 2) if mode == "Two-Sample" else np.sqrt(n)
    current_power = stats.norm.cdf((h * factor) - np.abs(stats.norm.ppf(alpha / 2)))
    
    with col2:
        result_card("Statistical Power", f"{current_power:.3f}", variant="success" if current_power >= 0.8 else "warning", description="Adequately powered (>= 0.80)." if current_power >= 0.8 else "Underpowered — consider increasing sample size.")  #[cite: 1]
        result_card("Effect Size (Cohen h)", f"{h:.4f}", description="Arcsine-transformed difference between proportions.")  #[cite: 1]
        insight_box("Power and Effect Size", """
            <p>Statistical power is the probability that the test correctly rejects the null hypothesis when the alternative hypothesis is true.</p>
            <p>Power increases with sample size (n) and effect size.</p>
            <p style="font-weight: bold; font-style: italic; color: #78350F; margin-top: 10px;">Caution — the risks of underpowered studies:</p>
            <p>When power is too low (typically below 0.80), several serious problems arise. First, a non-significant result cannot be interpreted as evidence that no difference exists — it may simply mean the study was too small to detect it. Second, underpowered studies that do find a significant result tend to <strong>overestimate effect sizes</strong>, a phenomenon known as the "Winner's Curse," because only the largest random fluctuations cross the significance threshold. Third, findings from underpowered studies have low reproducibility. Always determine your required sample size <em>before</em> collecting data, not after.</p>
        """)  #[cite: 1]

elif module == "Binomial Confidence Int.":
    st.title("Binomial Confidence Interval")
    st.caption("Confidence interval for a proportion.")  #[cite: 1]
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        x = st.number_input("Successes (x)", value=15, min_value=0, step=1)  #[cite: 1]
        n = st.number_input("Trials (n)", value=100, min_value=1, step=1)  #[cite: 1]
        conf = st.number_input("Confidence Level", value=0.95, min_value=0.01, max_value=0.999)  #[cite: 1]
        
    p = x / n
    moe = np.abs(stats.norm.ppf((1 - conf) / 2)) * np.sqrt((p * (1 - p)) / n)
    
    with col2:
        result_card("Observed Proportion", f"{p*100:.1f}", unit="%", description="x / n")  #[cite: 1]
        sub1, sub2 = st.columns(2)
        with sub1:
            result_card("Lower Bound (Wald)", f"{max(0.0, p - moe)*100:.1f}", unit="%", description=f"{(conf * 100):.0f}% CI lower bound.")  #[cite: 1]
        with sub2:
            result_card("Upper Bound (Wald)", f"{min(1.0, p + moe)*100:.1f}", unit="%", description=f"{(conf * 100):.0f}% CI upper bound.")  #[cite: 1]
        result_card("Margin of Error", f"{moe*100:.2f}", unit="%")  #[cite: 1]
        insight_box("Proportion Estimation & Limitations", """
            <p>The <strong>Wald method</strong> used here is based on the normal approximation to the binomial distribution and has limitations. It often fails to maintain the target coverage probability (e.g., 95%) when successes are rare, <em>n</em> is small, or the proportion is near 0 or 1.</p>
            <p style="font-weight: bold; font-style: italic; color: #78350F; margin-top: 10px;">Recommendation:</p>
            <p>For higher accuracy, practitioners recommend the <strong>Wilson Score</strong> interval or the <strong>Agresti-Coull</strong> ("plus four") method, which perform much better across all sample sizes.</p>
        """)  #[cite: 1]

elif module == "Bayesian Binomial":
    st.title("Bayesian Credible Intervals")
    st.caption("Update Beta prior with observed data.")  #[cite: 1]
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        priorA = st.number_input("Prior α", value=0.5, min_value=0.001)  #[cite: 1]
        priorB = st.number_input("Prior β", value=0.5, min_value=0.001)  #[cite: 1]
        x = st.number_input("Observed Successes (x)", value=15, min_value=0, step=1)  #[cite: 1]
        n = st.number_input("Total Trials (n)", value=100, min_value=1, step=1)  #[cite: 1]
        
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
        fig.update_layout(title="Prior vs Posterior Distribution", height=240, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        result_card("Posterior alpha", f"{postA:.2f}", description=f"Prior alpha + successes: {priorA} + {x}")  #[cite: 1]
        result_card("Posterior beta", f"{postB:.2f}", description=f"Prior beta + failures: {priorB} + {n - x}")  #[cite: 1]
        
        sub1, sub2 = st.columns(2)
        with sub1:
            result_card("95% Credible Lower", f"{lo*100:.2f}", unit="%", description="Lower bound of the 95% credible interval.")  #[cite: 1]
        with sub2:
            result_card("95% Credible Upper", f"{hi*100:.2f}", unit="%", description="Upper bound of the 95% credible interval.")  #[cite: 1]
        result_card("Posterior Mean", f"{postMean*100:.2f}", unit="%", description=f"Best single-point estimate of the true proportion after updating the prior (alpha={priorA}, beta={priorB}) with {x} successes from {n} trials.")  #[cite: 1]
        
        insight_box("Bayesian Learning", """
            <p>Bayesian inference combines existing knowledge (the Prior) with new observed evidence to produce an updated belief — the Posterior distribution.</p>
            <p style="font-weight: bold; font-style: italic; color: #78350F; margin-top: 10px;">Variable explanations:</p>
            <ul style="padding-left: 20px;">
                <li><strong>Prior α:</strong> Encodes prior belief in successes. Interpreted as pseudo-counts of successes seen before this study. Setting α=1 alongside β=1 gives a flat, uninformative prior. Setting α=0.5 and β=0.5 gives a U-shaped prior. α and β can be any positive real numbers.</li>
                <li><strong>Prior β:</strong> Encodes prior belief in failures. Analogous to α but for failures. Larger values shift the prior toward lower proportions.</li>
                <li><strong>x (Successes):</strong> The number of positive outcomes observed in your new data. This updates the prior: the Posterior α becomes α + x.</li>
                <li><strong>n (Trials):</strong> Total number of trials in your new data. The Posterior β becomes β + (n − x). More trials pull the posterior sharply toward the observed rate.</li>
            </ul>
            <p>The posterior distribution is Beta(α + x, β + n − x). As n grows, the data overwhelms the prior and the posterior narrows around the true proportion.</p>
        """)  #[cite: 1]

elif module == "Bayes' Theorem":
    st.title("Bayes' Theorem")
    st.caption("Posterior probability calculation.")  #[cite: 1]
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        prior = st.number_input("Prior P(A)", value=0.01, min_value=0.0, max_value=1.0)  #[cite: 1]
        sens = st.number_input("Sensitivity P(B|A)", value=0.95, min_value=0.0, max_value=1.0)  #[cite: 1]
        fpr = st.number_input("False Positive Rate P(B|¬A)", value=0.05, min_value=0.0, max_value=1.0)  #[cite: 1]
        
    post = (sens * prior) / ((sens * prior) + (fpr * (1 - prior)))
    
    with col2:
        result_card("Posterior P(A|B)", f"{post*100:.2f}", unit="%", description="The probability that the condition is truly present given a positive test result.")  #[cite: 1]
        insight_box("The Base Rate Fallacy", """
            <p>Bayes' Theorem shows how even a very accurate test can have a low posterior probability if the prior probability (base rate) is extremely low.</p>
            <p style="font-weight: bold; font-style: italic; color: #78350F; margin-top: 10px;">Variable explanations:</p>
            <ul style="padding-left: 20px;">
                <li><strong>Prior P(A):</strong> Your baseline belief that the condition is present before any test is run — e.g., the prevalence of a disease in the population. The complement is P(¬A) = 1 - P(A).</li>
                <li><strong>Sensitivity P(B|A):</strong> The probability that the test returns a positive result given that the condition is truly present. A value of 0.95 means the test correctly identifies 95% of true cases.</li>
                <li><strong>False Positive Rate P(B|¬A):</strong> The probability that the test returns a positive result when the condition is actually absent. A value of 0.05 means 5% of healthy subjects will incorrectly test positive.</li>
            </ul>
        """)  #[cite: 1]

elif module == "Tolerance Intervals":
    st.title("Tolerance Intervals")
    st.caption("Bounds for population proportion.")  #[cite: 1]
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        mean = st.number_input("Mean", value=10.0)  #[cite: 1]
        sd = st.number_input("Std Dev", value=0.5, min_value=0.001)  #[cite: 1]
        confLevel = st.number_input("Confidence Level (1-α)", value=0.95, min_value=0.01, max_value=0.999)  #[cite: 1]
        popProp = st.number_input("Population Proportion (p)", value=0.99, min_value=0.01, max_value=0.999)  #[cite: 1]
        n = st.number_input("Sample Size (n)", value=50, min_value=2, step=1)  #[cite: 1]
        
    zp = np.abs(stats.norm.ppf((1 + popProp) / 2))
    zg = np.abs(stats.norm.ppf(1 - confLevel))
    k = zp * np.sqrt(1 + 1 / n) * (1 + (zg ** 2) / (2 * n - 2))
    
    with col2:
        result_card("Tolerance Factor k", f"{k:.4f}", description=f"k-factor for a {(popProp*100):.0f}%/{(confLevel*100):.0f}% two-sided tolerance interval (n = {n}).")  #[cite: 1]
        sub1, sub2 = st.columns(2)
        with sub1:
            result_card("Lower Bound", f"{mean - k * sd:.3f}", description=f"We are {(confLevel*100):.0f}% confident that at least {(popProp*100):.0f}% of all individual units fall above this value.")  #[cite: 1]
        with sub2:
            result_card("Upper Bound", f"{mean + k * sd:.3f}", description=f"We are {(confLevel*100):.0f}% confident that at least {(popProp*100):.0f}% of all individual units fall below this value.")  #[cite: 1]
        
        insight_box("Tolerance vs Confidence", f"""
            <p>A Confidence Interval covers a population <em>parameter</em> (e.g., the mean). A Tolerance Interval covers a fixed <em>proportion of individual measurements</em> from the population.</p>
            <p><strong>How to interpret:</strong> A {popProp*100:.0f}%/{confLevel*100:.0f}% tolerance interval means: "We are {confLevel*100:.0f}% confident that at least {popProp*100:.0f}% of all individual units from this process fall within these bounds." It answers the question <em>"Where do most individual values lie?"</em> rather than <em>"Where is the average?"</em></p>
            <p>This makes tolerance intervals especially useful in manufacturing and quality control, where you need to verify that the bulk of production output meets specification limits — not just that the mean does.</p>
        """)  #[cite: 1]

elif module == "ANOVA / F-Test":
    st.title("One-Way ANOVA")
    st.caption("Compare means across three or more independent groups.")  #[cite: 1]
    st.markdown("---")
    
    if "anova_df" not in st.session_state:
        st.session_state.anova_df = pd.DataFrame([
            {"Group": "Group 1", "Mean": 12.5, "Std Dev": 2.1, "n": 20},
            {"Group": "Group 2", "Mean": 14.8, "Std Dev": 2.4, "n": 20},
            {"Group": "Group 3", "Mean": 13.2, "Std Dev": 1.9, "n": 20}
        ])
        
    edited_df = st.data_editor(st.session_state.anova_df, num_rows="dynamic", use_container_width=True)  #[cite: 1]
    st.session_state.anova_df = edited_df
    alpha = st.number_input("Alpha (α)", value=0.05, min_value=0.001, max_value=0.5)  #[cite: 1]
    
    if len(edited_df) >= 2:
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
            st.table(pd.DataFrame({
                "Source": ["Between", "Within"],
                "SS": [f"{ssBetween:.3f}", f"{ssWithin:.3f}"],
                "df": [dfBetween, dfWithin],
                "MS": [f"{msBetween:.3f}", f"{msWithin:.3f}"],
                "F": [f"{fStat:.3f}", "—"]
            }))  #[cite: 1]
            
        with col2:
            sub1, sub2 = st.columns(2)
            with sub1:
                result_card("F-Statistic", f"{fStat:.3f}", description=f"df₁ = {dfBetween}, df₂ = {dfWithin}")  #[cite: 1]
            with sub2:
                result_card("P-Value", f"{pVal:.4f}", variant="success" if significant else "default", description=f"Significant at α = {alpha}" if significant else f"Not significant at α = {alpha}")  #[cite: 1]
            result_card("H₀ Decision", "Reject" if significant else "Fail to Reject", variant="success" if significant else "warning", description="At least one group mean differs significantly from the others." if significant else "No significant difference detected between group means.")  #[cite: 1]
            result_card("Effect Size η²", f"{etaSq:.3f}", description=f"{(etaSq * 100):.1f}% of total variance is explained by group membership. Small ≈ 0.01, Medium ≈ 0.06, Large ≈ 0.14.")  #[cite: 1]
            
            insight_box("Understanding One-Way ANOVA", """
                <p>ANOVA tests whether the variance <em>between</em> group means is larger than what would be expected from the variance <em>within</em> groups by chance alone. The F-statistic is the ratio of these two variance estimates.</p>
                <p><strong>Assumptions:</strong> Groups are independent, observations within each group are approximately normally distributed, and group variances are roughly equal (homoscedasticity). ANOVA is robust to moderate violations of normality with larger samples.</p>
                <p><strong>Important limitation:</strong> A significant F-test only tells you that <em>at least one</em> group differs — it does not tell you <em>which</em> groups differ. Follow up with post-hoc tests (Tukey, Bonferroni) to identify specific pairwise differences.</p>
                <p><strong>η² (Eta-squared)</strong> is the effect size: the proportion of total variance explained by the grouping factor. It complements the p-value by describing the <em>magnitude</em> of the effect, not just its statistical significance.</p>
            """)  #[cite: 1]

elif module == "Chi-Square":
    st.title("Chi-Square Test")
    st.caption("Test for goodness of fit or independence between categorical variables.")  #[cite: 1]
    st.markdown("---")
    
    mode = st.radio("Test Type", ["Goodness of Fit", "Independence"])  #[cite: 1]
    alpha = st.number_input("Alpha (α)", value=0.05, min_value=0.001, max_value=0.5)  #[cite: 1]
    
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
                result_card("χ² Statistic", f"{chi2:.3f}", description=f"Degrees of freedom: {df}")  #[cite: 1]
            with col2:
                result_card("P-Value", f"{p_val:.4f}", variant="success" if significant else "default", description=f"Significant at α = {alpha}" if significant else f"Not significant at α = {alpha}")  #[cite: 1]
                result_card("H₀ Decision", "Reject" if significant else "Fail to Reject", variant="success" if significant else "warning", description="The observed distribution differs significantly from the expected distribution." if significant else "No significant departure from the expected distribution.")  #[cite: 1]

    else:
        if "indep_df" not in st.session_state:
            st.session_state.indep_df = pd.DataFrame([[40, 20], [15, 35]], columns=["Col A", "Col B"], index=["Row A", "Row B"])
        indep_ed = st.data_editor(st.session_state.indep_df, num_rows="dynamic", use_container_width=True)
        st.session_state.indep_df = indep_ed
        
        matrix = indep_ed.astype(float).values
        if matrix.shape[0] >= 2 and matrix.shape[1] >= 2:
            chi2, p_val, df, expected_matrix = stats.chi2_contingency(matrix)
            n_total = np.sum(matrix)
            min_dim = min(matrix.shape) - 1
            cramers_v = np.sqrt(chi2 / (n_total * min_dim)) if n_total > 0 and min_dim > 0 else 0
            significant = p_val < alpha
            
            col1, col2 = st.columns(2)
            with col1:
                result_card("χ² Statistic", f"{chi2:.3f}", description=f"Degrees of freedom: {df}")  #[cite: 1]
                result_card("Cramér's V", f"{cramers_v:.3f}", description="Effect size: 0–0.1 negligible, 0.1–0.3 small, 0.3–0.5 moderate, 0.5+ strong.")  #[cite: 1]
            with col2:
                result_card("P-Value", f"{p_val:.4f}", variant="success" if significant else "default", description=f"Significant at α = {alpha}" if significant else f"Not significant at α = {alpha}")  #[cite: 1]
                result_card("H₀ Decision", "Reject" if significant else "Fail to Reject", variant="success" if significant else "warning", description="The two variables are not independent — a significant association exists." if significant else "No significant association detected between the two variables.")  #[cite: 1]

    insight_box("Understanding Chi-Square Tests", """
        <p>The chi-square test compares <em>observed</em> counts to <em>expected</em> counts. A large χ² statistic means the observed data deviate substantially from what the null hypothesis predicts.</p>
        <p><strong>Goodness of Fit</strong> tests whether a single categorical variable follows a specified theoretical distribution (e.g., are defects equally spread across shift times?).</p>
        <p><strong>Test of Independence</strong> tests whether two categorical variables are associated with each other (e.g., is defect type related to production line?). Expected counts are computed from the marginal totals.</p>
        <p><strong>Key assumption:</strong> Expected counts in each cell should be ≥ 5. If this is violated, consider Fisher's Exact Test or combining sparse categories.</p>
        <p><strong>Cramér's V</strong> measures association strength independently of sample size — a large sample can produce a significant p-value from a negligibly small true effect.</p>
    """)  #[cite: 1]

elif module == "Poisson":
    st.title("Poisson Distribution")
    st.caption("Probability of event counts over a fixed interval.")  #[cite: 1]
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        lam = st.number_input("Rate λ (lambda)", value=4.0, min_value=0.001)  #[cite: 1]
        k = st.number_input("k (number of events)", value=3, min_value=0, step=1)  #[cite: 1]
        
        pmf = stats.poisson.pmf(k, lam)
        cdfLe = stats.poisson.cdf(k, lam)
        cdfLt = stats.poisson.cdf(k - 1, lam)
        cdfGe = 1 - cdfLt
        
        result_card("P(X = k)", f"{pmf*100:.3f}", unit="%", description=f"Exact probability of exactly {int(k)} events.")  #[cite: 1]
        result_card("P(X ≤ k)", f"{cdfLe*100:.3f}", unit="%", description=f"Probability of {int(k)} or fewer events.")  #[cite: 1]
        result_card("P(X < k)", f"{cdfLt*100:.3f}", unit="%", description=f"Probability of fewer than {int(k)} events.")  #[cite: 1]
        result_card("P(X ≥ k)", f"{cdfGe*100:.3f}", unit="%", description=f"Probability of {int(k)} or more events.")  #[cite: 1]
        
    with col2:
        max_k = max(20, int(np.ceil(lam * 3)))
        ks = np.arange(0, max_k + 1)
        ps = stats.poisson.pmf(ks, lam)
        
        fig = px.bar(x=ks, y=ps, labels={'x': 'k Events', 'y': 'Probability Mass'}, title="Probability Mass Function")
        fig.update_traces(marker_color=['#6366f1' if x == k else '#c7d2fe' for x in ks])
        fig.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        insight_box("Understanding the Poisson Distribution", """
            <p>The Poisson distribution models the number of times a rare event occurs within a fixed interval of time, space, or volume — given a known average rate λ and assuming events occur independently of one another.</p>
            <p><strong>Typical applications in sampling and quality:</strong> modelling defect counts per unit, non-conformances per batch, customer arrivals per hour, or errors per 1,000 lines of code.</p>
            <p><strong>Key properties:</strong> Both the mean and variance equal λ. When λ is large (≥ ~10), the Poisson approximates a normal distribution with mean λ and standard deviation √λ.</p>
            <p><strong>Relationship to binomial:</strong> When n is large and p is small (rare events), the Binomial(n, p) is well-approximated by Poisson(λ = np). This is useful when individual trial probabilities are hard to estimate but the average count is known.</p>
        """)  #[cite: 1]

elif module == "Normal":
    st.title("Normal Distribution")
    st.caption("Probabilities under the Gaussian bell curve.")  #[cite: 1]
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        mu = st.number_input("Mean (μ)", value=0.0)  #[cite: 1]
        sigma = st.number_input("Std Dev (σ)", value=1.0, min_value=0.001)  #[cite: 1]
        x = st.number_input("x (value of interest)", value=1.0)  #[cite: 1]
        
        z = (x - mu) / sigma
        ple = stats.norm.cdf(z)
        pge = 1 - ple
        pdf_val = stats.norm.pdf(x, mu, sigma)
        
        result_card("P(X ≤ x)", f"{ple*100:.3f}", unit="%", description=f"Probability of a value at or below {x}.")  #[cite: 1]
        result_card("P(X > x)", f"{pge*100:.3f}", unit="%", description=f"Probability of a value above {x}.")  #[cite: 1]
        result_card("z-score", f"{z:.3f}", description=f"x is {abs(z):.2f} standard deviations { 'above' if x >= mu else 'below' } the mean.")  #[cite: 1]
        result_card("PDF f(x)", f"{pdf_val:.4f}", description="Probability density at exactly x (not a probability itself).")  #[cite: 1]
        
    with col2:
        x_vals = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
        y_vals = stats.norm.pdf(x_vals, mu, sigma)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color='#6366f1', width=2.5)))
        shade_x = x_vals[x_vals <= x]
        shade_y = y_vals[x_vals <= x]
        fig.add_trace(go.Scatter(x=np.concatenate(([shade_x[0]], shade_x, [shade_x[-1]])), y=np.concatenate(([0], shade_y, [0])), fill='toself', fillcolor='rgba(99, 102, 241, 0.2)', line=dict(color='rgba(255,255,255,0)')))
        fig.update_layout(title="Probability Density Function Area Shading", height=240, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        insight_box("Understanding the Normal Distribution", """
            <p>The normal (Gaussian) distribution is the most widely used continuous distribution in statistics. It is fully described by its mean μ and standard deviation σ, and is symmetric around μ.</p>
            <p><strong>The 68–95–99.7 rule:</strong> approximately 68% of values fall within ±1σ of the mean, 95% within ±2σ, and 99.7% within ±3σ.</p>
            <p><strong>Sampling relevance:</strong> By the Central Limit Theorem, the mean of a large sample from almost any distribution is approximately normally distributed — the foundation of confidence intervals and t-tests.</p>
            <p><strong>z-score:</strong> The standardised value z = (x − μ) / σ measures how many standard deviations x lies from the mean, enabling comparison across different scales.</p>
        """)  #[cite: 1]

elif module == "Binomial":
    st.title("Binomial Distribution")
    st.caption("Probability of k successes in n independent trials.")  #[cite: 1]
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        n = st.number_input("Trials (n)", value=20, min_value=1, max_value=100, step=1)  #[cite: 1]
        p = st.number_input("Success probability (p)", value=0.30, min_value=0.0, max_value=1.0)  #[cite: 1]
        k = st.number_input("k (successes of interest)", value=5, min_value=0, max_value=int(n), step=1)  #[cite: 1]
        
        pmf = stats.binom.pmf(k, n, p)
        cdfLe = stats.binom.cdf(k, n, p)
        cdfLt = stats.binom.cdf(k - 1, n, p) if k > 0 else 0
        cdfGe = 1 - cdfLt
        mean = n * p
        sd = np.sqrt(n * p * (1 - p))
        
        result_card("P(X = k)", f"{pmf*100:.3f}", unit="%", description=f"Exact probability of exactly {int(k)} successes.")  #[cite: 1]
        result_card("P(X ≤ k)", f"{cdfLe*100:.3f}", unit="%", description=f"Probability of {int(k)} or fewer successes.")  #[cite: 1]
        result_card("P(X ≥ k)", f"{cdfGe*100:.3f}", unit="%", description=f"Probability of {int(k)} or more successes.")  #[cite: 1]
        result_card("Mean / Std Dev", f"{mean:.2f} / {sd:.2f}", description=f"Expected successes np = {mean:.2f}. Spread √(np(1−p)) = {sd:.2f}.")  #[cite: 1]
        
    with col2:
        ks = np.arange(0, n + 1)
        ps = stats.binom.pmf(ks, n, p)
        
        fig = px.bar(x=ks, y=ps, labels={'x': 'k Successes', 'y': 'Probability'}, title="Probability Mass Function")
        fig.update_traces(marker_color=['#6366f1' if x == k else '#c7d2fe' for x in ks])
        fig.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        insight_box("Understanding the Binomial Distribution", """
            <p>The binomial distribution gives the probability of obtaining exactly k successes in n independent trials, where each trial has the same probability p of success.</p>
            <p><strong>Typical applications:</strong> pass/fail inspection (k defectives in n sampled items), survey responses, or attribute sampling in quality control.</p>
            <p><strong>Key properties:</strong> Mean = np, Variance = np(1−p). The distribution is symmetric when p = 0.5, right-skewed when p is small, and left-skewed when p is large.</p>
            <p><strong>Normal approximation:</strong> When np ≥ 5 and n(1−p) ≥ 5, the binomial is well-approximated by a normal distribution with mean np and std dev √(np(1−p)) — the basis of the Wald proportion CI.</p>
        """)  #[cite: 1]

# --- SYSTEM WIDE REPORT EXPORTER PANEL ---
st.sidebar.markdown("### Output Export Panel")
report_string = "\n".join([f"**{key}**: {val}" for key, val in st.session_state.report_data.items()])
st.sidebar.download_button(
    label="📥 Export Report Summary",
    data=f"Sampling Statistics Explorer Summary Report\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n{report_string}",
    file_name=f"sampling_report_{module.lower().replace(' ', '_')}.txt",
    mime="text/plain"
)