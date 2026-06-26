import streamlit as st
import pandas as pd
import tempfile
import time
from datetime import datetime
import markdown

from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.eda import EDA
from src.model_trainer import ModelTrainer
from src.model_selector import ModelSelector
from src.insight_generator import InsightGenerator
from src.visualizer import Visualizer
from src.llm_report_generator import LLMReportGenerator

import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Smart Data Analyst Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# ADVANCED CSS WITH DARK THEME
# ==================================================

st.markdown("""
<style>
    /* Global Styles - Dark Theme */
    .main {
        padding: 0rem 1rem;
        background-color: #0e1117;
    }
    
    /* Animated Gradient Header */
    .gradient-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        animation: fadeIn 0.8s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Cards - Dark Theme */
    .metric-card {
        background: #1e1e2e;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        color: #ffffff;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.5);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ffffff;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #a0a0b0;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Section Headers */
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Button Styles */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Expander Styling - Dark */
    .streamlit-expanderHeader {
        background: #1e1e2e;
        border-radius: 10px;
        font-weight: 600;
        color: #ffffff;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: #2a2a3e;
    }
    
    /* AI Report Container - ChatGPT/DeepSeek Style */
    .ai-report-container {
        background: #1e1e2e;
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #2a2a3e;
        box-shadow: 0 5px 30px rgba(0,0,0,0.5);
        margin: 1.5rem 0;
        animation: slideIn 0.5s ease-out;
        color: #e4e4e7;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .ai-report-container h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin-top: 0;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #2a2a3e;
        padding-bottom: 0.8rem;
    }
    
    .ai-report-container h2 {
        color: #a78bfa;
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .ai-report-container h3 {
        color: #8b9cf7;
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    
    .ai-report-container h4 {
        color: #c4b5fd;
        font-size: 1.1rem;
        font-weight: 500;
        margin-top: 1.2rem;
        margin-bottom: 0.5rem;
    }
    
    .ai-report-container p {
        color: #d4d4d8;
        line-height: 1.8;
        margin-bottom: 1rem;
    }
    
    .ai-report-container ul, .ai-report-container ol {
        padding-left: 1.5rem;
        color: #d4d4d8;
    }
    
    .ai-report-container li {
        margin: 0.5rem 0;
        color: #d4d4d8;
        line-height: 1.6;
    }
    
    .ai-report-container li::marker {
        color: #a78bfa;
    }
    
    .ai-report-container strong {
        color: #a78bfa;
        font-weight: 600;
    }
    
    .ai-report-container em {
        color: #8b9cf7;
    }
    
    .ai-report-container code {
        background: #2a2a3e;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        color: #c4b5fd;
        font-size: 0.9rem;
    }
    
    .ai-report-container blockquote {
        border-left: 4px solid #667eea;
        padding-left: 1rem;
        margin: 1rem 0;
        color: #a0a0b0;
        font-style: italic;
    }
    
    .ai-report-container hr {
        border: none;
        border-top: 2px solid #2a2a3e;
        margin: 2rem 0;
    }
    
    /* Number badges in report */
    .report-badge {
        display: inline-block;
        background: #2a2a3e;
        color: #a78bfa;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    
    /* Key metrics in report */
    .report-metric {
        display: inline-block;
        background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%);
        color: #c4b5fd;
        padding: 0.3rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid #2a2a3e;
        margin: 0.2rem 0;
    }
    
    /* Status Badges - Dark */
    .badge-success {
        background: #2a2a3e;
        color: #a78bfa;
        padding: 0.25rem 0.75rem;
        border-radius: 50px;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    .badge-info {
        background: #2a2a3e;
        color: #8b9cf7;
        padding: 0.25rem 0.75rem;
        border-radius: 50px;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    /* Tab Styling - Dark */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #1e1e2e;
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #2a2a3e;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        color: #a0a0b0;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #3a3a4e;
        color: #ffffff;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    /* Progress Bar - Dark */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar - Dark */
    .sidebar .sidebar-content {
        background: #0e1117;
    }
    
    /* Dataframe Styling - Dark */
    .dataframe {
        background: #1e1e2e !important;
        color: #d4d4d8 !important;
    }
    
    .dataframe thead tr th {
        background: #2a2a3e !important;
        color: #a78bfa !important;
    }
    
    .dataframe tbody tr:hover {
        background: #2a2a3e !important;
    }
    
    /* Custom Scrollbar - Dark */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e1e2e;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    
    /* Markdown text color - Dark */
    .stMarkdown {
        color: #d4d4d8;
    }
    
    /* Metric containers */
    [data-testid="metric-container"] {
        background: #1e1e2e;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #2a2a3e;
    }
    
    [data-testid="metric-container"] label {
        color: #a0a0b0 !important;
    }
    
    [data-testid="metric-container"] .value {
        color: #ffffff !important;
    }
    
    /* Warning/Info boxes - Dark */
    .stAlert {
        background: #1e1e2e !important;
        border: 1px solid #2a2a3e !important;
    }
    
    /* Download button in dark theme */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Success message */
    .stSuccess {
        background: #1e1e2e !important;
        border-left: 4px solid #a78bfa !important;
        color: #d4d4d8 !important;
    }
    
    /* JSON viewer - Dark */
    .stJson {
        background: #1e1e2e !important;
        border: 1px solid #2a2a3e !important;
        color: #d4d4d8 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================================================
# SESSION STATE INITIALIZATION
# ==================================================

if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False
if 'start_time' not in st.session_state:
    st.session_state.start_time = None

# ==================================================
# HEADER
# ==================================================

st.markdown("""
<div class="gradient-header">
    <h1 style="color: white; font-size: 3rem; margin: 0;">🚀 Smart Data Analyst Pro</h1>
    <p style="color: rgba(255,255,255,0.9); font-size: 1.2rem; margin-top: 0.5rem;">
        AI-Powered AutoML & Business Intelligence Platform
    </p>
    <div style="display: flex; gap: 1rem; margin-top: 1rem;">
        <span class="badge-success">✓ AutoML</span>
        <span class="badge-info">🤖 AI Insights</span>
        <span class="badge-success">📊 Visual Analytics</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "📂 Upload CSV Dataset",
        type=["csv"],
        help="Upload a CSV file to begin analysis"
    )
    
    if uploaded_file is None:
        st.info("👈 Upload a CSV file to get started")
        st.stop()
    
    st.markdown("---")
    
    # Show file info
    st.markdown("### 📄 File Information")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Size", f"{uploaded_file.size / 1024:.1f} KB")
    with col2:
        st.metric("Type", "CSV")
    
    st.markdown("---")
    
    # Load data
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    temp_file.write(uploaded_file.getvalue())
    temp_file.close()
    
    loader = DataLoader(temp_file.name)
    df = loader.load_data()
    
    # Target selection
    st.markdown("### 🎯 Target Selection")
    target_column = st.selectbox(
        "Select Target Column",
        df.columns,
        help="Choose the column you want to predict"
    )
    
    st.markdown("---")
    
    # Analysis button
    analyze = st.button(
        "🚀 Start Analysis",
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Sidebar footer
    st.markdown("""
    <div style="text-align: center; color: #a0a0b0; font-size: 0.8rem; margin-top: 2rem;">
        <p>⚡ Powered by AI</p>
        <p>📊 Version 2.0</p>
    </div>
    """, unsafe_allow_html=True)

# ==================================================
# WAIT FOR USER
# ==================================================

if not analyze:
    st.markdown("### 👋 Welcome to Smart Data Analyst Pro")
    st.markdown("""
    <div style="background: #1e1e2e; padding: 2rem; border-radius: 15px; margin: 1rem 0; border: 1px solid #2a2a3e;">
        <h4 style="color: #a78bfa;">🚀 Get Started</h4>
        <p style="color: #a0a0b0;">Upload your dataset and click <strong style="color: #a78bfa;">Start Analysis</strong> to begin the AI-powered analysis pipeline.</p>
        <ul style="color: #a0a0b0;">
            <li>📊 Automatic data cleaning & preprocessing</li>
            <li>🤖 Multiple ML models training & selection</li>
            <li>📈 Interactive visualizations</li>
            <li>🧠 AI-generated insights report</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Show basic stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    st.stop()

# ==================================================
# RUN PIPELINE
# ==================================================

st.session_state.start_time = time.time()

with st.spinner("🧠 Running AI Pipeline..."):
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Data Cleaning
    status_text.text("Step 1/5: Cleaning data...")
    progress_bar.progress(20)
    
    cleaner = DataCleaner()
    clean_df = cleaner.clean_data(df, target_column=target_column)
    
    # Step 2: EDA
    status_text.text("Step 2/5: Performing exploratory data analysis...")
    progress_bar.progress(40)
    eda = EDA(clean_df)
    
    report = {
        "dataset_summary": {
            "total_rows": clean_df.shape[0],
            "total_columns": clean_df.shape[1],
            "missing_values": int(clean_df.isnull().sum().sum()),
            "duplicates": int(clean_df.duplicated().sum())
        },
        "unique_values": {col: clean_df[col].nunique() for col in clean_df.columns},
        "data_types": clean_df.dtypes.to_frame(name="Datatype")
    }
    
    # Step 3: Model Training
    status_text.text("Step 3/5: Training ML models...")
    progress_bar.progress(60)
    trainer = ModelTrainer(clean_df, target_column=target_column)
    results = trainer.train()
    
    # Step 4: Model Selection
    status_text.text("Step 4/5: Selecting best model...")
    progress_bar.progress(80)
    selector = ModelSelector(results)
    best_result = selector.select_best_model()
    best_model = best_result["best_model"]
    
    feature_names = clean_df.drop(columns=[target_column]).columns
    insight = InsightGenerator(best_model, feature_names)
    top_features = insight.get_top_features()
    
    # Step 5: AI Report
    status_text.text("Step 5/5: Generating AI insights...")
    progress_bar.progress(90)
    
    ai_report = None
    ai_report_error = None
    
    if api_key:
        try:
            llm = LLMReportGenerator(api_key=api_key)
            ai_report = llm.generate_report(
                dataset_summary=report["dataset_summary"],
                best_model=best_result["best_model_name"],
                best_score=best_result["best_score"],
                top_features=top_features
            )
        except Exception as e:
            ai_report_error = str(e)
    else:
        ai_report_error = "MISTRAL_API_KEY not found in environment variables"
    
    progress_bar.progress(100)
    status_text.text("✅ Analysis Complete!")
    time.sleep(0.5)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

st.session_state.analysis_complete = True
st.session_state.report_generated = True

# ==================================================
# RESULTS DASHBOARD
# ==================================================

# Success message with timing
elapsed_time = time.time() - st.session_state.start_time
st.balloons()
st.success(f"✅ Analysis Completed Successfully in {elapsed_time:.1f} seconds!")

# ==================================================
# METRICS DASHBOARD
# ==================================================

st.markdown('<h2 class="section-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📊 Total Rows</div>
        <div class="metric-value">{clean_df.shape[0]:,}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📋 Total Columns</div>
        <div class="metric-value">{clean_df.shape[1]}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #e74c3c;">
        <div class="metric-label">⚠️ Missing Values</div>
        <div class="metric-value">{int(clean_df.isnull().sum().sum()):,}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #f39c12;">
        <div class="metric-label">🔄 Duplicates</div>
        <div class="metric-value">{int(clean_df.duplicated().sum()):,}</div>
    </div>
    """, unsafe_allow_html=True)

# ==================================================
# DATA PREVIEW WITH TABS
# ==================================================

st.markdown('<h2 class="section-header">📋 Data Explorer</h2>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Preview", "📑 Column Info", "📈 Statistics"])

with tab1:
    st.dataframe(clean_df.head(20), use_container_width=True, height=400)
    st.caption(f"Showing first 20 rows out of {clean_df.shape[0]:,} total rows")

with tab2:
    col_info = pd.DataFrame({
        "Column": clean_df.columns,
        "Datatype": clean_df.dtypes.astype(str),
        "Unique Values": [clean_df[col].nunique() for col in clean_df.columns],
        "Null Count": [clean_df[col].isnull().sum() for col in clean_df.columns],
        "Null %": [(clean_df[col].isnull().sum() / len(clean_df) * 100).round(2) for col in clean_df.columns]
    })
    st.dataframe(col_info, use_container_width=True, height=400)

with tab3:
    st.dataframe(clean_df.describe(), use_container_width=True, height=400)

# ==================================================
# TARGET COLUMN
# ==================================================

st.markdown('<h2 class="section-header">🎯 Target Variable</h2>', unsafe_allow_html=True)

st.markdown(f"""
<div style="background: #1e1e2e; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #667eea; border: 1px solid #2a2a3e;">
    <h3 style="color: #a78bfa; margin: 0;">{target_column}</h3>
    <p style="color: #a0a0b0; margin-top: 0.5rem;">
        Target column for prediction • {clean_df[target_column].dtype} • {clean_df[target_column].nunique()} unique values
    </p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# EDA REPORT
# ==================================================

st.markdown('<h2 class="section-header">📈 Exploratory Data Analysis</h2>', unsafe_allow_html=True)

eda_tab1, eda_tab2, eda_tab3 = st.tabs(
    [
        "📊 Summary Statistics",
        "🔢 Unique Values",
        "📋 Data Types"
    ]
)

with eda_tab1:
    st.json(report["dataset_summary"])

with eda_tab2:
    st.json(report["unique_values"])

with eda_tab3:
    st.dataframe(report["data_types"], use_container_width=True)

# ==================================================
# MACHINE LEARNING RESULTS
# ==================================================

st.markdown('<h2 class="section-header">🤖 Model Performance</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #27ae60;">
        <div class="metric-label">🏆 Best Model</div>
        <div class="metric-value" style="font-size: 1.8rem;">{best_result['best_model_name']}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #3498db;">
        <div class="metric-label">📈 Model Score</div>
        <div class="metric-value" style="font-size: 1.8rem;">{best_result['best_score']:.4f}</div>
        <div style="margin-top: 0.5rem;">
            <div style="background: #2a2a3e; height: 8px; border-radius: 4px; overflow: hidden;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); width: {best_result['best_score']*100:.1f}%; height: 100%;"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div style="background: #1e1e2e; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; border: 1px solid #2a2a3e;">
    <h4 style="color: #a78bfa;">⭐ Top Important Features</h4>
    <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem;">
        {''.join([f'<span style="background: #2a2a3e; color: #a78bfa; padding: 0.3rem 1rem; border-radius: 20px; font-size: 0.9rem; border: 1px solid #3a3a4e;">{feature}</span>' for feature in top_features[:8]])}
    </div>
</div>
""", unsafe_allow_html=True)

# ==================================================
# VISUALIZATIONS
# ==================================================

st.markdown('<h2 class="section-header">📊 Interactive Visualizations</h2>', unsafe_allow_html=True)

visualizer = Visualizer(clean_df)

viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    with st.expander("🔍 Missing Values Analysis", expanded=False):
        fig = visualizer.plot_missing_values()
        st.pyplot(fig, use_container_width=True)

with viz_col2:
    with st.expander("🔥 Correlation Heatmap", expanded=False):
        fig = visualizer.plot_correlation_heatmap()
        st.pyplot(fig, use_container_width=True)

with st.expander("📈 Distribution Histograms", expanded=False):
    st.caption("Distribution of numerical features")
    histograms = visualizer.plot_histograms()
    cols = st.columns(3)
    for idx, fig in enumerate(histograms):
        with cols[idx % 3]:
            st.pyplot(fig, use_container_width=True)

with st.expander("📦 Boxplots - Outlier Detection", expanded=False):
    st.caption("Outlier detection for numerical features")
    boxplots = visualizer.plot_boxplots()
    cols = st.columns(3)
    for idx, fig in enumerate(boxplots):
        with cols[idx % 3]:
            st.pyplot(fig, use_container_width=True)

# ==================================================
# AI REPORT - BEAUTIFULLY FORMATTED
# ==================================================

st.markdown('<h2 class="section-header">🧠 AI-Powered Insights</h2>', unsafe_allow_html=True)

if ai_report:
    # Clean and format the report for better display
    # Remove any markdown code blocks if present
    clean_report = ai_report.replace('```', '').replace('markdown', '')
    
    # Display the AI report in a beautiful container
    st.markdown(f"""
    <div class="ai-report-container">
        {clean_report}
    </div>
    """, unsafe_allow_html=True)
    
    # Add download and share buttons
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    with col1:
        st.download_button(
            label="📥 Download Report",
            data=ai_report,
            file_name=f"ai_insights_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    with col2:
        if st.button("📋 Copy Report", use_container_width=True):
            st.write("✅ Report copied to clipboard!")
    
elif ai_report_error:
    st.warning(f"⚠️ {ai_report_error}")
    
    # Show fallback summary in same style
    st.markdown("""
    <div class="ai-report-container">
        <h1>📊 Basic Summary Report</h1>
        <hr>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    ### 📋 Dataset Summary
    - **Total Records:** {report['dataset_summary']['total_rows']:,} customers
    - **Total Features:** {report['dataset_summary']['total_columns']}
    - **Missing Values:** {report['dataset_summary']['missing_values']:,}
    - **Duplicates:** {report['dataset_summary']['duplicates']:,}
    
    ### 🤖 Machine Learning Results
    - **Best Model:** {best_result['best_model_name']}
    - **Model Score:** {best_result['best_score']:.4f}
    
    ### ⭐ Top Features
    {chr(10).join([f'- {feature}' for feature in top_features[:5]])}
    """)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; padding: 1rem 0;">
    <div>
        <span style="color: #a0a0b0;">Built with ❤️ using Streamlit, AutoML, and AI</span>
    </div>
    <div>
        <span style="color: #a0a0b0; font-size: 0.9rem;">
            ⚡ v2.0 • {datetime}
        </span>
    </div>
</div>
""".format(datetime=datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)