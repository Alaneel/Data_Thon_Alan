"""
🏢 AI-Driven Company Intelligence Dashboard
SDS Datathon 2026 - Champions Group Dataset

This Streamlit dashboard provides interactive visualization and AI-powered
insights for company segmentation analysis.

Features:
- Overview dashboard with key metrics and cluster distribution
- Company search and exploration
- Cluster analysis with LLM-generated personas
- Anomaly detection and investigation
- Company comparison with AI competitive analysis

Usage:
    export GEMINI_API_KEY='your-api-key'
    streamlit run app.py

Author: SDS Datathon 2026 Team
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Page configuration
st.set_page_config(
    page_title="Company Intelligence Dashboard",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark professional theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #3d3d5c;
    }
    .cluster-tag {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .cluster-0 { background: #4f46e5; color: white; }
    .cluster-1 { background: #059669; color: white; }
    .cluster-2 { background: #d97706; color: white; }
    .cluster-3 { background: #dc2626; color: white; }
    .anomaly-badge { background: #ef4444; color: white; padding: 0.2rem 0.5rem; border-radius: 8px; }
    .normal-badge { background: #22c55e; color: white; padding: 0.2rem 0.5rem; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    """Load the segmentation results and full dataset"""
    results_df = pd.read_csv('company_segmentation_results.csv')
    full_df = pd.read_csv('champions_group_data.csv')
    return results_df, full_df

@st.cache_resource
def load_llm():
    """Initialize LLM generator"""
    try:
        from llm_insights import CompanyInsightGenerator
        api_key = os.environ.get('GEMINI_API_KEY')
        if api_key:
            return CompanyInsightGenerator(api_key=api_key)
        return CompanyInsightGenerator()
    except Exception as e:
        st.warning(f"LLM not available: {e}")
        return None

# Load resources
try:
    results_df, full_df = load_data()
    llm = load_llm()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("## 🏢 Company Intelligence")
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "📍 Navigation",
        ["📊 Overview", "💰 Lead Scoring", "🔍 Company Explorer", "📈 Cluster Analysis", "⚠️ Risk Detection", "⚖️ Company Comparison"],
        index=0
    )
    
    st.markdown("---")
    
    # Filters
    st.markdown("### 🔧 Filters")
    
    # Region filter
    regions = ['All'] + sorted(results_df['Region'].dropna().unique().tolist())
    selected_region = st.selectbox("Region", regions)
    
    # Cluster filter
    clusters = ['All'] + sorted(results_df['Cluster_Name'].dropna().unique().tolist())
    selected_cluster = st.selectbox("Cluster", clusters)
    
    # Anomaly filter
    show_anomalies_only = st.checkbox("Show Anomalies Only", False)
    
    st.markdown("---")
    st.markdown("### 📈 Dataset Stats")
    st.metric("Total Companies", f"{len(results_df):,}")
    st.metric("Clusters", results_df['Cluster'].nunique())
    anomaly_count = (results_df['Anomaly_Label'] == 'Anomaly').sum()
    st.metric("Anomalies", f"{anomaly_count} ({anomaly_count/len(results_df)*100:.1f}%)")

# Apply filters
filtered_df = results_df.copy()
if selected_region != 'All':
    filtered_df = filtered_df[filtered_df['Region'] == selected_region]
if selected_cluster != 'All':
    filtered_df = filtered_df[filtered_df['Cluster_Name'] == selected_cluster]
if show_anomalies_only:
    filtered_df = filtered_df[filtered_df['Anomaly_Label'] == 'Anomaly']

# ============== PAGE: OVERVIEW ==============
if page == "📊 Overview":
    st.markdown('<h1 class="main-header">📊 Company Intelligence Overview</h1>', unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 Companies in View", f"{len(filtered_df):,}")
    with col2:
        avg_revenue = filtered_df['Revenue_USD_Clean'].mean()
        st.metric("💰 Avg Revenue", f"${avg_revenue/1e6:.2f}M" if avg_revenue > 1e6 else f"${avg_revenue:,.0f}")
    with col3:
        avg_employees = filtered_df['Employees_Total_Clean'].mean()
        st.metric("👥 Avg Employees", f"{avg_employees:,.0f}")
    with col4:
        filtered_anomalies = (filtered_df['Anomaly_Label'] == 'Anomaly').sum()
        st.metric("⚠️ Anomalies", filtered_anomalies)
    
    st.markdown("---")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Cluster Distribution")
        cluster_counts = filtered_df['Cluster_Name'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Count']
        
        fig = px.pie(
            cluster_counts, 
            values='Count', 
            names='Cluster',
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=-0.2)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🌍 Regional Distribution")
        region_counts = filtered_df['Region'].value_counts().reset_index()
        region_counts.columns = ['Region', 'Count']
        
        fig = px.bar(
            region_counts,
            x='Region',
            y='Count',
            color='Count',
            color_continuous_scale='Viridis',
            labels={'Region': 'Region', 'Count': 'Number of Companies'}
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Revenue vs Employees scatter
    st.markdown("### 💼 Revenue vs Employees by Cluster")
    sample_df = filtered_df.sample(min(1000, len(filtered_df))) if len(filtered_df) > 1000 else filtered_df
    
    fig = px.scatter(
        sample_df,
        x='Employees_Total_Clean',
        y='Revenue_USD_Clean',
        color='Cluster_Name',
        size='Revenue_USD_Clean',
        size_max=30,
        hover_data=['Company Sites', 'SIC Description'],
        log_x=True,
        log_y=True,
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={
            'Employees_Total_Clean': 'Employees (Total)',
            'Revenue_USD_Clean': 'Revenue (USD)',
            'Cluster_Name': 'Cluster'
        }
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# ============== PAGE: LEAD SCORING ==============
elif page == "💰 Lead Scoring":
    st.markdown('<h1 class="main-header">💰 B2B Lead Scoring</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    **Business Use Case:** Prioritize sales outreach by scoring companies based on their revenue potential, 
    decision-making power, and data quality. Higher scores indicate higher-value prospects.
    """)
    
    # Check if Lead_Score column exists
    if 'Lead_Score' in filtered_df.columns:
        # Lead tier metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            priority = (filtered_df['Lead_Score'] >= 70).sum()
            st.metric("🔥 Priority Leads", f"{priority:,}", help="Score 70-100")
        with col2:
            hot = ((filtered_df['Lead_Score'] >= 50) & (filtered_df['Lead_Score'] < 70)).sum()
            st.metric("🌡️ Hot Leads", f"{hot:,}", help="Score 50-70")
        with col3:
            warm = ((filtered_df['Lead_Score'] >= 30) & (filtered_df['Lead_Score'] < 50)).sum()
            st.metric("☀️ Warm Leads", f"{warm:,}", help="Score 30-50")
        with col4:
            cold = (filtered_df['Lead_Score'] < 30).sum()
            st.metric("❄️ Cold Leads", f"{cold:,}", help="Score 0-30")
        
        st.markdown("---")
        
        # Lead Score Distribution
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📊 Lead Score Distribution")
            fig = px.histogram(
                filtered_df, x='Lead_Score', nbins=20,
                color_discrete_sequence=['#667eea'],
                labels={'Lead_Score': 'Lead Score', 'count': 'Number of Companies'}
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Lead Tier Breakdown")
            tier_counts = filtered_df['Lead_Tier'].value_counts().reset_index()
            tier_counts.columns = ['Tier', 'Count']
            fig = px.pie(
                tier_counts, values='Count', names='Tier',
                color='Tier',
                color_discrete_map={'Priority': '#ef4444', 'Hot': '#f97316', 'Warm': '#eab308', 'Cold': '#3b82f6'}
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Top Leads Table
        st.markdown("### 🏆 Top Priority Leads")
        top_leads = filtered_df.nlargest(50, 'Lead_Score')[['Company Sites', 'Country', 'SIC Description', 'Revenue_USD_Clean', 'Employees_Total_Clean', 'Lead_Score', 'Lead_Tier']]
        st.dataframe(top_leads, use_container_width=True, height=400)
        
        # Lead Score Explanation
        with st.expander("ℹ️ How Lead Scores are Calculated"):
            st.markdown("""
            **Lead Score (0-100)** is calculated based on:
            
            | Factor | Weight | Description |
            |--------|--------|-------------|
            | Revenue Potential | 40% | Higher revenue = higher score |
            | Decision Power | 25% | HQ/Parent entities score higher |
            | Productivity | 20% | Revenue per employee efficiency |
            | Data Quality | 15% | Completeness of company data |
            """)
    else:
        st.warning("Lead Score data not available. Please run enhanced_analysis.py first.")

# ============== PAGE: COMPANY EXPLORER ==============
elif page == "🔍 Company Explorer":
    st.markdown('<h1 class="main-header">🔍 Company Explorer</h1>', unsafe_allow_html=True)
    
    # Search box
    search_query = st.text_input("🔎 Search Companies", placeholder="Enter company name...")
    
    if search_query:
        search_df = filtered_df[filtered_df['Company Sites'].str.contains(search_query, case=False, na=False)]
    else:
        search_df = filtered_df
    
    st.markdown(f"**Showing {len(search_df):,} companies**")
    
    # Display table
    display_cols = ['Company Sites', 'Country', 'SIC Description', 'Employees_Total_Clean', 'Revenue_USD_Clean', 'Cluster_Name', 'Anomaly_Label']
    st.dataframe(
        search_df[display_cols].head(100),
        use_container_width=True,
        height=400
    )
    
    # Company detail view
    st.markdown("---")
    st.markdown("### 📋 Company Detail View")
    
    company_list = search_df['Company Sites'].dropna().head(50).tolist()
    if company_list:
        selected_company = st.selectbox("Select a company for detailed view", company_list)
        
        if selected_company:
            company_data = search_df[search_df['Company Sites'] == selected_company].iloc[0]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📌 Basic Info")
                st.markdown(f"**Company:** {company_data['Company Sites']}")
                st.markdown(f"**Country:** {company_data['Country']}")
                st.markdown(f"**Industry:** {company_data['SIC Description']}")
                st.markdown(f"**Employees:** {company_data['Employees_Total_Clean']:,.0f}")
                st.markdown(f"**Revenue:** ${company_data['Revenue_USD_Clean']:,.0f}")
            
            with col2:
                st.markdown("#### 🏷️ Segmentation")
                cluster_name = company_data['Cluster_Name']
                anomaly_status = company_data['Anomaly_Label']
                
                st.markdown(f"**Cluster:** {cluster_name}")
                if anomaly_status == 'Anomaly':
                    st.markdown("**Status:** <span class='anomaly-badge'>⚠️ Anomaly</span>", unsafe_allow_html=True)
                else:
                    st.markdown("**Status:** <span class='normal-badge'>✓ Normal</span>", unsafe_allow_html=True)
            
            # LLM Insight
            if st.button("🤖 Generate AI Insight", key="company_insight"):
                if llm and llm.enabled:
                    with st.spinner("Generating insight..."):
                        # Get cluster average for comparison
                        cluster_avg = filtered_df[filtered_df['Cluster_Name'] == cluster_name][['Revenue_USD_Clean', 'Employees_Total_Clean']].mean()
                        insight = llm.explain_anomaly(company_data, cluster_avg)
                        st.markdown("#### 🧠 AI Analysis")
                        st.markdown(insight)
                else:
                    st.warning("LLM is not enabled. Set GEMINI_API_KEY environment variable.")

# ============== PAGE: CLUSTER ANALYSIS ==============
elif page == "📈 Cluster Analysis":
    st.markdown('<h1 class="main-header">📈 Cluster Analysis</h1>', unsafe_allow_html=True)
    
    # Cluster profiles
    cluster_profiles = results_df.groupby('Cluster_Name').agg({
        'DUNS Number ': 'count',
        'Revenue_USD_Clean': 'median',
        'Employees_Total_Clean': 'median',
        'Region': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A',
        'SIC Description': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A'
    }).reset_index()
    cluster_profiles.columns = ['Cluster', 'Size', 'Median_Revenue', 'Median_Employees', 'Top_Region', 'Top_Industry']
    cluster_profiles['Percentage'] = (cluster_profiles['Size'] / cluster_profiles['Size'].sum() * 100).round(1).astype(str) + '%'
    
    st.markdown("### 📊 Cluster Profiles")
    st.dataframe(cluster_profiles, use_container_width=True)
    
    # Visual comparison
    st.markdown("### 📉 Cluster Comparison")
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            cluster_profiles,
            x='Cluster',
            y='Median_Revenue',
            title='Median Revenue by Cluster',
            color='Cluster',
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={'Cluster': 'Cluster', 'Median_Revenue': 'Median Revenue (USD)'}
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            cluster_profiles,
            x='Cluster',
            y='Median_Employees',
            title='Median Employees by Cluster',
            color='Cluster',
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={'Cluster': 'Cluster', 'Median_Employees': 'Median Employees'}
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # LLM Cluster Insights
    st.markdown("---")
    st.markdown("### 🤖 AI-Powered Cluster Insights")
    
    selected_cluster_for_insight = st.selectbox("Select Cluster for AI Analysis", cluster_profiles['Cluster'].tolist())
    
    if st.button("🧠 Generate Cluster Persona", key="cluster_persona"):
        if llm and llm.enabled:
            with st.spinner("Generating cluster insight..."):
                profile = cluster_profiles[cluster_profiles['Cluster'] == selected_cluster_for_insight].iloc[0].to_dict()
                insight = llm.generate_cluster_insight(
                    cluster_id=selected_cluster_for_insight,
                    profile={
                        'Size': profile['Size'],
                        'Percentage': profile['Percentage'],
                        'Median_Revenue_USD': profile['Median_Revenue'],
                        'Median_Employees': profile['Median_Employees'],
                        'Top_Region': profile['Top_Region'],
                        'Top_Industry': profile['Top_Industry'],
                        'Top_Entity_Type': 'Mixed'
                    }
                )
                st.markdown(insight)
        else:
            st.warning("LLM is not enabled. Set GEMINI_API_KEY environment variable.")

# ============== PAGE: RISK DETECTION ==============
elif page == "⚠️ Risk Detection":
    st.markdown('<h1 class="main-header">⚠️ Risk Detection</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    **Business Use Case:** Identify high-risk entities for due diligence, compliance checks, 
    and data quality verification before business engagement.
    """)
    
    # Risk metrics
    col1, col2, col3, col4 = st.columns(4)
    
    has_risk_cols = 'Risk_Shell' in results_df.columns
    
    if has_risk_cols:
        with col1:
            shell_risk = results_df['Risk_Shell'].sum()
            st.metric("🏚️ Shell Company Risk", f"{shell_risk:,}", help="High revenue, zero employees")
        with col2:
            anomalies = (results_df['Anomaly_Label'] == 'Anomaly').sum()
            st.metric("📊 Statistical Anomaly", f"{anomalies:,}", help="Isolation Forest detected")
        with col3:
            high_risk = (results_df['Risk_Flags'] >= 2).sum()
            st.metric("🚨 High-Risk Entities", f"{high_risk:,}", help="2+ risk flags")
        with col4:
            pct_risk = (results_df['Risk_Flags'] > 0).sum() / len(results_df) * 100
            st.metric("📈 Risk Rate", f"{pct_risk:.1f}%")
    else:
        anomalies = (results_df['Anomaly_Label'] == 'Anomaly').sum()
        st.metric("📊 Anomalies Detected", f"{anomalies:,}")
    
    st.markdown("---")
    
    # Risk type breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏚️ Shell Company Risk")
        st.markdown("Companies with high revenue but zero employees - potential holding companies or data issues.")
        if has_risk_cols:
            shell_df = results_df[results_df['Risk_Shell'] == True].nlargest(20, 'Revenue_USD_Clean')
            if len(shell_df) > 0:
                st.dataframe(
                    shell_df[['Company Sites', 'Revenue_USD_Clean', 'Employees_Total_Clean', 'SIC Description']].head(10),
                    use_container_width=True
                )
            else:
                st.info("No shell company risks detected.")
    
    with col2:
        st.markdown("### 📊 Statistical Anomalies")
        st.markdown("Companies with unusual patterns detected by Isolation Forest algorithm.")
        anomalies_df = results_df[results_df['Anomaly_Label'] == 'Anomaly'].nlargest(20, 'Revenue_USD_Clean')
        if len(anomalies_df) > 0:
            st.dataframe(
                anomalies_df[['Company Sites', 'Revenue_USD_Clean', 'Employees_Total_Clean', 'Cluster_Name']].head(10),
                use_container_width=True
            )
    
    st.markdown("---")
    
    # AI Investigation
    st.markdown("### 🕵️ AI Risk Investigation")
    
    risk_df = results_df[results_df['Anomaly_Label'] == 'Anomaly'].copy()
    anomaly_list = risk_df['Company Sites'].dropna().head(20).tolist()
    if anomaly_list:
        selected_anomaly = st.selectbox("Select entity to investigate", anomaly_list)
        
        if st.button("🔍 Investigate with AI", key="investigate_risk"):
            if llm and llm.enabled:
                with st.spinner("Analyzing risk..."):
                    anomaly_data = risk_df[risk_df['Company Sites'] == selected_anomaly].iloc[0]
                    cluster_avg = results_df[results_df['Cluster_Name'] == anomaly_data['Cluster_Name']][['Revenue_USD_Clean', 'Employees_Total_Clean']].mean()
                    insight = llm.explain_anomaly(anomaly_data, cluster_avg)
                    st.markdown(insight)
            else:
                st.warning("LLM is not enabled. Set GEMINI_API_KEY environment variable.")

# ============== PAGE: COMPANY COMPARISON ==============
elif page == "⚖️ Company Comparison":
    st.markdown('<h1 class="main-header">⚖️ Company Comparison</h1>', unsafe_allow_html=True)
    
    st.markdown("Select two companies to compare their profiles and get AI-powered competitive analysis.")
    
    col1, col2 = st.columns(2)
    
    company_list = filtered_df['Company Sites'].dropna().unique().tolist()[:500]
    
    with col1:
        st.markdown("### 🏢 Company A")
        company_a = st.selectbox("Select Company A", company_list, key="comp_a")
        if company_a:
            data_a = filtered_df[filtered_df['Company Sites'] == company_a].iloc[0]
            st.markdown(f"**Industry:** {data_a['SIC Description']}")
            st.markdown(f"**Employees:** {data_a['Employees_Total_Clean']:,.0f}")
            st.markdown(f"**Revenue:** ${data_a['Revenue_USD_Clean']:,.0f}")
            st.markdown(f"**Cluster:** {data_a['Cluster_Name']}")
    
    with col2:
        st.markdown("### 🏢 Company B")
        remaining_companies = [c for c in company_list if c != company_a]
        company_b = st.selectbox("Select Company B", remaining_companies, key="comp_b")
        if company_b:
            data_b = filtered_df[filtered_df['Company Sites'] == company_b].iloc[0]
            st.markdown(f"**Industry:** {data_b['SIC Description']}")
            st.markdown(f"**Employees:** {data_b['Employees_Total_Clean']:,.0f}")
            st.markdown(f"**Revenue:** ${data_b['Revenue_USD_Clean']:,.0f}")
            st.markdown(f"**Cluster:** {data_b['Cluster_Name']}")
    
    if company_a and company_b:
        # Visual comparison
        st.markdown("---")
        st.markdown("### 📊 Quick Comparison")
        
        comparison_data = pd.DataFrame({
            'Metric': ['Employees', 'Revenue (USD)'],
            'Company A': [data_a['Employees_Total_Clean'], data_a['Revenue_USD_Clean']],
            'Company B': [data_b['Employees_Total_Clean'], data_b['Revenue_USD_Clean']]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Company A', x=comparison_data['Metric'], y=comparison_data['Company A'], marker_color='#667eea'))
        fig.add_trace(go.Bar(name='Company B', x=comparison_data['Metric'], y=comparison_data['Company B'], marker_color='#764ba2'))
        fig.update_layout(
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # AI Comparison
        st.markdown("---")
        if st.button("🤖 Generate AI Competitive Analysis", key="compare_ai"):
            if llm and llm.enabled:
                with st.spinner("Generating competitive analysis..."):
                    insight = llm.compare_companies(data_a, data_b)
                    st.markdown("### 🧠 AI Competitive Analysis")
                    st.markdown(insight)
            else:
                st.warning("LLM is not enabled. Set GEMINI_API_KEY environment variable.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>🏢 SDS Datathon 2026 - AI-Driven Company Intelligence Dashboard</div>",
    unsafe_allow_html=True
)
