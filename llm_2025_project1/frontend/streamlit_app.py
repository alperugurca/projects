import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Configure page
st.set_page_config(
    page_title="CV Analyzer",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

def analyze_cv(file):
    """Send CV file to backend for analysis"""
    files = {'file': file}
    response = requests.post('http://localhost:8000/analyze', files=files)
    return response.json()

def display_analysis(analysis):
    """Display CV analysis results"""
    if not analysis['success']:
        st.error(f"Analysis failed: {analysis['error']}")
        return
    
    result = analysis['analysis']
    
    # Overall Score
    col1, col2, col3 = st.columns(3)
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result['overall_score'],
            title={'text': "Overall Score"},
            gauge={'axis': {'range': [0, 100]},
                  'bar': {'color': "darkblue"},
                  'steps': [
                      {'range': [0, 50], 'color': "lightgray"},
                      {'range': [50, 75], 'color': "gray"},
                      {'range': [75, 100], 'color': "darkgray"}
                  ]}
        ))
        st.plotly_chart(fig)

    # Skills Analysis
    st.subheader("Skills Analysis")
    skills_df = pd.DataFrame(result['skills'])
    fig = px.bar(skills_df, x='name', y='level', 
                 title="Skills Assessment",
                 color='level')
    st.plotly_chart(fig)

    # Experience Impact
    st.subheader("Experience Analysis")
    experience_df = pd.DataFrame(result['experience'])
    fig = px.scatter(experience_df, x='company', y='impact_score',
                    size='impact_score', hover_data=['position', 'duration'],
                    title="Experience Impact Analysis")
    st.plotly_chart(fig)

    # Improvement Suggestions
    st.subheader("Improvement Suggestions")
    suggestions_df = pd.DataFrame(result['improvement_suggestions'])
    suggestions_df = suggestions_df.sort_values('priority', ascending=False)
    for _, row in suggestions_df.iterrows():
        with st.expander(f"{row['section']} (Priority: {row['priority']})"):
            st.write("Current State:", row['current_state'])
            st.write("Suggestion:", row['suggestion'])

    # Strengths and Weaknesses
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Strengths")
        for strength in result['strengths']:
            st.write("✅", strength)
    with col2:
        st.subheader("Weaknesses")
        for weakness in result['weaknesses']:
            st.write("❌", weakness)

    # Industry Fit
    st.subheader("Industry Fit")
    for industry in result['industry_fit']:
        st.write("🎯", industry)

    # Keyword Optimization
    st.subheader("Keyword Optimization")
    keywords_df = pd.DataFrame(list(result['keyword_optimization'].items()),
                             columns=['Keyword', 'Relevance'])
    fig = px.bar(keywords_df, x='Keyword', y='Relevance',
                 title="Keyword Relevance Analysis")
    st.plotly_chart(fig)

    # Save analysis to history
    st.session_state.analysis_history.append({
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'overall_score': result['overall_score']
    })

def show_dashboard():
    """Display analytics dashboard"""
    st.title("Analytics Dashboard")

    if not st.session_state.analysis_history:
        st.info("No analysis data available yet. Upload a CV to see analytics.")
        return

    # Convert history to DataFrame
    history_df = pd.DataFrame(st.session_state.analysis_history)

    # Score Distribution
    fig = px.histogram(history_df, x='overall_score',
                      title="Score Distribution",
                      nbins=10)
    st.plotly_chart(fig)

    # Score Timeline
    fig = px.line(history_df, x='timestamp', y='overall_score',
                  title="Score Timeline")
    st.plotly_chart(fig)

def main():
    st.title("CV Analyzer")
    
    # Sidebar navigation
    page = st.sidebar.radio("Navigation", ["CV Analysis", "Dashboard"])
    
    if page == "CV Analysis":
        st.write("Upload your CV for analysis")
        uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'docx'])
        
        if uploaded_file:
            with st.spinner("Analyzing CV..."):
                analysis = analyze_cv(uploaded_file)
                display_analysis(analysis)
                
            # Feedback collection
            st.divider()
            st.subheader("Feedback")
            feedback = st.slider("How helpful was this analysis?", 1, 5, 3)
            feedback_text = st.text_area("Additional comments (optional)")
            if st.button("Submit Feedback"):
                # Here you would typically send this to your backend
                st.success("Thank you for your feedback!")
    
    else:  # Dashboard page
        show_dashboard()

if __name__ == "__main__":
    main()