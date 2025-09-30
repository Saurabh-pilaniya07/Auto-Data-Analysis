import streamlit as st
import pandas as pd
import numpy as np
import os
from utils.data_processing import DataProcessor
from utils.visualization import ChartGenerator
from utils.insights import InsightGenerator
from utils.qa_system import QASystem
from utils.auto_ml import AutoML
import tempfile
import base64
from io import BytesIO, StringIO
import time
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import plotly.io as pio
from reportlab.lib.utils import ImageReader


# Page configuration
st.set_page_config(
    page_title="Auto Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .nav-container {
        display: flex;
        flex-direction: column;
        gap: 5px;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'df': None,
        'processed': False,
        'qa_system': None,
        'data_processor': None,
        'profile_report': None,
        'file_uploaded': False,
        'file_name': None,
        'current_page': "Data Upload",
        'generated_report': None,
        'report_generated': False,
        'view_report': False,
        'generation_status': "ready",
        'view_full_report': False,
        'pdf_report': None,
        'chart_insights': None,
        'profile': None,
        'cleaned_df': None,
        'pdf_data': None,
        'report_status': "ready",
        'view_pdf_report': False,
        'advanced_charts': None,
        'navigation_triggered': False,
        'charts_generated': False,
        'last_chart_generation': None,
        'pdf_generation_in_progress': False,
        'pdf_generated': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialize session state
initialize_session_state()

def generate_pdf(profile, chart_insights, df):
    """Generate PDF report without chart images to ensure fast completion"""
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=20)
        
        styles = getSampleStyleSheet()
        story = []
        
        # Title Section
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=20,
            spaceAfter=20,
            alignment=1,
            textColor=colors.HexColor('#1f77b4')
        )
        story.append(Paragraph("COMPREHENSIVE DATA ANALYSIS REPORT", title_style))
        story.append(Paragraph(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # 1. DATASET OVERVIEW
        story.append(Paragraph("1. DATASET OVERVIEW", styles['Heading2']))
        story.append(Spacer(1, 10))
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        overview_data = [
            ["Total Records", str(df.shape[0])],
            ["Total Columns", str(df.shape[1])],
            ["Numeric Columns", str(len(numeric_cols))],
            ["Categorical Columns", str(len(categorical_cols))],
            ["Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"]
        ]
        
        overview_table = Table(overview_data, colWidths=[2*inch, 2*inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f2f6')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(overview_table)
        story.append(Spacer(1, 20))
        
        # 2. DATA PROFILE DETAILS
        if profile:
            story.append(Paragraph("2. DATA PROFILE ANALYSIS", styles['Heading2']))
            story.append(Spacer(1, 10))
            
            # Data Overview Table
            story.append(Paragraph("Data Overview", styles['Heading3']))
            if 'overview' in profile and not profile['overview'].empty:
                overview_data = [["Column", "Non-Null Count", "Null Count", "Data Type"]]
                for _, row in profile['overview'].iterrows():
                    overview_data.append([
                        str(row['Column']), 
                        str(row['Non-Null Count']), 
                        str(row['Null Count']), 
                        str(row['Data Type'])
                    ])
                
                profile_table = Table(overview_data, repeatRows=1)
                profile_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f9f9f9')),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('FONTSIZE', (0, 0), (-1, -1), 7)
                ]))
                story.append(profile_table)
                story.append(Spacer(1, 15))
            
            # Missing Values Analysis
            story.append(Paragraph("Missing Values Analysis", styles['Heading3']))
            if 'missing_values' in profile and not profile['missing_values'].empty:
                missing_data = [["Column", "Missing Values", "Percentage"]]
                for _, row in profile['missing_values'].iterrows():
                    missing_data.append([str(row['Column']), str(row['Missing Values']), f"{row['Percentage']:.2f}%"])
                
                missing_table = Table(missing_data, repeatRows=1)
                missing_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f9f9f9')),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey)
                ]))
                story.append(missing_table)
                story.append(Spacer(1, 15))
        
        # 3. VISUAL ANALYSIS - TEXT ONLY (No chart images)
        if chart_insights:
            story.append(Paragraph("3. VISUAL ANALYSIS & INSIGHTS", styles['Heading2']))
            story.append(Spacer(1, 10))
            
            for i, (chart, chart_type, columns, insight) in enumerate(chart_insights[:4]):  # Limit to 4 charts
                story.append(Paragraph(f"Visualization {i+1}: {chart_type.upper()} - {', '.join(columns)}", styles['Heading3']))
                story.append(Spacer(1, 5))
                
                # Add insights only (no chart images for speed)
                story.append(Paragraph("Key Insights:", styles['Heading4']))
                insight_lines = insight.split('\n')
                for line in insight_lines:
                    if line.strip() and len(line.strip()) > 5:
                        clean_line = line.strip().replace('**', '')
                        story.append(Paragraph(f"• {clean_line}", styles['Normal']))
                
                story.append(Spacer(1, 15))
        
        # 4. RECOMMENDATIONS
        story.append(Paragraph("4. RECOMMENDATIONS", styles['Heading2']))
        story.append(Spacer(1, 10))
        
        recommendations = [
            "Review and address identified data quality issues",
            "Consider the insights from visual analysis for decision making",
            "Use correlation findings for predictive modeling opportunities",
            "Monitor key metrics over time for trend analysis",
            "Validate findings with domain experts where applicable"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
        
        story.append(Spacer(1, 20))
        story.append(Paragraph("--- END OF REPORT ---", styles['Heading2']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
        
    except Exception as e:
        print(f"PDF generation error: {e}")
        return None

def display_pdf_viewer(pdf_data):
    """Display PDF in the app using base64 encoding"""
    try:
        base64_pdf = base64.b64encode(pdf_data).decode('utf-8')
        pdf_display = f'''
        <iframe src="data:application/pdf;base64,{base64_pdf}" 
                width="100%" 
                height="600" 
                type="application/pdf"
                style="border: 1px solid #ccc; border-radius: 5px;">
        </iframe>
        '''
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying PDF: {str(e)}")

# App title
st.markdown('<h1 class="main-header">Automated Data Analysis Platform</h1>', unsafe_allow_html=True)

def main():
    # Define pages
    pages = {
        "Data Upload": "Upload Your Dataset",
        "Data Profiling": "Data Quality Report", 
        "Visualization": "Advanced Visualizations",
        "Q&A System": "Natural Language Q&A",
        "AutoML": "Automated Machine Learning",
        "Report Generation": "Generate Comprehensive PDF Report"
    }
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown('<div class="nav-container">', unsafe_allow_html=True)
    
    # Create navigation buttons with unique keys
    current_page = st.session_state.current_page
    
    for page_name, page_title in pages.items():
        # Create a button for each page
        if st.sidebar.button(page_name, key=f"nav_{page_name.replace(' ', '_')}", use_container_width=True):
            if page_name != current_page:
                st.session_state.current_page = page_name
                st.session_state.view_pdf_report = False
                st.rerun()
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    # Get current page
    app_mode = st.session_state.current_page
    
    # Render the appropriate page
    if app_mode == "Data Upload":
        render_data_upload()
    elif app_mode == "Data Profiling":
        render_data_profiling()
    elif app_mode == "Visualization":
        render_visualization()
    elif app_mode == "Q&A System":
        render_qa_system()
    elif app_mode == "AutoML":
        render_automl()
    elif app_mode == "Report Generation":
        render_report_generation()
    else:
        render_data_upload()

def render_data_upload():
    st.header("Upload Your Dataset")
    
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Store in session state
            st.session_state.df = df
            st.session_state.processed = False
            st.session_state.data_processor = DataProcessor(df)
            st.session_state.file_uploaded = True
            st.session_state.file_name = uploaded_file.name
            st.session_state.profile_report = None
            st.session_state.generated_report = None
            st.session_state.report_generated = False
            st.session_state.view_report = False
            st.session_state.cleaned_df = None
            st.session_state.view_pdf_report = False
            st.session_state.qa_system = None
            st.session_state.advanced_charts = None
            st.session_state.charts_generated = False
            st.session_state.chart_insights = None
            st.session_state.report_status = "ready"
            st.session_state.pdf_data = None
            
            st.success("File uploaded successfully!")
            st.write(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Show preview
            with st.expander("Data Preview"):
                st.dataframe(df.head(3))
                st.write(f"Columns: {list(df.columns)}")
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    elif st.session_state.file_uploaded and st.session_state.df is not None:
        st.success(f"File already loaded: {st.session_state.file_name}")
        st.write(f"Dataset Shape: {st.session_state.df.shape[0]} rows, {st.session_state.df.shape[1]} columns")
        
        # Use a unique key for the clear button to avoid conflicts
        if st.button("Clear Current File & Upload New", key="clear_file_btn"):
            # Only reset data-related session state, not navigation-related
            data_keys_to_reset = [
                'df', 'processed', 'data_processor', 'profile_report', 
                'file_uploaded', 'file_name', 'generated_report', 'report_generated',
                'view_report', 'cleaned_df', 'view_pdf_report', 'qa_system',
                'advanced_charts', 'charts_generated', 'chart_insights', 
                'pdf_data', 'report_status'
            ]
            
            for key in data_keys_to_reset:
                if key in st.session_state:
                    st.session_state[key] = None
            
            # Re-initialize the data processor with empty state
            st.session_state.data_processor = None
            st.session_state.file_uploaded = False
            st.session_state.charts_generated = False
            st.session_state.report_status = "ready"
            
            st.success("File cleared! You can now upload a new file.")
            st.rerun()

def render_data_profiling():
    st.header("Data Quality Report")
    
    if st.session_state.df is not None:
        # Ensure data_processor is initialized
        if st.session_state.data_processor is None:
            st.session_state.data_processor = DataProcessor(st.session_state.df)
        
        # Generate profile report
        if st.session_state.profile_report is None:
            with st.spinner("Analyzing data quality..."):
                st.session_state.profile_report = st.session_state.data_processor.generate_data_profile()
        
        profile_report = st.session_state.profile_report
        
        # Show data preview
        st.subheader("Data Preview")
        st.dataframe(st.session_state.df.head())
        
        # Show basic info
        st.subheader("Basic Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", st.session_state.df.shape[0])
        with col2:
            st.metric("Total Columns", st.session_state.df.shape[1])
        with col3:
            st.metric("Memory Usage", f"{st.session_state.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Show data types
        st.subheader("Data Types")
        dtype_df = pd.DataFrame({
            'Column': st.session_state.df.columns,
            'Data Type': st.session_state.df.dtypes.astype(str),
            'Non-Null Count': st.session_state.df.count().values,
            'Null Count': st.session_state.df.isnull().sum().values
        })
        st.dataframe(dtype_df)
        
        # Show the profile report
        if profile_report:
            st.subheader("Data Overview")
            st.dataframe(profile_report['overview'])
            
            st.subheader("Missing Values")
            st.dataframe(profile_report['missing_values'])
            
            if not profile_report['stats'].empty:
                st.subheader("Descriptive Statistics")
                st.dataframe(profile_report['stats'])
            
            # Show issues and recommendations
            if profile_report['issues']:
                st.subheader("Data Quality Issues")
                for issue in profile_report['issues']:
                    st.warning(issue)
                
                st.subheader("Recommended Actions")
                if st.button("Apply Automated Data Cleaning"):
                    with st.spinner("Cleaning data..."):
                        cleaned_df = st.session_state.data_processor.auto_clean_data()
                        st.session_state.cleaned_df = cleaned_df
                        st.session_state.processed = True
                        st.session_state.profile_report = DataProcessor(cleaned_df).generate_data_profile()
                        st.success("Data cleaning completed!")
                        st.rerun()
            else:
                st.success("No significant data quality issues detected!")
                st.session_state.processed = True
        
        # Download cleaned dataset button
        if st.session_state.cleaned_df is not None:
            st.subheader("Download Cleaned Data")
            csv = st.session_state.cleaned_df.to_csv(index=False)
            st.download_button(
                label="Download Cleaned Dataset as CSV",
                data=csv,
                file_name="cleaned_dataset.csv",
                mime="text/csv"
            )
    else:
        st.warning("Please upload a dataset first from the Data Upload section.")

def render_visualization():
    st.header("Advanced Visualizations")
    
    if st.session_state.df is not None:
        df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df
        
        # Chart options
        st.sidebar.subheader("Chart Options")
        show_advanced = st.sidebar.checkbox("Show Advanced Charts", value=True)
        max_charts = st.sidebar.slider("Maximum Charts to Display", 3, 10, 6)
        
        # Check if we need to generate charts
        if not st.session_state.charts_generated or st.sidebar.button("Regenerate Charts"):
            try:
                chart_gen = ChartGenerator(df)
                insight_gen = InsightGenerator(df)
                
                with st.spinner("Generating visualizations..."):
                    # Generate basic charts
                    basic_charts = chart_gen.generate_all_charts()
                    
                    # Generate advanced charts if enabled
                    advanced_charts = []
                    if show_advanced:
                        advanced_charts = chart_gen.generate_advanced_charts()
                    
                    all_charts = basic_charts + advanced_charts
                    
                    if all_charts:
                        # Store charts for report generation
                        chart_insights = []
                        for chart, chart_type, columns in all_charts[:max_charts]:
                            insight = insight_gen.generate_insight(chart_type, columns)
                            chart_insights.append((chart, chart_type, columns, insight))
                        
                        st.session_state.chart_insights = chart_insights
                        st.session_state.charts_generated = True
                        st.success(f"Generated {len(chart_insights)} charts!")
                    else:
                        st.info("No charts could be generated with the current data.")
            except Exception as e:
                st.error(f"Error generating visualizations: {str(e)}")
        
        # Display charts if available
        if st.session_state.chart_insights:
            # Display charts with tabs for better organization
            tab1, tab2 = st.tabs(["Basic Charts", "Advanced Charts"])
            
            with tab1:
                st.subheader("Basic Analytical Charts")
                basic_shown = 0
                for i, (chart, chart_type, columns, insight) in enumerate(st.session_state.chart_insights):
                    if chart_type in ['distribution', 'categorical', 'correlation', 'relationship'] and basic_shown < max_charts//2:
                        st.plotly_chart(chart, use_container_width=True)
                        with st.expander(f"Insights for {chart_type}"):
                            st.write(insight)
                        st.markdown("---")
                        basic_shown += 1
            
            with tab2:
                st.subheader("Advanced Visualizations")
                if show_advanced:
                    advanced_shown = 0
                    for i, (chart, chart_type, columns, insight) in enumerate(st.session_state.chart_insights):
                        if chart_type not in ['distribution', 'categorical', 'correlation', 'relationship'] and advanced_shown < max_charts//2:
                            st.plotly_chart(chart, use_container_width=True)
                            with st.expander(f"Insights for {chart_type}"):
                                st.write(insight)
                            st.markdown("---")
                            advanced_shown += 1
                    
                    if advanced_shown == 0:
                        st.info("No advanced charts available. Try enabling 'Show Advanced Charts' and regenerating.")
                else:
                    st.info("Enable 'Show Advanced Charts' in the sidebar to see advanced visualizations.")
        else:
            st.info("Click 'Regenerate Charts' in the sidebar to generate visualizations.")
            
    else:
        st.warning("Please upload a dataset first.")

def render_qa_system():
    st.header("Natural Language Q&A with Google Gemini")
    
    if st.session_state.df is not None:
        # Initialize Q&A system
        if st.session_state.qa_system is None:
            with st.spinner("Initializing Q&A system..."):
                st.session_state.qa_system = QASystem(st.session_state.df)
        
        # Q&A interface
        question = st.text_area("Ask a question about your data:", 
                               placeholder="e.g., What are the summary statistics? Which column has the highest values? Show me correlations between numeric columns.")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("Get Answer", use_container_width=True)
        
        if ask_button and question:
            with st.spinner("Analyzing your question..."):
                try:
                    answer, chart = st.session_state.qa_system.answer_question(question)
                    
                    st.subheader("Answer:")
                    st.write(answer)
                    
                    if chart is not None:
                        st.subheader("Visualization:")
                        st.plotly_chart(chart, use_container_width=True)
                    else:
                        st.info("Try asking questions about specific columns or relationships for visualizations.")
                        
                except Exception as e:
                    st.error(f"Error processing your question: {str(e)}")
        
        # Example questions
        with st.expander("Example Questions"):
            st.write("""
            - What are the basic statistics of numeric columns?
            - Which column has the most missing values?
            - Show me the distribution of [column_name]
            - What is the correlation between numeric columns?
            - How many unique values are in [categorical_column]?
            """)
    else:
        st.warning("Please upload a dataset first from the Data Upload section.")

def render_automl():
    st.header("Automated Machine Learning")
    
    if st.session_state.df is not None:
        df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df
        
        if df is not None:
            try:
                auto_ml = AutoML(df)
                
                st.info("This feature automatically trains machine learning models based on your data.")
                
                target_col = st.selectbox("Select target variable", 
                                         [""] + list(df.columns))
                
                if st.button("Run AutoML Analysis"):
                    with st.spinner("Training machine learning models..."):
                        try:
                            if target_col == "":
                                result = auto_ml.auto_detect_and_train()
                            else:
                                result = auto_ml.train_model(target_col)
                            
                            st.subheader("AutoML Results")
                            st.write(f"Problem Type: {result['problem_type']}")
                            st.write(f"Target Variable: {result['target']}")
                            st.write(f"Best Model: {result['best_model']}")
                            st.write(f"Validation Score: {result['score']:.4f}")
                            
                            if 'feature_importance' in result and result['feature_importance'] is not None:
                                st.subheader("Feature Importance")
                                st.plotly_chart(result['feature_importance'], use_container_width=True)
                                
                        except Exception as e:
                            st.error(f"AutoML failed: {str(e)}")
            except Exception as e:
                st.error(f"Error in AutoML section: {str(e)}")
        else:
            st.warning("Please upload a dataset first.")
    else:
        st.warning("Please upload a dataset first.")

def render_report_generation():
    st.header("Generate Comprehensive PDF Report")
    
    if st.session_state.df is not None:
        st.info("This will generate a comprehensive PDF report containing all data profiling information, visualizations, and insights.")
        
        # Check if we have charts ready
        if st.session_state.chart_insights is None:
            st.warning("Please generate visualizations first in the Visualization section to include charts in the report.")
            return
        
        # Initialize PDF generation state
        if 'pdf_generation_in_progress' not in st.session_state:
            st.session_state.pdf_generation_in_progress = False
        if 'pdf_generated' not in st.session_state:
            st.session_state.pdf_generated = False
        
        # Generate PDF button
        if not st.session_state.pdf_generation_in_progress and not st.session_state.pdf_generated:
            if st.button("Generate PDF Report", use_container_width=True, type="primary", key="generate_pdf_btn"):
                st.session_state.pdf_generation_in_progress = True
                st.session_state.pdf_generated = False
        
        # PDF generation process
        if st.session_state.pdf_generation_in_progress:
            try:
                with st.spinner("Generating PDF report... Please wait."):
                    start_time = time.time()
                    df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df
                    
                    # Ensure profile report exists
                    if st.session_state.profile_report is None:
                        data_processor = DataProcessor(df)
                        st.session_state.profile_report = data_processor.generate_data_profile()
                    
                    # Generate PDF without charts first (fast)
                    pdf_bytes = generate_pdf(
                        st.session_state.profile_report, 
                        st.session_state.chart_insights, 
                        df
                    )
                    
                    if pdf_bytes:
                        st.session_state.pdf_data = pdf_bytes
                        st.session_state.pdf_generation_in_progress = False
                        st.session_state.pdf_generated = True
                        end_time = time.time()
                        st.success(f"PDF Report generated successfully in {end_time - start_time:.2f} seconds!")
                    else:
                        st.error("Failed to generate PDF report")
                        st.session_state.pdf_generation_in_progress = False
                        
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
                st.session_state.pdf_generation_in_progress = False
        
        # Show report actions when completed
        if st.session_state.pdf_generated and st.session_state.pdf_data is not None:
            st.subheader("Report Ready")
            
            # Action buttons
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="Download PDF Report",
                    data=st.session_state.pdf_data,
                    file_name=f"analysis_report_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="download_pdf"
                )
            
            with col2:
                if st.button("View Report", use_container_width=True, key="view_report"):
                    st.session_state.view_pdf_report = True
            
            # Generate new report button
            if st.button("Generate New Report", key="new_report"):
                st.session_state.pdf_generated = False
                st.session_state.view_pdf_report = False
                st.session_state.pdf_data = None
                st.rerun()
            
            # PDF viewer
            if st.session_state.view_pdf_report:
                st.subheader("PDF Report Preview")
                display_pdf_viewer(st.session_state.pdf_data)
        
        # Initial state
        elif not st.session_state.pdf_generation_in_progress and not st.session_state.pdf_generated:
            st.info("Click the button above to generate a comprehensive PDF report.")
            
    else:
        st.warning("Please upload a dataset first from the Data Upload section.")

if __name__ == "__main__":
    main()
