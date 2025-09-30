import pandas as pd
import numpy as np
import plotly.express as px
import re
import os
import streamlit as st
from io import StringIO

class QASystem:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Enhanced API key handling for Windows
        self.gemini_model = None
        self.api_key = self._get_api_key()
        
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.gemini_model = genai.GenerativeModel('gemini-pro')
                st.sidebar.success("Google Gemini API connected successfully!")
            except Exception as e:
                st.sidebar.error(f"Gemini initialization failed: {e}")
                self.gemini_model = None
        else:
            st.sidebar.info("Using enhanced rule-based Q&A system. Add Google API key for Gemini features.")
    
    def _get_api_key(self):
        """Get API key from multiple sources with priority (Windows compatible)"""
        api_key = None
        
        # Try different methods to get the API key
        methods = [
            self._get_from_streamlit_secrets,
            self._get_from_env_file,
            self._get_from_env_var,
        ]
        
        for method in methods:
            api_key = method()
            if api_key:
                print(f"API key found via {method.__name__}")
                break
        
        return api_key
    
    def _get_from_streamlit_secrets(self):
        """Try to get API key from Streamlit secrets"""
        try:
            if hasattr(st, 'secrets'):
                if 'GOOGLE_API_KEY' in st.secrets:
                    return st.secrets['GOOGLE_API_KEY']
        except Exception as e:
            print(f"Streamlit secrets not available: {e}")
        return None
    
    def _get_from_env_file(self):
        """Try to get API key from .env file"""
        try:
            from dotenv import load_dotenv
            # Try common paths for Windows
            possible_paths = [
                '.env',
                '../.env',
                '../../.env',
                'C:/Users/sanju/.streamlit/secrets.toml',
                'C:/Users/sanju/OneDrive/Desktop/Automated Data Analysis and Visualization Platform/.streamlit/secrets.toml'
            ]
            
            for env_path in possible_paths:
                if os.path.exists(env_path):
                    if env_path.endswith('.toml'):
                        # Handle TOML file
                        import toml
                        with open(env_path, 'r') as f:
                            secrets = toml.load(f)
                            if 'GOOGLE_API_KEY' in secrets:
                                return secrets['GOOGLE_API_KEY']
                    else:
                        # Handle .env file
                        load_dotenv(env_path)
                        api_key = os.getenv('GOOGLE_API_KEY')
                        if api_key:
                            return api_key
                            
        except ImportError:
            print("python-dotenv or toml not installed, skipping env file")
        except Exception as e:
            print(f"Error loading env file: {e}")
        return None
    
    def _get_from_env_var(self):
        """Try to get API key from environment variable"""
        return os.getenv('GOOGLE_API_KEY')
    
    def answer_question(self, question):
        """Answer natural language questions about the data"""
        try:
            # Use enhanced rule-based system first
            rule_based_answer, chart = self._enhanced_rule_based_qa(question)
            
            # Try Gemini for complex questions if available
            if self.gemini_model and self._is_complex_question(question):
                try:
                    gemini_answer = self._gemini_qa(question)
                    if gemini_answer and len(gemini_answer) > 50:
                        return f"Gemini Analysis:\n{gemini_answer}", chart
                except Exception as e:
                    print(f"Gemini Q&A failed: {e}")
                    # Fallback to rule-based
                    return f"Rule-based Analysis:\n{rule_based_answer}", chart
            
            return f"Rule-based Analysis:\n{rule_based_answer}", chart
            
        except Exception as e:
            return f"I encountered an error: {str(e)}", None
    
    def _is_complex_question(self, question):
        """Determine if a question is complex enough for Gemini"""
        complex_keywords = ['relationship', 'pattern', 'trend', 'insight', 'analysis', 
                           'why', 'how', 'what if', 'predict', 'forecast', 'recommend']
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in complex_keywords)
    
    def _gemini_qa(self, question):
        """Use Google Gemini for complex questions"""
        try:
            context = self._create_data_context()
            prompt = f"""
            Based on the following dataset information:
            {context}
            
            Please provide a concise, data-driven answer to: {question}
            
            Focus on:
            - Key patterns and trends
            - Data-driven insights
            - Actionable recommendations
            - Statistical significance
            
            Keep the answer under 200 words.
            """
            
            response = self.gemini_model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            print(f"Gemini Q&A error: {e}")
            return None
    
    def _create_data_context(self):
        """Create comprehensive context from dataframe"""
        context = f"Dataset: {len(self.df)} rows x {len(self.df.columns)} columns\n\n"
        
        # Column summaries
        context += "Columns:\n"
        for col in self.df.columns:
            if col in self.numeric_cols:
                context += f"- {col} (numeric): range {self.df[col].min():.2f} to {self.df[col].max():.2f}, mean {self.df[col].mean():.2f}\n"
            else:
                top_values = self.df[col].value_counts().head(3)
                context += f"- {col} (categorical): {self.df[col].nunique()} values, top: {', '.join([str(v) for v in top_values.index])}\n"
        
        # Basic statistics
        if self.numeric_cols:
            context += f"\nCorrelations (top 3):\n"
            corr_matrix = self.df[self.numeric_cols].corr()
            for i, col1 in enumerate(self.numeric_cols[:3]):
                for j, col2 in enumerate(self.numeric_cols[i+1:i+2]):
                    if j < len(self.numeric_cols[i+1:i+2]):
                        context += f"- {col1} vs {col2}: {corr_matrix.loc[col1, col2]:.3f}\n"
        
        return context

    def _enhanced_rule_based_qa(self, question):
        """Enhanced rule-based question answering"""
        question_lower = question.lower()
        
        # Statistical questions
        if any(word in question_lower for word in ['statistic', 'summary', 'describe', 'overview']):
            return self._answer_dataset_summary(question)
        
        # Which has the highest
        elif any(word in question_lower for word in ['highest', 'maximum', 'max', 'largest']):
            return self._answer_extremum(question, "max")
        
        # Which has the lowest
        elif any(word in question_lower for word in ['lowest', 'minimum', 'min', 'smallest']):
            return self._answer_extremum(question, "min")
        
        # Average questions
        elif any(word in question_lower for word in ['average', 'mean', 'avg']):
            return self._answer_statistical(question, "mean")
        
        # Median questions
        elif 'median' in question_lower:
            return self._answer_statistical(question, "median")
        
        # Count questions
        elif any(word in question_lower for word in ['how many', 'count', 'number of', 'total']):
            return self._answer_count(question)
        
        # Correlation questions
        elif any(word in question_lower for word in ['correlation', 'relationship', 'related']):
            return self._answer_correlation(question)
        
        # Distribution questions
        elif any(word in question_lower for word in ['distribution', 'histogram', 'frequency']):
            return self._answer_distribution(question)
        
        # Unique values questions
        elif any(word in question_lower for word in ['unique', 'different', 'distinct']):
            return self._answer_unique_values(question)
        
        # Missing values questions
        elif any(word in question_lower for word in ['missing', 'null', 'na']):
            return self._answer_missing_values(question)
        
        else:
            return self._provide_comprehensive_help(), None
    
    def _answer_dataset_summary(self, question):
        """Provide comprehensive dataset summary"""
        summary = f"Dataset Summary:\n"
        summary += f"- Total records: {len(self.df)}\n"
        summary += f"- Total columns: {len(self.df.columns)}\n"
        summary += f"- Numeric columns: {len(self.numeric_cols)}\n"
        summary += f"- Categorical columns: {len(self.categorical_cols)}\n"
        summary += f"- Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n"
        
        if self.numeric_cols:
            summary += "Numeric Columns Summary:\n"
            for col in self.numeric_cols[:5]:  # Limit to first 5
                summary += f"- {col}: min={self.df[col].min():.2f}, max={self.df[col].max():.2f}, mean={self.df[col].mean():.2f}\n"
        
        # Create a summary chart
        if self.numeric_cols:
            chart = px.histogram(self.df, x=self.numeric_cols[0], 
                               title=f"Distribution of {self.numeric_cols[0]}")
            return summary, chart
        
        return summary, None
    
    def _answer_extremum(self, question, ext_type):
        """Answer questions about maximum or minimum values"""
        target_col = self._find_column_in_question(question, self.numeric_cols)
        
        if not target_col and self.numeric_cols:
            target_col = self.numeric_cols[0]
        
        if target_col:
            if ext_type == "max":
                value = self.df[target_col].max()
                idx = self.df[target_col].idxmax()
                answer = f"The maximum value of {target_col} is {value:.2f}."
                
                # Create a bar chart showing top values
                top_values = self.df.nlargest(10, target_col)
                if len(self.categorical_cols) > 0:
                    cat_col = self.categorical_cols[0]
                    chart = px.bar(top_values, x=cat_col, y=target_col, 
                                 title=f"Top 10 {target_col} values")
                else:
                    chart = px.histogram(self.df, x=target_col, 
                                       title=f"Distribution of {target_col}")
                
                return answer, chart
            else:
                value = self.df[target_col].min()
                answer = f"The minimum value of {target_col} is {value:.2f}."
                chart = px.histogram(self.df, x=target_col, 
                                   title=f"Distribution of {target_col}")
                return answer, chart
        
        return "I couldn't identify a numeric column in your question. Please specify which column you're interested in.", None
    
    def _answer_statistical(self, question, stat_type):
        """Answer statistical questions"""
        target_col = self._find_column_in_question(question, self.numeric_cols)
        
        if not target_col and self.numeric_cols:
            target_col = self.numeric_cols[0]
        
        if target_col:
            if stat_type == "mean":
                value = self.df[target_col].mean()
                answer = f"The average {target_col} is {value:.2f}."
            elif stat_type == "median":
                value = self.df[target_col].median()
                answer = f"The median {target_col} is {value:.2f}."
            
            chart = px.box(self.df, y=target_col, title=f"Distribution of {target_col}")
            return answer, chart
        
        return "I couldn't identify a numeric column in your question.", None
    
    def _answer_count(self, question):
        """Answer count questions"""
        question_lower = question.lower()
        
        # Count total records
        if any(word in question_lower for word in ['record', 'row', 'total', 'dataset']):
            count = len(self.df)
            return f"There are {count} total records in the dataset.", None
        
        # Count by category
        target_col = self._find_column_in_question(question, self.categorical_cols)
        if target_col:
            value_counts = self.df[target_col].value_counts()
            answer = f"Value counts for {target_col}:\n"
            for category, count in value_counts.head(10).items():
                percentage = (count / len(self.df)) * 100
                answer += f"- {category}: {count} records ({percentage:.1f}%)\n"
            
            # Create a bar chart
            chart_data = value_counts.head(10).reset_index()
            chart_data.columns = ['category', 'count']
            chart = px.bar(chart_data, x='category', y='count', 
                         title=f"Distribution of {target_col}")
            return answer, chart
        
        return "I couldn't understand what you want to count.", None
    
    def _answer_correlation(self, question):
        """Answer correlation questions"""
        if len(self.numeric_cols) >= 2:
            col1, col2 = self.numeric_cols[0], self.numeric_cols[1]
            correlation = self.df[col1].corr(self.df[col2])
            
            answer = f"The correlation between {col1} and {col2} is {correlation:.2f}. "
            if correlation > 0.7:
                answer += "This indicates a strong positive relationship."
            elif correlation > 0.3:
                answer += "This indicates a moderate positive relationship."
            elif correlation > -0.3:
                answer += "This indicates a weak relationship."
            elif correlation > -0.7:
                answer += "This indicates a moderate negative relationship."
            else:
                answer += "This indicates a strong negative relationship."
            
            chart = px.scatter(self.df, x=col1, y=col2, title=f"Relationship between {col1} and {col2}")
            return answer, chart
        
        return "I need at least two numeric columns for correlation analysis.", None
    
    def _answer_distribution(self, question):
        """Answer distribution questions"""
        target_col = self._find_column_in_question(question, self.numeric_cols)
        
        if not target_col and self.numeric_cols:
            target_col = self.numeric_cols[0]
        
        if target_col:
            stats = self.df[target_col].describe()
            answer = f"Distribution of {target_col}:\n"
            answer += f"- Mean: {stats['mean']:.2f}\n"
            answer += f"- Median: {self.df[target_col].median():.2f}\n"
            answer += f"- Standard Deviation: {stats['std']:.2f}\n"
            answer += f"- Range: {stats['min']:.2f} to {stats['max']:.2f}\n"
            
            chart = px.histogram(self.df, x=target_col, title=f"Distribution of {target_col}")
            return answer, chart
        
        return "I couldn't identify a numeric column for distribution analysis.", None
    
    def _answer_unique_values(self, question):
        """Answer questions about unique values"""
        target_col = self._find_column_in_question(question, self.df.columns)
        
        if target_col:
            unique_count = self.df[target_col].nunique()
            answer = f"Column '{target_col}' has {unique_count} unique values."
            
            if unique_count <= 10:
                unique_values = self.df[target_col].unique()
                answer += f"\nUnique values: {', '.join(map(str, unique_values))}"
            
            return answer, None
        
        return "Please specify which column you want to check for unique values.", None
    
    def _answer_missing_values(self, question):
        """Answer questions about missing values"""
        missing_counts = self.df.isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing > 0:
            answer = f"Missing values summary:\n"
            answer += f"- Total missing values: {total_missing}\n"
            answer += f"- Columns with missing values: {len(missing_counts[missing_counts > 0])}\n\n"
            
            for col, count in missing_counts[missing_counts > 0].items():
                percentage = (count / len(self.df)) * 100
                answer += f"- {col}: {count} missing ({percentage:.1f}%)\n"
        else:
            answer = "No missing values found in the dataset."
        
        return answer, None
    
    def _find_column_in_question(self, question, possible_columns):
        """Find which column is mentioned in the question"""
        question_lower = question.lower()
        for col in possible_columns:
            if col.lower() in question_lower:
                return col
        return None
    
    def _provide_comprehensive_help(self):
        """Provide comprehensive help when question isn't understood"""
        help_text = "I can answer questions about:\n\n"
        help_text += "Dataset Information:\n"
        help_text += "- 'Show me dataset summary'\n"
        help_text += "- 'What are the basic statistics?'\n\n"
        
        help_text += "Column Analysis:\n"
        help_text += "- 'Which column has the highest values?'\n"
        help_text += "- 'Show distribution of age column'\n"
        help_text += "- 'How many unique categories in gender?'\n\n"
        
        help_text += "Statistical Questions:\n"
        help_text += "- 'What is the average salary?'\n"
        help_text += "- 'Show correlation between age and income'\n"
        help_text += "- 'Are there any missing values?'\n\n"
        
        if self.numeric_cols:
            help_text += f"Available numeric columns: {', '.join(self.numeric_cols[:5])}\n"
        if self.categorical_cols:
            help_text += f"Available categorical columns: {', '.join(self.categorical_cols[:5])}"
        
        return help_text