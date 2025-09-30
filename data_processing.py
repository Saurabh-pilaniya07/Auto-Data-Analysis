import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, df):
        self.df = df.copy()
        
    def generate_data_profile(self):
        """Generate a comprehensive data quality report"""
        profile = {}
        
        # Basic overview
        profile['overview'] = pd.DataFrame({
            'Column': self.df.columns,
            'Non-Null Count': self.df.count().values,
            'Null Count': self.df.isnull().sum().values,
            'Data Type': self.df.dtypes.values
        })
        
        # Missing values
        missing_values = self.df.isnull().sum()
        missing_percent = (missing_values / len(self.df)) * 100
        profile['missing_values'] = pd.DataFrame({
            'Column': missing_values.index,
            'Missing Values': missing_values.values,
            'Percentage': missing_percent.values
        })
        
        # Descriptive statistics
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            profile['stats'] = self.df[numeric_cols].describe().T
        else:
            profile['stats'] = pd.DataFrame()
        
        # Identify data quality issues
        profile['issues'] = self._identify_data_issues()
        
        return profile
    
    def _identify_data_issues(self):
        """Identify data quality issues"""
        issues = []
        
        # Check for missing values
        missing_cols = self.df.columns[self.df.isnull().any()].tolist()
        if missing_cols:
            issues.append(f"Missing values in: {', '.join(missing_cols)}")
        
        # Check for duplicates
        duplicate_rows = self.df.duplicated().sum()
        if duplicate_rows > 0:
            issues.append(f"{duplicate_rows} duplicate rows found")
        
        return issues
    
    def auto_clean_data(self):
        """Automatically clean the dataset"""
        cleaned_df = self.df.copy()
        
        # Handle missing values
        for col in cleaned_df.columns:
            if cleaned_df[col].isnull().sum() > 0:
                if cleaned_df[col].dtype == 'object':
                    # For categorical, fill with mode
                    mode_val = cleaned_df[col].mode()
                    if not mode_val.empty:
                        cleaned_df[col].fillna(mode_val[0], inplace=True)
                else:
                    # For numerical, fill with median
                    median_val = cleaned_df[col].median()
                    cleaned_df[col].fillna(median_val, inplace=True)
        
        # Remove duplicates
        cleaned_df.drop_duplicates(inplace=True)
        
        return cleaned_df