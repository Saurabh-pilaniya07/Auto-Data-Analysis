import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go

class AutoML:
    def __init__(self, df):
        self.df = df.copy()
        self.le_dict = {}  # For storing label encoders
        self.imputer = SimpleImputer(strategy='most_frequent')  # For handling missing values
    
    def auto_detect_and_train(self):
        """Automatically detect the target variable and train a model"""
        # Try to identify a likely target column
        potential_targets = []
        
        # Look for common target names
        common_targets = ['target', 'label', 'class', 'result', 'outcome', 'score', 
                         'rating', 'status', 'default', 'churn', 'purchase']
        
        for col in self.df.columns:
            col_lower = col.lower()
            if any(common in col_lower for common in common_targets):
                potential_targets.append(col)
        
        # If no common names found, use heuristic: column with fewest unique values that's not ID-like
        if not potential_targets:
            for col in self.df.columns:
                if self.df[col].nunique() > 1 and self.df[col].nunique() < 20 and not any(
                    id_term in col.lower() for id_term in ['id', 'number', 'code']):
                    potential_targets.append(col)
        
        # If still no targets, use last column
        if not potential_targets and len(self.df.columns) > 0:
            potential_targets.append(self.df.columns[-1])
        
        if potential_targets:
            # Use the first potential target
            target = potential_targets[0]
            return self.train_model(target)
        else:
            raise ValueError("Could not automatically identify a target variable. Please specify one.")
    
    def train_model(self, target_column):
        """Train a model for the specified target column"""
        try:
            # Prepare the data
            X = self.df.drop(columns=[target_column])
            y = self.df[target_column]
            
            # Clean and preprocess the data
            X_cleaned = self._clean_data(X)
            y_cleaned = self._clean_target(y)
            
            # Handle categorical features with robust encoding
            X_encoded = self._encode_categorical(X_cleaned)
            
            # Handle missing values in features
            X_encoded = self._handle_missing_values(X_encoded)
            
            # Determine problem type
            if self._is_classification_problem(y_cleaned):
                problem_type = "classification"
                le = LabelEncoder()
                y_encoded = le.fit_transform(y_cleaned.astype(str))  # Ensure all values are strings
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                problem_type = "regression"
                y_encoded = y_cleaned.values
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded, y_encoded, test_size=0.2, random_state=42
            )
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Evaluate
            if problem_type == "classification":
                y_pred = model.predict(X_test)
                score = accuracy_score(y_test, y_pred)
            else:
                y_pred = model.predict(X_test)
                score = 1 - (mean_squared_error(y_test, y_pred) / np.var(y_test))  # R²-like metric
            
            # Get feature importance
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': X_encoded.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                feature_importance = px.bar(
                    importance_df.head(10), 
                    x='importance', 
                    y='feature', 
                    orientation='h',
                    title="Top 10 Feature Importances"
                )
            
            return {
                'problem_type': problem_type,
                'target': target_column,
                'best_model': type(model).__name__,
                'score': score,
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            raise ValueError(f"AutoML training failed: {str(e)}")
    
    def _clean_data(self, X):
        """Clean the data by converting all values to consistent types"""
        X_clean = X.copy()
        
        for col in X_clean.columns:
            # Convert all values to string first, then handle appropriately
            X_clean[col] = X_clean[col].astype(str)
            
            # Try to convert to numeric where possible
            try:
                numeric_vals = pd.to_numeric(X_clean[col], errors='coerce')
                # If more than 70% of values can be converted to numeric, use numeric
                if numeric_vals.notna().mean() > 0.7:
                    X_clean[col] = numeric_vals
            except:
                # Keep as string if conversion fails
                pass
                
        return X_clean
    
    def _clean_target(self, y):
        """Clean the target variable"""
        y_clean = y.copy()
        
        # Convert to string and handle missing values
        y_clean = y_clean.astype(str)
        y_clean = y_clean.replace(['nan', 'NaN', 'NULL', 'null', 'None', 'none', ''], np.nan)
        
        # For classification, fill NaN with most frequent value
        if y_clean.nunique() < 20:  # Likely classification
            most_frequent = y_clean.mode()
            if not most_frequent.empty:
                y_clean.fillna(most_frequent[0], inplace=True)
            else:
                y_clean.fillna('Unknown', inplace=True)
        else:  # Likely regression
            # Try to convert to numeric and use median
            try:
                y_numeric = pd.to_numeric(y_clean, errors='coerce')
                median_val = y_numeric.median()
                y_clean = y_numeric.fillna(median_val)
            except:
                # If conversion fails, use mode
                most_frequent = y_clean.mode()
                if not most_frequent.empty:
                    y_clean.fillna(most_frequent[0], inplace=True)
        
        return y_clean
    
    def _is_classification_problem(self, y):
        """Determine if this is a classification problem"""
        # If target has few unique values or is object type, it's classification
        if y.dtype == 'object' or y.nunique() <= 15:
            return True
        return False
    
    def _encode_categorical(self, X):
        """Encode categorical variables with robust type handling"""
        X_encoded = X.copy()
        
        for col in X_encoded.columns:
            # Skip if already numeric
            if pd.api.types.is_numeric_dtype(X_encoded[col]):
                continue
                
            # For categorical columns with reasonable cardinality
            if X_encoded[col].nunique() <= 50:
                try:
                    # Convert to string to ensure consistent data type
                    X_encoded[col] = X_encoded[col].astype(str)
                    
                    # Use label encoding
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X_encoded[col])
                    self.le_dict[col] = le
                except Exception as e:
                    print(f"Warning: Could not encode column {col}: {str(e)}")
                    # Drop column if encoding fails
                    X_encoded.drop(columns=[col], inplace=True)
            else:
                # High cardinality - drop for simplicity
                X_encoded.drop(columns=[col], inplace=True)
        
        return X_encoded
    
    def _handle_missing_values(self, X):
        """Handle missing values in the feature matrix"""
        X_clean = X.copy()
        
        for col in X_clean.columns:
            if X_clean[col].isnull().any():
                if pd.api.types.is_numeric_dtype(X_clean[col]):
                    # For numeric columns, use median
                    X_clean[col].fillna(X_clean[col].median(), inplace=True)
                else:
                    # For categorical columns, use mode
                    mode_val = X_clean[col].mode()
                    if not mode_val.empty:
                        X_clean[col].fillna(mode_val[0], inplace=True)
                    else:
                        X_clean[col].fillna(0, inplace=True)  # Fallback
        
        return X_clean