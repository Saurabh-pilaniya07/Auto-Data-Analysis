import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import random

class ChartGenerator:
    def __init__(self, df):
        self.df = df
        self.color_palettes = [
            px.colors.qualitative.Set3,
            px.colors.qualitative.Pastel,
            px.colors.qualitative.Bold,
            px.colors.sequential.Viridis,
        ]
    
    def generate_all_charts(self):
        """Generate basic chart types"""
        charts = []
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # 1. Histograms for numeric columns
        for col in numeric_cols[:2]:
            if self.df[col].nunique() > 1:
                fig = self._create_histogram(col)
                charts.append((fig, "distribution", [col]))
        
        # 2. Bar charts for categorical columns
        for col in categorical_cols[:2]:
            if 2 <= self.df[col].nunique() <= 15:
                fig = self._create_bar_chart(col)
                charts.append((fig, "categorical", [col]))
        
        # 3. Correlation heatmap
        if len(numeric_cols) > 1:
            fig = self._create_correlation_heatmap(numeric_cols)
            charts.append((fig, "correlation", numeric_cols))
        
        # 4. Scatter plot
        if len(numeric_cols) >= 2:
            fig = self._create_scatter_plot(numeric_cols[0], numeric_cols[1])
            charts.append((fig, "relationship", [numeric_cols[0], numeric_cols[1]]))
        
        return charts
    
    def generate_advanced_charts(self):
        """Generate advanced chart types with fallback if dependencies missing"""
        charts = []
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # 1. Box plots (always available)
        for col in numeric_cols[:2]:
            try:
                fig = self._create_box_plot(col)
                charts.append((fig, "box_plot", [col]))
            except Exception as e:
                print(f"Box plot failed for {col}: {e}")
        
        # 2. Violin plots
        if numeric_cols and categorical_cols:
            try:
                fig = self._create_violin_plot(categorical_cols[0], numeric_cols[0])
                charts.append((fig, "violin_plot", [categorical_cols[0], numeric_cols[0]]))
            except Exception as e:
                print(f"Violin plot failed: {e}")
        
        # 3. Density contour plot (fallback to regular scatter if statsmodels missing)
        if len(numeric_cols) >= 2:
            try:
                fig = self._create_density_plot(numeric_cols[0], numeric_cols[1])
                charts.append((fig, "density_plot", [numeric_cols[0], numeric_cols[1]]))
            except Exception as e:
                # Fallback to enhanced scatter plot
                print(f"Density plot failed, using scatter: {e}")
                fig = self._create_enhanced_scatter_plot(numeric_cols[0], numeric_cols[1])
                charts.append((fig, "scatter_plot", [numeric_cols[0], numeric_cols[1]]))
        
        # 4. Area charts as alternative advanced visualization
        if len(numeric_cols) >= 1:
            try:
                fig = self._create_area_chart(numeric_cols[0])
                charts.append((fig, "area_chart", [numeric_cols[0]]))
            except Exception as e:
                print(f"Area chart failed: {e}")
        
        return charts
    
    def _get_random_palette(self):
        return random.choice(self.color_palettes)
    
    def _create_histogram(self, col):
        palette = self._get_random_palette()
        fig = px.histogram(self.df, x=col, title=f"Distribution of {col}", 
                          color_discrete_sequence=palette, nbins=30)
        return fig
    
    def _create_bar_chart(self, col):
        value_counts = self.df[col].value_counts().reset_index()
        value_counts.columns = ['category', 'count']
        palette = self._get_random_palette()
        fig = px.bar(value_counts, x='category', y='count', title=f"{col} Distribution",
                    color='category', color_discrete_sequence=palette)
        fig.update_layout(showlegend=False, xaxis_title=col, yaxis_title="Count")
        return fig
    
    def _create_correlation_heatmap(self, numeric_cols):
        corr_matrix = self.df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, title="Correlation Matrix", 
                       color_continuous_scale='RdBu_r', aspect="auto")
        return fig
    
    def _create_scatter_plot(self, col1, col2):
        # Simple scatter plot without trendline that requires statsmodels
        fig = px.scatter(self.df, x=col1, y=col2, title=f"{col1} vs {col2}")
        return fig
    
    def _create_enhanced_scatter_plot(self, col1, col2):
        # Enhanced scatter without statsmodels dependency
        fig = px.scatter(self.df, x=col1, y=col2, title=f"{col1} vs {col2}",
                        opacity=0.6, size_max=10)
        # Add a simple linear trendline using numpy (basic approximation)
        try:
            x_vals = self.df[col1].dropna()
            y_vals = self.df[col2].dropna()
            if len(x_vals) > 1 and len(y_vals) > 1:
                # Basic linear fit
                coefficients = np.polyfit(x_vals, y_vals, 1)
                polynomial = np.poly1d(coefficients)
                x_range = np.linspace(x_vals.min(), x_vals.max(), 100)
                y_range = polynomial(x_range)
                
                fig.add_trace(go.Scatter(x=x_range, y=y_range, 
                                       mode='lines', name='Trend',
                                       line=dict(color='red', width=2)))
        except Exception as e:
            print(f"Trendline failed: {e}")
        
        return fig
    
    def _create_box_plot(self, col):
        palette = self._get_random_palette()
        fig = px.box(self.df, y=col, title=f"Box Plot of {col}", 
                    color_discrete_sequence=palette)
        return fig
    
    def _create_violin_plot(self, cat_col, num_col):
        palette = self._get_random_palette()
        fig = px.violin(self.df, x=cat_col, y=num_col, 
                       title=f"{num_col} Distribution by {cat_col}",
                       color=cat_col, color_discrete_sequence=palette, box=True)
        return fig
    
    def _create_density_plot(self, col1, col2):
        # Try to create density plot, fallback if statsmodels not available
        try:
            # Check if statsmodels is available
            import statsmodels.api as sm
            fig = px.density_contour(self.df, x=col1, y=col2, 
                                    title=f"Density Plot: {col1} vs {col2}")
            fig.update_traces(contours_coloring="fill", contours_showlabels=True)
            return fig
        except ImportError:
            # Fallback to hexbin plot
            fig = px.density_heatmap(self.df, x=col1, y=col2,
                                   title=f"Distribution Heatmap: {col1} vs {col2}")
            return fig
    
    def _create_area_chart(self, col):
        # Create a simple area chart as an alternative advanced visualization
        if len(self.df) > 50:  # Only for larger datasets
            # Create binned data for area chart
            bins = np.linspace(self.df[col].min(), self.df[col].max(), 20)
            binned_data = pd.cut(self.df[col], bins=bins, include_lowest=True)
            area_data = binned_data.value_counts().sort_index().reset_index()
            area_data.columns = ['bin', 'count']
            area_data['bin_mid'] = [interval.mid for interval in area_data['bin']]
            
            fig = px.area(area_data, x='bin_mid', y='count', 
                         title=f"Distribution Area Chart: {col}")
            return fig
        else:
            # For small datasets, use line chart
            sorted_df = self.df.sort_values(col)
            fig = px.line(sorted_df, x=sorted_df.index, y=col,
                         title=f"Line Chart: {col}")
            return fig