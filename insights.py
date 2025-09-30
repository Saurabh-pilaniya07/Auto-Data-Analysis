import pandas as pd
import numpy as np
from scipy import stats

class InsightGenerator:
    def __init__(self, df):
        self.df = df
    
    def generate_insight(self, chart_type, columns):
        """Generate detailed, comprehensive insights based on chart type"""
        if chart_type == "histogram" and len(columns) == 1:
            return self._generate_histogram_insight(columns[0])
        
        elif chart_type == "distribution" and len(columns) == 1:  # For box plots
            return self._generate_distribution_insight(columns[0])
        
        elif chart_type == "bar" and len(columns) == 1:
            return self._generate_bar_insight(columns[0])
        
        elif chart_type == "pie" and len(columns) == 1:
            return self._generate_pie_insight(columns[0])
        
        elif chart_type == "categorical" and len(columns) == 1:  # For donut/treemap
            return self._generate_categorical_insight(columns[0])
        
        elif chart_type == "scatter" and len(columns) == 2:
            return self._generate_scatter_insight(columns[0], columns[1])
        
        elif chart_type == "relationship" and len(columns) == 2:
            return self._generate_relationship_insight(columns[0], columns[1])
        
        elif chart_type == "correlation" and len(columns) > 1:
            return self._generate_correlation_insight(columns)
        
        elif chart_type == "timeseries" and len(columns) == 2:
            return self._generate_timeseries_insight(columns[0], columns[1])
        
        elif chart_type == "comparison" and len(columns) == 2:  # For violin plots
            return self._generate_comparison_insight(columns[0], columns[1])
        
        else:
            return self._generate_general_insight(chart_type, columns)
    
    def _generate_histogram_insight(self, col):
        """Generate detailed insight for histogram"""
        data = self.df[col].dropna()
        
        if len(data) == 0:
            return f"No data available for {col} to generate insights."
        
        # Comprehensive statistics
        stats = data.describe()
        skewness = data.skew()
        kurtosis = data.kurtosis()
        
        insight = f"**Distribution Analysis for {col}**\n\n"
        
        # Shape analysis
        insight += "**Distribution Shape:** "
        if abs(skewness) > 1:
            direction = "right" if skewness > 0 else "left"
            insight += f"Strong {direction} skew (skewness: {skewness:.2f}). "
            insight += f"This indicates most values cluster on one side with outliers extending to the {direction}. "
        elif abs(skewness) > 0.5:
            direction = "right" if skewness > 0 else "left"
            insight += f"Moderate {direction} skew. The data is somewhat asymmetrical. "
        else:
            insight += "Relatively symmetric distribution. The data is well-balanced. "
        
        # Kurtosis analysis
        insight += f"Kurtosis: {kurtosis:.2f} indicates "
        if kurtosis > 3:
            insight += "heavier tails than normal distribution (leptokurtic). "
        elif kurtosis < 3:
            insight += "lighter tails than normal distribution (platykurtic). "
        else:
            insight += "normal tail behavior (mesokurtic). "
        
        # Value analysis
        insight += f"\n\n**Value Statistics:**\n"
        insight += f"- Range: {stats['min']:.2f} to {stats['max']:.2f}\n"
        insight += f"- Mean: {stats['mean']:.2f}, Median: {data.median():.2f}\n"
        insight += f"- Standard Deviation: {stats['std']:.2f} ("
        
        # Variability context
        cv = stats['std'] / stats['mean'] if stats['mean'] != 0 else stats['std']
        if cv > 0.5:
            insight += "high variability"
        elif cv > 0.2:
            insight += "moderate variability"
        else:
            insight += "low variability"
        insight += ")\n"
        
        # Percentiles
        insight += f"- IQR: {stats['75%'] - stats['25%']:.2f} (Q1: {stats['25%']:.2f}, Q3: {stats['75%']:.2f})\n"
        
        # Outlier analysis
        Q1 = stats['25%']
        Q3 = stats['75%']
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        insight += f"- Potential outliers: {len(outliers)} values outside {lower_bound:.2f} to {upper_bound:.2f}\n"
        
        # Business implications
        insight += f"\n**Business Implications:**\n"
        if len(outliers) > 0:
            insight += f"- Investigate {len(outliers)} potential outliers for data quality issues\n"
        if abs(skewness) > 1:
            insight += f"- Consider data transformation for modeling due to strong skewness\n"
        insight += f"- {col} shows typical values around {stats['mean']:.2f} with expected range {stats['min']:.2f}-{stats['max']:.2f}"
        
        return insight
    
    def _generate_distribution_insight(self, col):
        """Generate insight for box plot distribution"""
        data = self.df[col].dropna()
        stats = data.describe()
        
        insight = f"**Distribution Summary for {col}**\n\n"
        
        insight += f"**Central Tendency:** Mean = {stats['mean']:.2f}, Median = {data.median():.2f}. "
        if abs(stats['mean'] - data.median()) / stats['mean'] > 0.1:
            insight += "The difference between mean and median suggests some skewness. "
        
        insight += f"\n\n**Spread:** Standard deviation = {stats['std']:.2f}. "
        insight += f"Interquartile Range (IQR) = {stats['75%'] - stats['25%']:.2f}, "
        insight += f"showing middle 50% of values range from {stats['25%']:.2f} to {stats['75%']:.2f}.\n"
        
        # Outlier analysis
        Q1, Q3 = stats['25%'], stats['75%']
        IQR = Q3 - Q1
        outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
        
        insight += f"**Outliers:** {len(outliers)} potential outliers detected. "
        if len(outliers) > 0:
            insight += f"These extreme values may represent unusual cases or data errors.\n"
        
        insight += f"\n**Data Quality:** {data.isnull().sum()} missing values. "
        insight += "The distribution appears " + ("normal" if abs(data.skew()) < 0.5 else "skewed") + "."
        
        return insight
    
    def _generate_bar_insight(self, col):
        """Generate insight for bar chart"""
        value_counts = self.df[col].value_counts()
        total = len(self.df)
        
        insight = f"**Category Analysis for {col}**\n\n"
        insight += f"**Diversity:** {len(value_counts)} unique categories\n"
        
        # Top categories
        top_3 = value_counts.head(3)
        insight += "**Top Categories:**\n"
        for i, (category, count) in enumerate(top_3.items(), 1):
            percentage = (count / total) * 100
            insight += f"{i}. {category}: {count} ({percentage:.1f}%)\n"
        
        # Balance analysis
        dominance = value_counts.iloc[0] / total
        insight += f"\n**Balance:** "
        if dominance > 0.5:
            insight += f"Highly imbalanced - top category dominates with {dominance*100:.1f}% share\n"
        elif dominance > 0.3:
            insight += f"Moderately imbalanced - some categories are more frequent\n"
        else:
            insight += f"Relatively balanced distribution across categories\n"
        
        # Missing values
        missing = self.df[col].isnull().sum()
        if missing > 0:
            insight += f"**Data Quality:** {missing} missing values ({missing/total*100:.1f}%)\n"
        
        insight += f"\n**Recommendations:** "
        if len(value_counts) > 10:
            insight += "Consider grouping smaller categories for analysis. "
        if dominance > 0.5:
            insight += "Address class imbalance if using for machine learning. "
        
        return insight
    
    def _generate_pie_insight(self, col):
        """Generate insight for pie chart"""
        value_counts = self.df[col].value_counts()
        total = len(self.df)
        
        insight = f"**Proportional Analysis for {col}**\n\n"
        
        # Major segments
        insight += "**Major Segments:**\n"
        for i, (category, count) in enumerate(value_counts.head(3).items(), 1):
            percentage = (count / total) * 100
            insight += f"- {category}: {percentage:.1f}% of total\n"
        
        # Minority segments
        if len(value_counts) > 3:
            minority_total = value_counts[3:].sum()
            insight += f"- Other categories: {minority_total/total*100:.1f}% combined\n"
        
        # Concentration analysis
        top_percentage = value_counts.iloc[0] / total * 100
        insight += f"\n**Concentration:** "
        if top_percentage > 60:
            insight += f"Highly concentrated - top category represents {top_percentage:.1f}%\n"
        elif top_percentage > 40:
            insight += f"Moderately concentrated\n"
        else:
            insight += f"Diverse distribution\n"
        
        insight += f"\n**Business Context:** This shows how {col} is distributed across different values. "
        insight += "Useful for understanding market segments, customer types, or category prevalence."
        
        return insight
    
    def _generate_categorical_insight(self, col):
        """Generate insight for categorical charts (donut/treemap)"""
        value_counts = self.df[col].value_counts()
        
        insight = f"**Categorical Overview for {col}**\n\n"
        insight += f"**Total Categories:** {len(value_counts)}\n"
        insight += f"**Most Frequent:** {value_counts.index[0]} ({value_counts.iloc[0]} records)\n"
        insight += f"**Least Frequent:** {value_counts.index[-1]} ({value_counts.iloc[-1]} records)\n"
        
        # Diversity index (simplified)
        if len(value_counts) > 1:
            diversity = 1 - (value_counts.iloc[0] / value_counts.sum())**2
            insight += f"**Diversity Score:** {diversity:.2f} (0=monopoly, 1=perfect diversity)\n"
        
        insight += f"\n**Patterns:** "
        if value_counts.iloc[0] > value_counts.sum() * 0.4:
            insight += "Dominant category present. "
        if len(value_counts) > 10:
            insight += "High cardinality - consider grouping. "
        
        return insight
    
    def _generate_scatter_insight(self, col1, col2):
        """Generate insight for scatter plot"""
        clean_data = self.df[[col1, col2]].dropna()
        
        if len(clean_data) < 10:
            return f"Insufficient data points for meaningful correlation analysis between {col1} and {col2}."
        
        correlation = clean_data[col1].corr(clean_data[col2])
        
        insight = f"**Relationship Analysis: {col1} vs {col2}**\n\n"
        
        insight += f"**Correlation:** {correlation:.3f} - "
        if correlation > 0.7:
            insight += "Strong positive relationship\n"
        elif correlation > 0.3:
            insight += "Moderate positive relationship\n"
        elif correlation > -0.3:
            insight += "Weak or no linear relationship\n"
        elif correlation > -0.7:
            insight += "Moderate negative relationship\n"
        else:
            insight += "Strong negative relationship\n"
        
        # Statistical significance
        if len(clean_data) > 30:
            p_value = stats.pearsonr(clean_data[col1], clean_data[col2])[1]
            insight += f"**Statistical Significance:** p-value = {p_value:.4f} - "
            insight += "Statistically significant\n" if p_value < 0.05 else "Not statistically significant\n"
        
        insight += f"\n**Data Points:** {len(clean_data)} pairs analyzed\n"
        
        # Pattern description
        insight += f"**Pattern:** "
        if correlation > 0.5:
            insight += f"As {col1} increases, {col2} tends to increase\n"
        elif correlation < -0.5:
            insight += f"As {col1} increases, {col2} tends to decrease\n"
        else:
            insight += "No clear linear pattern observed\n"
        
        insight += f"\n**Business Implication:** "
        if abs(correlation) > 0.5:
            insight += f"Strong relationship suggests potential for prediction or causal investigation"
        else:
            insight += "Variables appear independent in their current measurement"
        
        return insight
    
    def _generate_relationship_insight(self, col1, col2):
        """Generate insight for relationship charts"""
        return self._generate_scatter_insight(col1, col2)  # Reuse scatter insight
    
    def _generate_correlation_insight(self, columns):
        """Generate insight for correlation matrix"""
        corr_matrix = self.df[columns].corr()
        
        insight = "**Comprehensive Correlation Analysis**\n\n"
        
        # Find strongest relationships
        strong_relationships = []
        for i in range(len(columns)):
            for j in range(i+1, len(columns)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.5:
                    strong_relationships.append((columns[i], columns[j], corr))
        
        if strong_relationships:
            insight += "**Strong Relationships Found:**\n"
            for var1, var2, corr in strong_relationships:
                direction = "positive" if corr > 0 else "negative"
                insight += f"- {var1} ↔ {var2}: {corr:.3f} ({direction})\n"
        else:
            insight += "**No strong correlations** found among these variables\n"
        
        insight += f"\n**Variables Analyzed:** {len(columns)} numeric columns\n"
        insight += "**Interpretation:** Values closer to ±1 indicate stronger linear relationships"
        
        return insight
    
    def _generate_timeseries_insight(self, date_col, num_col):
        """Generate insight for time series"""
        time_series = self.df.groupby(date_col)[num_col].mean()
        
        insight = f"**Time Series Analysis: {num_col} over Time**\n\n"
        
        insight += f"**Time Period:** {time_series.index.min()} to {time_series.index.max()}\n"
        insight += f"**Data Points:** {len(time_series)} time periods\n"
        
        # Trend analysis
        if len(time_series) > 1:
            x = np.arange(len(time_series))
            y = time_series.values
            slope = np.polyfit(x, y, 1)[0]
            
            insight += f"**Overall Trend:** "
            if slope > 0:
                insight += f"Upward trend ({slope:.4f} per period)\n"
            elif slope < 0:
                insight += f"Downward trend ({slope:.4f} per period)\n"
            else:
                insight += "Relatively stable\n"
        
        # Volatility
        volatility = time_series.std() / time_series.mean() if time_series.mean() != 0 else time_series.std()
        insight += f"**Volatility:** {volatility:.3f} - "
        if volatility > 0.3:
            insight += "High fluctuation over time\n"
        elif volatility > 0.1:
            insight += "Moderate fluctuation\n"
        else:
            insight += "Stable over time\n"
        
        insight += f"\n**Pattern Analysis:** "
        if len(time_series) > 12:
            insight += "Potential seasonal patterns observable\n"
        else:
            insight += "Insufficient data for seasonal analysis\n"
        
        return insight
    
    def _generate_comparison_insight(self, cat_col, num_col):
        """Generate insight for comparison charts (violin/box)"""
        grouped = self.df.groupby(cat_col)[num_col]
        
        insight = f"**Comparison Analysis: {num_col} across {cat_col} categories**\n\n"
        
        insight += f"**Categories Compared:** {len(grouped)} groups\n"
        
        # Statistics by group
        means = grouped.mean()
        stds = grouped.std()
        
        insight += "**Group Statistics:**\n"
        for category in means.index:
            insight += f"- {category}: Mean = {means[category]:.2f}, Std = {stds[category]:.2f}\n"
        
        # Variation analysis
        max_mean = means.max()
        min_mean = means.min()
        mean_diff = max_mean - min_mean
        overall_std = self.df[num_col].std()
        
        insight += f"\n**Variation Analysis:** "
        if mean_diff > overall_std:
            insight += f"Significant differences between groups (difference: {mean_diff:.2f})\n"
        else:
            insight += f"Moderate differences between groups\n"
        
        insight += f"\n**Business Insight:** "
        if mean_diff > overall_std:
            insight += f"{cat_col} appears to be a strong differentiator for {num_col}"
        else:
            insight += f"{cat_col} has limited impact on {num_col} values"
        
        return insight
    
    def _generate_general_insight(self, chart_type, columns):
        """Generate general insight for unknown chart types"""
        return f"**Chart Analysis:** This {chart_type} chart shows the relationship between {', '.join(columns)}. Analyze patterns, trends, and outliers to derive meaningful business insights from this visualization."