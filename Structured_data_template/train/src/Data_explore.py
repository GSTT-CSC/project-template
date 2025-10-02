# Interactive Data Exploration Notebook
# This notebook provides easy data visualization and insights using existing utilities

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import json

# Import your existing utilities
from utils.data_utils import DataProfiler, FeatureEngineer, DataTransformer
from utils.visualise import DataVisualizer

# For interactive widgets
try:
    import ipywidgets as widgets
    from IPython.display import display, HTML, clear_output
    WIDGETS_AVAILABLE = True
except ImportError:
    print("Install ipywidgets for interactive features: pip install ipywidgets")
    WIDGETS_AVAILABLE = False

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

class InteractiveDataExplorer:
    """Interactive data exploration tool using existing utilities."""
    
    def __init__(self):
        self.data = None
        self.target_column = None
        self.data_profiler = DataProfiler()
        self.visualizer = DataVisualizer(output_dir="exploration_plots")
        self.feature_engineer = FeatureEngineer()
        self.data_transformer = DataTransformer()
        self.profile = None
        
    def load_data(self, file_path):
        """Load data and perform initial setup."""
        try:
            self.data = pd.read_csv(file_path)
            print(f"✓ Data loaded successfully!")
            print(f"Shape: {self.data.shape}")
            print(f"Columns: {list(self.data.columns)}")
            return True
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
    
    def quick_overview(self):
        """Generate a quick data overview."""
        if self.data is None:
            print("Please load data first!")
            return
        
        print("="*60)
        print("QUICK DATA OVERVIEW")
        print("="*60)
        
        # Basic info
        print(f"Dataset Shape: {self.data.shape}")
        print(f"Memory Usage: {self.data.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
        print(f"Duplicate Rows: {self.data.duplicated().sum()}")
        
        # Column types
        print(f"\nColumn Types:")
        print(f"  Numerical: {len(self.data.select_dtypes(include=[np.number]).columns)}")
        print(f"  Categorical: {len(self.data.select_dtypes(include=['object', 'category']).columns)}")
        print(f"  Datetime: {len(self.data.select_dtypes(include=['datetime64']).columns)}")
        
        # Missing data
        missing_data = self.data.isnull().sum()
        columns_with_missing = missing_data[missing_data > 0]
        if len(columns_with_missing) > 0:
            print(f"\nColumns with Missing Data:")
            for col, count in columns_with_missing.items():
                pct = (count / len(self.data)) * 100
                print(f"  {col}: {count} ({pct:.1f}%)")
        else:
            print(f"\nNo missing data found!")
    
    def generate_comprehensive_profile(self, target_col=None):
        """Generate comprehensive data profile."""
        if self.data is None:
            print("Please load data first!")
            return
        
        self.target_column = target_col
        print("Generating comprehensive data profile...")
        
        self.profile = self.data_profiler.generate_data_profile(
            self.data, 
            target_col=target_col
        )
        
        # Save profile
        self.data_profiler.save_profile_report(self.profile, "exploration_data_profile.json")
        
        # Display key insights
        self._display_profile_insights()
    
    def _display_profile_insights(self):
        """Display key insights from the data profile."""
        if not self.profile:
            return
        
        print("\n" + "="*60)
        print("DATA PROFILE INSIGHTS")
        print("="*60)
        
        # Data Quality Score
        quality = self.profile.get('data_quality', {})
        if quality:
            print(f"Overall Data Quality Score: {quality.get('overall_quality', 0):.2f}/1.00")
            print(f"  Completeness: {quality.get('completeness', 0):.2f}")
            print(f"  Uniqueness: {quality.get('uniqueness', 0):.2f}")
            print(f"  Consistency: {quality.get('consistency', 0):.2f}")
        
        # Missing data summary
        missing = self.profile.get('missing_data', {})
        if missing:
            print(f"\nMissing Data: {missing.get('missing_percentage_overall', 0):.1f}% overall")
            print(f"Complete Rows: {missing.get('complete_rows_percentage', 0):.1f}%")
        
        # High correlations (if available)
        corr = self.profile.get('correlations', {})
        high_corr = corr.get('high_correlations', [])
        if high_corr:
            print(f"\nHigh Correlations Found:")
            for pair in high_corr[:5]:  # Show top 5
                print(f"  {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")
        
        # Target analysis (if available)
        if self.target_column and 'target_analysis' in self.profile:
            target_info = self.profile['target_analysis']
            print(f"\nTarget Variable '{self.target_column}':")
            print(f"  Type: {target_info.get('dtype', 'unknown')}")
            print(f"  Missing Values: {target_info.get('missing_count', 0)}")
            print(f"  Unique Values: {target_info.get('unique_count', 0)}")
            print(f"  Recommended Task: {target_info.get('recommended_task', 'unknown')}")
    
    def create_visualizations(self, target_col=None):
        """Create comprehensive visualizations."""
        if self.data is None:
            print("Please load data first!")
            return
        
        print("Creating visualizations...")
        
        # Basic distributions
        self.visualizer.plot_distributions(
            self.data,
            target_column=target_col,
            save_name="exploration_distributions"
        )
        
        # Correlation matrix
        numerical_data = self.data.select_dtypes(include=[np.number])
        if len(numerical_data.columns) > 1:
            self.visualizer.plot_correlation_matrix(
                self.data,
                save_name="exploration_correlations"
            )
        
        # Missing data patterns
        if self.data.isnull().any().any():
            self.visualizer.plot_missing_data_pattern(
                self.data,
                save_name="exploration_missing_patterns"
            )
        
        # Outlier analysis
        if len(numerical_data.columns) > 0:
            self.visualizer.plot_outliers_analysis(
                self.data,
                save_name="exploration_outliers"
            )
        
        print("Visualizations created in 'exploration_plots' directory!")
    
    def analyze_target_relationship(self, target_col):
        """Analyze relationships with target variable."""
        if self.data is None or target_col not in self.data.columns:
            print("Please load data and specify a valid target column!")
            return
        
        print(f"\nAnalyzing relationships with target: '{target_col}'")
        print("="*50)
        
        target_data = self.data[target_col]
        
        # Target distribution
        print(f"Target Distribution:")
        value_counts = target_data.value_counts()
        for value, count in value_counts.items():
            pct = (count / len(target_data)) * 100
            print(f"  {value}: {count} ({pct:.1f}%)")
        
        # Check for imbalance
        min_pct = (value_counts.min() / len(target_data)) * 100
        if min_pct < 10:
            print(f"\n Class imbalance detected! Smallest class: {min_pct:.1f}%")
        
        # Correlations with numerical features (if target is numerical)
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)
            if numerical_cols:
                correlations = self.data[numerical_cols + [target_col]].corr()[target_col].drop(target_col)
                top_corr = correlations.abs().sort_values(ascending=False).head(5)
                
                print(f"\nTop Correlations with {target_col}:")
                for feature, corr in top_corr.items():
                    print(f"  {feature}: {corr:.3f}")
    
    def suggest_preprocessing_steps(self):
        """Suggest preprocessing steps based on data analysis."""
        if self.data is None:
            print("Please load data first!")
            return
        
        print("\n" + "="*60)
        print("PREPROCESSING RECOMMENDATIONS")
        print("="*60)
        
        suggestions = []
        
        # Missing data
        missing_data = self.data.isnull().sum()
        columns_with_missing = missing_data[missing_data > 0]
        if len(columns_with_missing) > 0:
            suggestions.append(" Handle missing data:")
            for col, count in columns_with_missing.items():
                pct = (count / len(self.data)) * 100
                if pct > 50:
                    suggestions.append(f"   - Consider dropping '{col}' ({pct:.1f}% missing)")
                elif self.data[col].dtype in ['object', 'category']:
                    suggestions.append(f"   - Impute '{col}' with mode/most frequent")
                else:
                    suggestions.append(f"   - Impute '{col}' with median/mean")
        
        # Duplicates
        if self.data.duplicated().sum() > 0:
            suggestions.append(f" Remove {self.data.duplicated().sum()} duplicate rows")
        
        # Outliers
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        outlier_cols = []
        for col in numerical_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.data[(self.data[col] < Q1 - 1.5*IQR) | (self.data[col] > Q3 + 1.5*IQR)]
            if len(outliers) > 0:
                outlier_cols.append((col, len(outliers)))
        
        if outlier_cols:
            suggestions.append(" Handle outliers:")
            for col, count in outlier_cols:
                pct = (count / len(self.data)) * 100
                suggestions.append(f"   - '{col}' has {count} outliers ({pct:.1f}%)")
        
        # Skewed data
        skewed_cols = []
        for col in numerical_cols:
            skewness = abs(self.data[col].skew())
            if skewness > 2:
                skewed_cols.append((col, skewness))
        
        if skewed_cols:
            suggestions.append(" Handle skewed distributions:")
            for col, skew in skewed_cols:
                suggestions.append(f"   - Apply log/box-cox transformation to '{col}' (skew: {skew:.2f})")
        
        # Categorical encoding
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            suggestions.append(" Encode categorical variables:")
            for col in categorical_cols:
                unique_count = self.data[col].nunique()
                if unique_count > 10:
                    suggestions.append(f"   - Consider target encoding for '{col}' ({unique_count} categories)")
                else:
                    suggestions.append(f"   - One-hot encode '{col}' ({unique_count} categories)")
        
        # Feature engineering
        if len(numerical_cols) >= 2:
            suggestions.append("- Consider feature engineering:")
            suggestions.append("   - Create interaction features between numerical variables")
            suggestions.append("   - Create polynomial features for non-linear relationships")
        
        # Print suggestions
        if suggestions:
            for suggestion in suggestions:
                print(suggestion)
        else:
            print("Data looks clean! No major preprocessing steps needed.")
    
    def interactive_column_analysis(self):
        """Create interactive widgets for column analysis."""
        if not WIDGETS_AVAILABLE:
            print("Interactive widgets not available. Install ipywidgets.")
            return
        
        if self.data is None:
            print("Please load data first!")
            return
        
        def analyze_column(column_name):
            with output:
                clear_output(wait=True)
                col_data = self.data[column_name]
                
                print(f"Analysis for: {column_name}")
                print("="*40)
                print(f"Data Type: {col_data.dtype}")
                print(f"Non-null Values: {col_data.count()}/{len(col_data)} ({col_data.count()/len(col_data)*100:.1f}%)")
                print(f"Unique Values: {col_data.nunique()}")
                
                if pd.api.types.is_numeric_dtype(col_data):
                    print(f"\nStatistics:")
                    print(f"  Mean: {col_data.mean():.2f}")
                    print(f"  Std: {col_data.std():.2f}")
                    print(f"  Min: {col_data.min():.2f}")
                    print(f"  Max: {col_data.max():.2f}")
                    print(f"  Skewness: {col_data.skew():.2f}")
                    
                    # Simple histogram
                    plt.figure(figsize=(8, 4))
                    plt.hist(col_data.dropna(), bins=20, alpha=0.7, edgecolor='black')
                    plt.title(f'Distribution of {column_name}')
                    plt.xlabel(column_name)
                    plt.ylabel('Frequency')
                    plt.show()
                
                else:
                    print(f"\nTop Values:")
                    top_values = col_data.value_counts().head(10)
                    for value, count in top_values.items():
                        pct = (count / len(col_data)) * 100
                        print(f"  {value}: {count} ({pct:.1f}%)")
                    
                    # Simple bar plot
                    plt.figure(figsize=(10, 4))
                    top_values.plot(kind='bar')
                    plt.title(f'Top Values for {column_name}')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.show()
        
        # Create dropdown widget
        column_dropdown = widgets.Dropdown(
            options=list(self.data.columns),
            value=self.data.columns[0],
            description='Column:',
            style={'description_width': 'initial'}
        )
        
        output = widgets.Output()
        
        # Connect widget to function
        widgets.interact(analyze_column, column_name=column_dropdown)
        
        display(output)

# Initialize the explorer
explorer = InteractiveDataExplorer()



# Example with the obesity dataset
explorer.load_data('/Users/ksonar/Documents/Technical/project-template/data/heart.csv')
explorer.quick_overview()
explorer.generate_comprehensive_profile(target_col='target')
explorer.create_visualizations(target_col='target')
explorer.analyze_target_relationship('target')
explorer.suggest_preprocessing_steps()

# Interactive analysis (if widgets available)
explorer.interactive_column_analysis()
