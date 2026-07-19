"""
Bank Customer Churn Risk Scoring Analysis
==========================================
This module provides comprehensive analysis and visualization of customer churn risk.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

class ChurnRiskAnalyzer:
    """Analyze and predict customer churn risk"""
    
    def __init__(self, csv_file):
        """Initialize the analyzer with CSV data"""
        self.df = pd.read_csv(csv_file)
        self.scaler = StandardScaler()
        self.model = None
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Clean and prepare data for analysis"""
        print("📊 Loading data...")
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nColumns: {list(self.df.columns)}")
        
        # Display basic statistics
        print("\n📈 Data Summary:")
        print(self.df.describe())
        
        # Check for missing values
        print(f"\n❌ Missing values:\n{self.df.isnull().sum()}")
    
    def analyze_risk_distribution(self):
        """Analyze the distribution of risk categories"""
        print("\n🎯 Risk Category Distribution:")
        risk_dist = self.df['Risk_Category'].value_counts()
        print(risk_dist)
        print(f"\nRisk Distribution %:")
        print((risk_dist / len(self.df) * 100).round(2))
        
        return risk_dist
    
    def analyze_churn_metrics(self):
        """Analyze churn-related metrics"""
        print("\n📉 Churn Analysis:")
        churn_rate = self.df['Exited'].mean() * 100
        print(f"Overall Churn Rate: {churn_rate:.2f}%")
        
        # Churn by risk category
        print("\nChurn Rate by Risk Category:")
        churn_by_risk = self.df.groupby('Risk_Category')['Exited'].agg(['sum', 'count', 'mean'])
        churn_by_risk.columns = ['Churned', 'Total', 'Churn_Rate']
        churn_by_risk['Churn_Rate'] = (churn_by_risk['Churn_Rate'] * 100).round(2)
        print(churn_by_risk)
        
        return churn_by_risk
    
    def analyze_demographics(self):
        """Analyze demographic patterns"""
        print("\n👥 Demographic Analysis:")
        
        # Age analysis
        print(f"\nAge Statistics:")
        print(f"  Min: {self.df['Age'].min()}")
        print(f"  Max: {self.df['Age'].max()}")
        print(f"  Mean: {self.df['Age'].mean():.2f}")
        print(f"  Median: {self.df['Age'].median():.2f}")
        
        # Gender distribution
        print(f"\nGender Distribution:")
        gender_dist = self.df['Gender'].value_counts()
        print(f"  Female (0): {gender_dist.get(0, 0)}")
        print(f"  Male (1): {gender_dist.get(1, 0)}")
        
        # Geography
        print(f"\nGeography Distribution:")
        print(f"  France: {len(self.df[(self.df['Geography_Germany'] == False) & (self.df['Geography_Spain'] == False)])}")
        print(f"  Germany: {self.df['Geography_Germany'].sum()}")
        print(f"  Spain: {self.df['Geography_Spain'].sum()}")
    
    def analyze_financial_metrics(self):
        """Analyze financial characteristics"""
        print("\n💰 Financial Analysis:")
        
        print(f"\nCredit Score:")
        print(f"  Mean: {self.df['CreditScore'].mean():.2f}")
        print(f"  Median: {self.df['CreditScore'].median():.2f}")
        print(f"  Range: {self.df['CreditScore'].min()} - {self.df['CreditScore'].max()}")
        
        print(f"\nBalance:")
        print(f"  Mean: ${self.df['Balance'].mean():,.2f}")
        print(f"  Median: ${self.df['Balance'].median():,.2f}")
        print(f"  Customers with 0 Balance: {(self.df['Balance'] == 0).sum()}")
        
        print(f"\nEstimated Salary:")
        print(f"  Mean: ${self.df['EstimatedSalary'].mean():,.2f}")
        print(f"  Median: ${self.df['EstimatedSalary'].median():,.2f}")
        
        print(f"\nNumber of Products:")
        print(self.df['NumOfProducts'].value_counts().sort_index())
    
    def risk_probability_analysis(self):
        """Analyze risk probability distribution"""
        print("\n⚠️ Risk Probability Analysis:")
        
        print(f"Risk Probability Statistics:")
        print(f"  Min: {self.df['Risk_Probability'].min():.4f}")
        print(f"  Max: {self.df['Risk_Probability'].max():.4f}")
        print(f"  Mean: {self.df['Risk_Probability'].mean():.4f}")
        print(f"  Median: {self.df['Risk_Probability'].median():.4f}")
        
        # Correlation with actual churn
        correlation = self.df['Risk_Probability'].corr(self.df['Exited'])
        print(f"\nCorrelation with Actual Churn: {correlation:.4f}")
    
    def build_predictive_model(self):
        """Build a random forest model to predict churn"""
        print("\n🤖 Building Predictive Model...")
        
        # Prepare features
        feature_cols = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 
                       'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
                       'EstimatedSalary', 'Geography_Germany', 'Geography_Spain',
                       'Risk_Probability', 'Risk_Score']
        
        X = self.df[feature_cols]
        y = self.df['Exited']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        print(f"\n✅ Model Performance:")
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"\n📊 Top 10 Important Features:")
        print(feature_importance.head(10))
        
        return feature_importance
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*70)
        print("BANK CUSTOMER CHURN RISK ANALYSIS REPORT")
        print("="*70)
        
        self.analyze_risk_distribution()
        self.analyze_churn_metrics()
        self.analyze_demographics()
        self.analyze_financial_metrics()
        self.risk_probability_analysis()
        
        print("\n" + "="*70)


# ============================================================================
# 2. VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_risk_distribution(df):
    """Create visualization of risk distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Risk category counts
    risk_counts = df['Risk_Category'].value_counts()
    axes[0].bar(risk_counts.index, risk_counts.values, color=['green', 'orange', 'red'])
    axes[0].set_title('Risk Category Distribution')
    axes[0].set_ylabel('Count')
    
    # Risk score distribution
    axes[1].hist(df['Risk_Score'], bins=50, color='steelblue', edgecolor='black')
    axes[1].set_title('Risk Score Distribution')
    axes[1].set_xlabel('Risk Score')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('risk_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: risk_distribution.png")
    plt.close()


def visualize_churn_analysis(df):
    """Visualize churn patterns"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Churn by risk category
    churn_by_risk = df.groupby('Risk_Category')['Exited'].mean() * 100
    axes[0, 0].bar(churn_by_risk.index, churn_by_risk.values, color=['green', 'orange', 'red'])
    axes[0, 0].set_title('Churn Rate by Risk Category')
    axes[0, 0].set_ylabel('Churn Rate (%)')
    
    # Age vs Churn
    axes[0, 1].scatter(df['Age'], df['Exited'], alpha=0.3)
    axes[0, 1].set_title('Age vs Churn')
    axes[0, 1].set_xlabel('Age')
    axes[0, 1].set_ylabel('Exited (0/1)')
    
    # Tenure vs Churn
    axes[1, 0].scatter(df['Tenure'], df['Exited'], alpha=0.3, color='green')
    axes[1, 0].set_title('Tenure vs Churn')
    axes[1, 0].set_xlabel('Tenure (Years)')
    axes[1, 0].set_ylabel('Exited (0/1)')
    
    # Balance vs Risk Score
    axes[1, 1].scatter(df['Balance'], df['Risk_Score'], alpha=0.3, color='red')
    axes[1, 1].set_title('Balance vs Risk Score')
    axes[1, 1].set_xlabel('Balance ($)')
    axes[1, 1].set_ylabel('Risk Score')
    
    plt.tight_layout()
    plt.savefig('churn_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: churn_analysis.png")
    plt.close()


# ============================================================================
# 3. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Initialize analyzer
    analyzer = ChurnRiskAnalyzer('Bank-Customer-Churn-Risk-Scoring')
    
    # Generate comprehensive report
    analyzer.generate_report()
    
    # Build predictive model
    analyzer.build_predictive_model()
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    visualize_risk_distribution(analyzer.df)
    visualize_churn_analysis(analyzer.df)
    
    print("\n✨ Analysis complete!")
