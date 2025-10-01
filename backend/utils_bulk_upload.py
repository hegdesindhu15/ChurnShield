import os
import pandas as pd
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from analyzer import perform_analysis
import logging
from datetime import datetime
import json
from model_loader import *
from model_utils import *
import shap, joblib

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define required columns
REQUIRED_COLUMNS = [
    'Customer_ID', 'Age', 'Gender', 'Marital_Status', 'Region', 'Education_Level',
    'Employment_Status', 'Policy_Type', 'Policy_Tenure', 'Policy_Renewal_Count',
    'Total_Claims_Filed', 'Claim_Amount_Total', 'Missed_Payments', 'Family_Members_Insured',
    'Policy_Cancellation_History', 'Mode_of_Communication', 'Payment_Mode',
    'Customer_Complaints', 'Complaint_Resolution_Time', 'Customer_Support_Calls',
    'Policy_Start_Date', 'Policy_End_Date', 'Last_Claim_Date', 'Last_Payment_Date',
    'Vehicle_Age', 'Vehicle_Type', 'Premium_Amount', 'Churn'
]

# List of columns not suitable for grouping
NON_GROUPING_COLUMNS = [
    'Customer_ID', 'Churn', 'Policy_Start_Date', 'Policy_End_Date', 
    'Last_Claim_Date', 'Last_Payment_Date', 'Claim_Amount_Total', 
    'Customer_Complaints', 'Complaint_Resolution_Time', 'Customer_Support_Calls'
]

def get_grouping_columns(df):
    """Return a list of columns suitable for grouping."""
    return [col for col in df.columns if col not in NON_GROUPING_COLUMNS]

def save_uploaded_file(file, upload_folder='frontend/static/uploads'):
    os.makedirs(upload_folder, exist_ok=True)
    filename = secure_filename(file.filename)
    file_path = os.path.join(upload_folder, filename)
    file.save(file_path)
    logger.debug(f"File saved to: {file_path}")
    return file_path

def read_csv_file(file_path):
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            return None, "Empty CSV file"
        
        # Transform date columns to a consistent format (DD-MM-YYYY)
        date_cols = ['Policy_Start_Date', 'Policy_End_Date', 'Last_Claim_Date', 'Last_Payment_Date']
        for col in date_cols:
            if col in df.columns:
                # Try parsing with multiple formats
                try:
                    # First try YYYY-MM-DD
                    df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')
                except ValueError:
                    # Then try DD-MM-YYYY
                    df[col] = pd.to_datetime(df[col], format='%d-%m-%Y', errors='coerce')
                # If still failing, try a more flexible parsing
                if df[col].isna().all():
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                # Convert to DD-MM-YYYY string format
                df[col] = df[col].dt.strftime('%d-%m-%Y').fillna('')
                logger.debug(f"Transformed {col} to DD-MM-YYYY format")

        # Validate required columns
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            return None, f"Missing columns: {', '.join(missing_cols)}"
        
        numeric_cols = [
            'Age', 'Policy_Tenure', 'Policy_Renewal_Count', 'Total_Claims_Filed',
            'Claim_Amount_Total', 'Missed_Payments', 'Family_Members_Insured',
            'Policy_Cancellation_History', 'Customer_Complaints', 'Complaint_Resolution_Time',
            'Customer_Support_Calls', 'Vehicle_Age', 'Premium_Amount'
        ]
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                return None, f"{col} must be numeric"
        
        if not df['Age'].between(18, 100).all():
            return None, "Age must be between 18 and 100"
        if not df['Policy_Tenure'].ge(0).all():
            return None, "Policy_Tenure must be non-negative"
        if not df['Policy_Renewal_Count'].ge(0).all():
            return None, "Policy_Renewal_Count must be non-negative"
        if not df['Total_Claims_Filed'].ge(0).all():
            return None, "Total_Claims_Filed must be non-negative"
        if not df['Claim_Amount_Total'].ge(0).all():
            return None, "Claim_Amount_Total must be non-negative"
        if not df['Missed_Payments'].between(0, 10).all():
            return None, "Missed_Payments must be between 0 and 10"
        if not df['Family_Members_Insured'].ge(0).all():
            return None, "Family_Members_Insured must be non-negative"
        if not df['Policy_Cancellation_History'].ge(0).all():
            return None, "Policy_Cancellation_History must be non-negative"
        if not df['Customer_Complaints'].ge(0).all():
            return None, "Customer_Complaints must be non-negative"
        if not df['Complaint_Resolution_Time'].ge(0).all():
            return None, "Complaint_Resolution_Time must be non-negative"
        if not df['Customer_Support_Calls'].ge(0).all():
            return None, "Customer_Support_Calls must be non-negative"
        if not df['Vehicle_Age'].ge(0).all():
            return None, "Vehicle_Age must be non-negative"
        if not (df['Premium_Amount'] > 100).all():
            return None, "Premium_Amount must be greater than 100"
        
        if not df['Churn'].isin([0, 1]).all():
            return None, "Churn must be 0 or 1"
        
        categorical_cols = [
            'Customer_ID', 'Gender', 'Marital_Status', 'Region', 'Education_Level',
            'Employment_Status', 'Policy_Type', 'Mode_of_Communication', 'Payment_Mode',
            'Vehicle_Type'
        ]
        for col in categorical_cols:
            if not df[col].apply(lambda x: isinstance(x, str) and x.strip() != '').all():
                return None, f"{col} must be a non-empty string"
        
        for date_col in date_cols:
            if df[date_col].eq('').all():
                continue
            try:
                pd.to_datetime(df[date_col], format='%d-%m-%Y', errors='raise')
            except:
                return None, f"{date_col} must be in DD-MM-YYYY format"
        
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
        for date_col in date_cols:
            df[date_col] = df[date_col].replace('', pd.to_datetime('01-01-2000', format='%d-%m-%Y'))
        
        logger.debug(f"CSV columns: {list(df.columns)}")
        return df, None
    except Exception as e:
        logger.error(f"Error reading CSV: {str(e)}")
        return None, f"Error reading CSV: {str(e)}"


def generate_pattern_plots(df, project_name, output_dir='frontend/static/plots'):
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import json
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}
    plot_data = {}

    try:
        # Plot 1: Churn Rate by Age Group
        if 'Age' in df.columns and 'Churn' in df.columns:
            plt.figure(figsize=(10, 6))
            age_bins = [18, 25, 35, 45, 55, 65, 100]
            df['Age_Group'] = pd.cut(df['Age'], bins=age_bins)
            age_churn = df.groupby('Age_Group', observed=False)['Churn'].mean()
            sns.barplot(x=age_churn.index.astype(str), y=age_churn.values, palette="viridis")
            plt.xticks(rotation=45)
            plt.title('Churn Rate by Age Group')
            plt.ylabel('Churn Rate')
            plt.xlabel('Age Group')
            plt.tight_layout()
            age_plot_path = os.path.join(output_dir, f'{project_name}_churn_age_group.png')
            plt.savefig(age_plot_path)
            plt.close()
            plot_paths['Churn Rate by Age Group'] = age_plot_path
            plot_data['Churn Rate by Age Group'] = {
                'type': 'bar',
                'labels': age_churn.index.astype(str).tolist(),
                'data': age_churn.values.tolist(),
                'backgroundColor': sns.color_palette("viridis", len(age_churn)).as_hex(),
                'section': 'Churn Overview'
            }

        # Plot 2: Policy Tenure Distribution
        if 'Policy_Tenure' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df['Policy_Tenure'], bins=20, kde=True, color='skyblue')
            plt.title('Policy Tenure Distribution')
            plt.xlabel('Policy Tenure (Years)')
            plt.ylabel('Count')
            plt.tight_layout()
            tenure_plot_path = os.path.join(output_dir, f'{project_name}_policy_tenure_distribution.png')
            plt.savefig(tenure_plot_path)
            plt.close()
            hist_data = pd.cut(df['Policy_Tenure'], bins=20).value_counts().sort_index()
            plot_paths['Policy Tenure Distribution'] = tenure_plot_path
            plot_data['Policy Tenure Distribution'] = {
                'type': 'bar',
                'labels': [str(interval) for interval in hist_data.index],
                'data': hist_data.values.tolist(),
                'backgroundColor': 'skyblue',
                'section': 'Trends and Patterns'
            }

        # Plot 3: Missed Payments vs Churn
        if 'Missed_Payments' in df.columns and 'Churn' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Churn', y='Missed_Payments', data=df, palette="Set2")
            plt.title('Missed Payments vs Churn')
            plt.xlabel('Churn Status')
            plt.ylabel('Number of Missed Payments')
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'{project_name}_missed_payments_vs_churn.png')
            plt.savefig(plot_path)
            plt.close()
            churned = df[df['Churn'] == 1]['Missed_Payments']
            not_churned = df[df['Churn'] == 0]['Missed_Payments']
            plot_paths['Missed Payments vs Churn'] = plot_path
            plot_data['Missed Payments vs Churn'] = {
                'type': 'bar',
                'labels': ['Not Churned', 'Churned'],
                'data': [not_churned.mean(), churned.mean()],
                'backgroundColor': sns.color_palette("Set2", 2).as_hex(),
                'section': 'Risk Groups'
            }

        # Plot 4: Churn Rate by Policy Type
        if 'Policy_Type' in df.columns and 'Churn' in df.columns:
            churn_by_policy = df.groupby('Policy_Type')['Churn'].mean()
            plt.figure(figsize=(10, 6))
            sns.barplot(x=churn_by_policy.index, y=churn_by_policy.values, palette='coolwarm')
            plt.title('Churn Rate by Policy Type')
            plt.ylabel('Churn Rate')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'{project_name}_churn_by_policy_type.png')
            plt.savefig(plot_path)
            plt.close()
            plot_paths['Churn Rate by Policy Type'] = plot_path
            plot_data['Churn Rate by Policy Type'] = {
                'type': 'bar',
                'labels': churn_by_policy.index.tolist(),
                'data': churn_by_policy.values.tolist(),
                'backgroundColor': sns.color_palette("coolwarm", len(churn_by_policy)).as_hex(),
                'section': 'Policy Info'
            }

    except Exception as e:
        logger.error(f"Error in generate_pattern_plots: {str(e)}")

    # Save plot data as JSON
    data_path = os.path.join(output_dir, f'{project_name}_plot_data.json')
    with open(data_path, 'w') as f:
        json.dump(plot_data, f)

    plot_paths['plot_data'] = f'plots/{os.path.basename(data_path)}'
    return plot_paths

def generate_grouping_plot(df, project_name, group_features, output_dir='frontend/static/plots'):
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}
    plot_data = {}
    
    try:
        if not isinstance(group_features, list):
            group_features = [group_features]
        
        if all(f in df.columns for f in group_features) and 'Churn' in df.columns:
            df['Churn_Numeric'] = df['Churn']
            
            for feature in group_features:
                if pd.api.types.is_numeric_dtype(df[feature]):
                    df[f"{feature}_binned"] = pd.qcut(df[feature], q=5, duplicates='drop')
                    group_features[group_features.index(feature)] = f"{feature}_binned"
            
            group_cols = group_features
            churn_summary = df.groupby(group_cols, observed=False)['Churn_Numeric'].mean().reset_index()
            churn_summary.columns = group_cols + ['Churn_Rate']
            
            plt.figure(figsize=(12, 6))
            if len(group_cols) == 1:
                sns.barplot(x=group_cols[0], y='Churn_Rate', data=churn_summary, hue=group_cols[0], palette="viridis", legend=False)
                plt.xticks(rotation=45)
            else:
                pivot_data = churn_summary.pivot(index=group_cols[0], columns=group_cols[1], values='Churn_Rate')
                pivot_data.plot(kind='bar', stacked=False, figsize=(12, 6), colormap='viridis')
                plt.legend(title=group_cols[1])
            plt.title(f'Churn Rate by {", ".join([f.replace("_binned", "") for f in group_cols])}')
            plt.ylabel('Churn Rate')
            plt.xlabel(group_cols[0].replace("_binned", ""))
            plt.tight_layout()
            
            plot_path = os.path.join(output_dir, f'{project_name}_churn_by_{"_".join([f.lower().replace("_binned", "") for f in group_cols])}.png')
            plt.savefig(plot_path)
            plt.close()
            plot_paths[f'Churn Rate by {", ".join([f.replace("_binned", "") for f in group_cols])}'] = plot_path
            plot_data[f'Churn Rate by {", ".join([f.replace("_binned", "") for f in group_cols])}'] = {
                'type': 'bar',
                'labels': churn_summary[group_cols[0]].astype(str).tolist(),
                'data': churn_summary['Churn_Rate'].tolist(),
                'backgroundColor': sns.color_palette("viridis", len(churn_summary)).as_hex()
            }
            logger.debug(f"Added Churn Rate by {', '.join(group_cols)}: {plot_path}")
        else:
            logger.warning(f"Missing 'Churn' or group features: {group_features}")
    except Exception as e:
        logger.error(f"Error generating grouping plot: {str(e)}")
    
    relative_paths = {title: f'plots/{os.path.basename(full_path)}' for title, full_path in plot_paths.items()}
    # Save plot data as JSON
    data_path = os.path.join(output_dir, f'{project_name}_grouping_plot_data.json')
    with open(data_path, 'w') as f:
        json.dump(plot_data, f)
    relative_paths['grouping_plot_data'] = f'plots/{os.path.basename(data_path)}'
    logger.debug(f"Grouping analysis relative_paths: {relative_paths}")
    return relative_paths

def generate_time_based_plots(df, project_name, group_features, output_dir='frontend/static/plots'):
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}
    plot_data = {}

    try:
        if not isinstance(group_features, list):
            group_features = [group_features] if group_features else []
        group_feature = group_features[0] if group_features else None

        # Ensure 'Policy_Start_Date' and 'Churn' are present
        if 'Policy_Start_Date' not in df.columns or 'Churn' not in df.columns:
            logger.error("Missing required columns 'Policy_Start_Date' or 'Churn' for time-based plots")
            return plot_paths

        # Parse 'Policy_Start_Date' with error handling
        df = df.copy()  # Avoid modifying the original DataFrame
        df['Policy_Start_Date'] = pd.to_datetime(df['Policy_Start_Date'], format='%d-%m-%Y', errors='coerce')
        if df['Policy_Start_Date'].isna().all():
            logger.error("All 'Policy_Start_Date' values are invalid or cannot be parsed as DD-MM-YYYY")
            return plot_paths

        current_date = pd.to_datetime('2025-06-03', format='%Y-%m-%d')  # Updated to current date
        df['Policy_Start_Date'] = df.apply(
            lambda row: row['Policy_Start_Date'] if pd.notna(row['Policy_Start_Date'])
            else current_date - pd.Timedelta(days=int(row['Policy_Tenure'] * 365.25)),
            axis=1
        )

        # Extract YearMonth for grouping
        df['YearMonth'] = df['Policy_Start_Date'].dt.to_period('M')
        if df['YearMonth'].isna().any():
            logger.warning("Some 'YearMonth' values are NaN after conversion")

        # Plot 1: Churn Rate Over Time (Line Plot)
        churn_by_month = df.groupby('YearMonth')['Churn'].mean().reset_index()
        if churn_by_month.empty:
            logger.warning("No data available for 'Monthly Churn Rate Over Time' plot")
        else:
            churn_by_month['YearMonth'] = churn_by_month['YearMonth'].astype(str)
            plt.figure(figsize=(12, 6))
            sns.lineplot(x='YearMonth', y='Churn', data=churn_by_month, marker='o')
            plt.title('Monthly Churn Rate Over Time')
            plt.xticks(ticks=range(0, len(churn_by_month), 3), labels=churn_by_month['YearMonth'][::3], rotation=45, ha='right', fontsize=10)        
            plt.ylabel('Churn Rate')
            plt.xlabel('Year-Month')
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'{project_name}_churn_over_time.png')
            plt.savefig(plot_path)
            plt.close()
            plot_paths['Monthly Churn Rate Over Time'] = plot_path
            plot_data['Monthly Churn Rate Over Time'] = {
                'type': 'line',
                'labels': churn_by_month['YearMonth'].tolist(),
                'data': churn_by_month['Churn'].tolist(),
                'borderColor': 'blue',
                'fill': False
            }
            logger.debug(f"Added Churn Rate Over Time: {plot_path}")

        # Plot 2: Monthly Churn Rate by Group Feature (Stacked Bar Plot, if group feature is provided)
        if group_feature and group_feature in df.columns:
            churn_by_month_group = df.groupby(['YearMonth', group_feature])['Churn'].mean().reset_index()
            if churn_by_month_group.empty:
                logger.warning(f"No data available for 'Monthly Churn Rate by {group_feature}' plot")
            else:
                churn_by_month_group['YearMonth'] = churn_by_month_group['YearMonth'].astype(str)
                plt.figure(figsize=(12, 6))
                sns.barplot(x='YearMonth', y='Churn', hue=group_feature, data=churn_by_month_group)
                plt.title(f'Monthly Churn Rate by {group_feature}')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.ylabel('Churn Rate')
                plt.xlabel('Year-Month')
                plt.tight_layout()
                plot_path = os.path.join(output_dir, f'{project_name}_churn_by_{group_feature.lower()}_over_time.png')
                plt.savefig(plot_path)
                plt.close()
                plot_paths[f'Monthly Churn Rate by {group_feature}'] = plot_path
                # Prepare data for potential Chart.js rendering
                unique_groups = churn_by_month_group[group_feature].unique()
                datasets = []
                for group in unique_groups:
                    group_data = churn_by_month_group[churn_by_month_group[group_feature] == group]
                    datasets.append({
                        'label': str(group),
                        'data': group_data['Churn'].tolist(),
                        'backgroundColor': sns.color_palette("viridis", len(unique_groups)).as_hex()[list(unique_groups).index(group)]
                    })
                plot_data[f'Monthly Churn Rate by {group_feature}'] = {
                    'type': 'bar',
                    'labels': churn_by_month_group['YearMonth'].unique().tolist(),
                    'datasets': datasets
                }
                logger.debug(f"Added Monthly Churn Rate by {group_feature}: {plot_path}")

    except Exception as e:
        logger.error(f"Error in generate_time_based_plots: {str(e)}")

    # Save plot data as JSON
    data_path = os.path.join(output_dir, f'{project_name}_time_plot_data.json')
    with open(data_path, 'w') as f:
        json.dump(plot_data, f)

    plot_paths['time_plot_data'] = f'plots/{os.path.basename(data_path)}'
    return plot_paths

def generate_summary_insights(df, output_dir, project_name):
    os.makedirs(output_dir, exist_ok=True)
    insights = {}
    summary = {}
    plot_paths = {}

    total_customers = len(df)
    total_churned = df['Churn'].sum()
    churn_rate = (total_churned / total_customers) * 100 if total_customers > 0 else 0

    summary['total_customers'] = total_customers
    summary['total_churned'] = total_churned
    summary['churn_rate'] = round(churn_rate, 2)

    plt.figure(figsize=(8, 8))
    labels = ['Retained', 'Churned']
    sizes = [total_customers - total_churned, total_churned]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'])
    plt.title('Churn Rate Overview')
    pie_path = os.path.join(output_dir, f'{project_name}_churn_pie.png')
    plt.savefig(pie_path)
    plt.close()

    plot_paths['Churn Rate Overview'] = f'plots/{os.path.basename(pie_path)}'

    insights['summary'] = summary
    insights['plot'] = plot_paths

    return insights

def generate_risk_group_analysis(df, project_name, output_dir='frontend/static/plots'):
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}
    plot_data = {}

    try:
        if 'Missed_Payments' in df.columns and 'Churn' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Churn', y='Missed_Payments', data=df, palette="Set2")
            plt.title('Missed Payments vs Churn')
            plt.xlabel('Churn Status')
            plt.ylabel('Number of Missed Payments')
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'{project_name}_missed_payments_vs_churn.png')
            plt.savefig(plot_path)
            plt.close()
            churned = df[df['Churn'] == 1]['Missed_Payments']
            not_churned = df[df['Churn'] == 0]['Missed_Payments']
            plot_paths['Missed Payments vs Churn'] = plot_path
            plot_data['Missed Payments vs Churn'] = {
                'type': 'bar',
                'labels': ['Not Churned', 'Churned'],
                'data': [not_churned.mean(), churned.mean()],
                'backgroundColor': sns.color_palette("Set2", 2).as_hex()
            }
            logger.debug(f"Added Missed Payments vs Churn: {plot_path}")

        if 'Customer_Support_Calls' in df.columns and 'Churn' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Churn', y='Customer_Support_Calls', data=df, palette="Set3")
            plt.title('Customer Support Calls vs Churn')
            plt.xlabel('Churn Status')
            plt.ylabel('Number of Customer Support Calls')
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'{project_name}_support_calls_vs_churn.png')
            plt.savefig(plot_path)
            plt.close()
            churned = df[df['Churn'] == 1]['Customer_Support_Calls']
            not_churned = df[df['Churn'] == 0]['Customer_Support_Calls']
            plot_paths['Customer Support Calls vs Churn'] = plot_path
            plot_data['Customer Support Calls vs Churn'] = {
                'type': 'bar',
                'labels': ['Not Churned', 'Churned'],
                'data': [not_churned.mean(), churned.mean()],
                'backgroundColor': sns.color_palette("Set3", 2).as_hex()
            }
            logger.debug(f"Added Customer Support Calls vs Churn: {plot_path}")

    except Exception as e:
        logger.error(f"Error in generate_risk_group_analysis: {str(e)}")

    # Save plot data as JSON
    data_path = os.path.join(output_dir, f'{project_name}_risk_plot_data.json')
    with open(data_path, 'w') as f:
        json.dump(plot_data, f)

    plot_paths['risk_plot_data'] = f'plots/{os.path.basename(data_path)}'
    return plot_paths

# backend/analyzer.py


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from model_loader import prediction_model, scaler, encoder, explainer, feature_names, anchor_date
import logging

logger = logging.getLogger(__name__)

def generate_top_churn_drivers_plot(df, project_name, output_dir='frontend/static/plots'):
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}

    try:
        # Ensure date columns are datetime
        df['Policy_Start_Date'] = pd.to_datetime(df['Policy_Start_Date'], errors='coerce')
        df['Policy_End_Date'] = pd.to_datetime(df['Policy_End_Date'], errors='coerce')
        df['Last_Claim_Date'] = pd.to_datetime(df['Last_Claim_Date'], errors='coerce')
        df['Last_Payment_Date'] = pd.to_datetime(df['Last_Payment_Date'], errors='coerce')

        # Recreate date features exactly as used during training
        df['Policy_Duration'] = (df['Policy_End_Date'] - df['Policy_Start_Date']).dt.days.fillna(0)
        df['Policy_Age'] = ((anchor_date - df['Policy_Start_Date']).dt.days / 365.25).fillna(0)
        df['Time_Since_Last_Claim'] = (anchor_date - df['Last_Claim_Date']).dt.days.fillna(0)
        df['Time_Since_Last_Payment'] = (anchor_date - df['Last_Payment_Date']).dt.days.fillna(0)

        # Numerical features in same order as training
        num_cols = [
            'Age', 'Policy_Tenure', 'Policy_Renewal_Count', 'Total_Claims_Filed', 
            'Claim_Amount_Total', 'Missed_Payments', 'Family_Members_Insured', 
            'Policy_Cancellation_History', 'Customer_Complaints', 'Complaint_Resolution_Time', 
            'Customer_Support_Calls', 'Vehicle_Age', 'Premium_Amount',
            'Policy_Age', 'Policy_Duration', 'Time_Since_Last_Payment', 'Time_Since_Last_Claim'
        ]
        X_num = df[num_cols].copy()
        X_num_scaled = scaler.transform(X_num)

        # Categorical features (same as training)
        cat_cols = [
            'Gender', 'Marital_Status', 'Region', 'Education_Level', 'Employment_Status',
            'Policy_Type', 'Mode_of_Communication', 'Payment_Mode', 'Vehicle_Type'
        ]
        cat_data = df[cat_cols].fillna('Unknown')
        X_cat = encoder.transform(cat_data)
        if hasattr(X_cat, "toarray"):
            X_cat = X_cat.toarray()

        # Combine
        X_full = np.hstack((X_num_scaled, X_cat))
        X_df = pd.DataFrame(X_full, columns=feature_names)

        # SHAP values
        shap_values = explainer.shap_values(X_df)
        shap_contrib = np.abs(shap_values).mean(axis=0)

        # Top 10 features
        top_idx = np.argsort(shap_contrib)[-10:][::-1]
        top_features = [feature_names[i] for i in top_idx]
        top_importance = shap_contrib[top_idx]

        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(top_features[::-1], top_importance[::-1], color='skyblue')
        plt.xlabel("Mean Absolute SHAP Value")
        plt.title("Top Churn Drivers")
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"{project_name}_top_churn_drivers.png")
        plt.savefig(plot_path)
        plt.close()

        try:
            json_data = {
             "labels": top_features['feature'].tolist(),
             "data": top_features['importance'].round(3).tolist()
             }
            json_path = os.path.join(output_dir, f"{project_name}_top_churn_drivers_plot_data.json")
            with open(json_path, 'w') as f:
                 json.dump(json_data, f)
            logger.info(f"✅ SHAP Top Churn Drivers plot and data saved: {plot_path}, {json_path}")
        except Exception as e:
            logger.warning(f"Failed to save JSON for Top Churn Drivers: {e}")

        plot_paths["Top Churn Drivers"] = plot_path
        logger.info(f"✅ SHAP Top Churn Drivers plot saved: {plot_path}")

    except Exception as e:
        logger.warning(f"[SHAP ERROR] Failed to generate top churn drivers: {str(e)}")

    return plot_paths

