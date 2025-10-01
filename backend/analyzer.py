import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from werkzeug.utils import secure_filename
from flask import flash
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def perform_analysis(df, output_dir, project_name):
    """
    Main function to perform visual analysis on uploaded dataset.
    :param df: pandas DataFrame from uploaded CSV
    :param output_dir: project-specific output directory to save plots
    :param project_name: Project name for naming files
    :return: dict of {plot_title: full_plot_path}
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}

    # --- Plot 1: Churn Rate Pie Chart ---
    if 'Churn' in df.columns:
        try:
            plt.figure(figsize=(6, 6))
            # Handle different Churn formats (Yes/No, 0/1, etc.)
            churn_counts = df['Churn'].value_counts()
            labels = churn_counts.index.tolist()
            # Map labels for display
            label_names = []
            for label in labels:
                if str(label).lower() in ['yes', 'true', '1']:
                    label_names.append('Churned')
                elif str(label).lower() in ['no', 'false', '0']:
                    label_names.append('Not Churned')
                else:
                    label_names.append(str(label))
            plt.pie(churn_counts, labels=label_names, autopct='%1.1f%%', colors=['#66b3ff', '#ff6666'])
            plt.title("Churn Rate Distribution")
            plot_path = os.path.join(output_dir, f"{project_name}_churn_rate_pie.png")
            plt.savefig(plot_path)
            plt.close()
            plot_paths["Churn Rate Distribution"] = plot_path
            logger.debug(f"Added Churn Rate Distribution: {plot_path}")
        except Exception as e:
            logger.error(f"Error generating Churn Rate Distribution: {str(e)}")
    else:
        logger.warning("Churn column not found, skipping pie chart")

    logger.debug(f"plot_paths from perform_analysis: {plot_paths}")
    return plot_paths

def analyze_and_generate_insights(file_path):
    """
    Generate additional insights for uploaded dataset (not used in bulk upload plot grid).
    """
    try:
        df = pd.read_csv(file_path)
        original_columns = df.columns.tolist()

        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

        # Define reference important columns
        important_columns = {
            'age', 'gender', 'marital_status', 'region', 'education_level', 'employment_status',
            'policy_type', 'policy_tenure', 'policy_renewal_count', 'total_claims_filed', 'claim_amount_total',
            'missed_payments', 'family_members_insured', 'policy_cancellation_history', 'mode_of_communication',
            'payment_mode', 'customer_complaints', 'complaint_resolution_time', 'customer_support_calls',
            'vehicle_age', 'vehicle_type', 'premium_amount', 'churn'
        }

        # Find missing or extra columns
        uploaded_columns = set(df.columns)
        missing_columns = important_columns - uploaded_columns
        extra_columns = uploaded_columns - important_columns

        if missing_columns:
            flash(f"Warning: Missing important columns: {', '.join(missing_columns)}. Some analyses may be limited.", "warning")

        # Churn Pie Chart
        if 'churn' in df.columns:
            plt.figure(figsize=(5, 5))
            churn_counts = df['churn'].value_counts()
            plt.pie(churn_counts, labels=churn_counts.index.map({0: 'Not Churned', 1: 'Churned'}), autopct='%1.1f%%', startangle=90, colors=['skyblue', 'salmon'])
            plt.title('Customer Churn Distribution')
            pie_path = os.path.join('static', 'churn_pie_chart.png')
            plt.savefig(pie_path)
            plt.close()
        else:
            flash("Notice: Churn column not found. Churn Rate analysis skipped.", "info")

        # Churn by Gender
        if 'gender' in df.columns and 'churn' in df.columns:
            plt.figure(figsize=(6, 4))
            sns.countplot(data=df, x='gender', hue='churn', palette='Set2')
            plt.title('Churn by Gender')
            plt.xlabel('Gender')
            plt.ylabel('Count')
            plt.legend(['Not Churned', 'Churned'])
            gender_churn_path = os.path.join('static', 'churn_by_gender.png')
            plt.savefig(gender_churn_path)
            plt.close()

        # Churn by Region
        if 'region' in df.columns and 'churn' in df.columns:
            plt.figure(figsize=(7, 5))
            sns.countplot(data=df, x='region', hue='churn', palette='Set3')
            plt.title('Churn by Region')
            plt.xlabel('Region')
            plt.ylabel('Count')
            plt.legend(['Not Churned', 'Churned'])
            region_churn_path = os.path.join('static', 'churn_by_region.png')
            plt.savefig(region_churn_path)
            plt.close()

        # Churn by Vehicle Type
        if 'vehicle_type' in df.columns and 'churn' in df.columns:
            plt.figure(figsize=(7, 5))
            sns.countplot(data=df, x='vehicle_type', hue='churn', palette='coolwarm')
            plt.title('Churn by Vehicle Type')
            plt.xlabel('Vehicle Type')
            plt.ylabel('Count')
            plt.legend(['Not Churned', 'Churned'])
            vehicle_churn_path = os.path.join('static', 'churn_by_vehicle_type.png')
            plt.savefig(vehicle_churn_path)
            plt.close()

        # Churn Rate by Age Distribution
        if 'age' in df.columns and 'churn' in df.columns:
            plt.figure(figsize=(7, 5))
            sns.histplot(data=df, x='age', hue='churn', kde=True, element='step', palette='pastel')
            plt.title('Churn Distribution by Age')
            plt.xlabel('Age')
            plt.ylabel('Count')
            plt.legend(['Not Churned', 'Churned'])
            age_churn_path = os.path.join('static', 'churn_by_age.png')
            plt.savefig(age_churn_path)
            plt.close()

        flash("Analysis Completed Successfully!", "success")
    
    except Exception as e:
        flash(f"Error during analysis: {str(e)}", "danger")