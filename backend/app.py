import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file
from werkzeug.security import generate_password_hash, check_password_hash
import pymysql
from pymysql.cursors import DictCursor
from config import MYSQL_CONFIG
import shap, os
from werkzeug.utils import secure_filename
from analyzer import *
from utils_bulk_upload import *
# from analysis_clustering_logic import *
from report_generator import *
from functools import wraps
from flask import Flask, request, redirect, url_for, flash, render_template, session, send_file
from flask_login import login_required, LoginManager
from werkzeug.utils import secure_filename
import os
import pandas as pd
import logging
from utils_bulk_upload import save_uploaded_file, read_csv_file, generate_pattern_plots, generate_grouping_plot
from report_generator import generate_pdf_report
from flask_session import Session
import pickle
# from ai_explainer import explain_plot_text
from ai_utils import *  # Top of app.py
# from ai_utils import explain_plot_with_gpt
from dotenv import load_dotenv
import pdfkit
from report_generator import *

import openai


app = Flask(__name__, template_folder="../frontend/templates", static_folder="../frontend/static")
app.secret_key = "your_secret_key"

app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = 'flask_session'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True

app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'AIzaSyA_vUu6GkPehCtjznsb-PfcIP5KbtZsclc')  # Ensure a secret key is set
load_dotenv() 

os.makedirs('flask_session', exist_ok=True)
Session(app)

# Define upload and plot directories
UPLOAD_FOLDER = 'frontend/static/uploads'
PLOT_FOLDER = 'frontend/static/plots'
REPORT_FOLDER = 'frontend/static/reports'  # Added REPORT_FOLDER
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PLOT_FOLDER'] = PLOT_FOLDER
app.config['REPORT_FOLDER'] = REPORT_FOLDER  # Added to app config
# PDF_CONFIG = pdfkit.configuration(
#     wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
# )

# Load artifacts
print("Loading artifacts from models/...")
try:
    prediction_model = joblib.load('models/churn_model_xgb.pkl')
    scaler = joblib.load('models/scaler.pkl')
    encoder = joblib.load('models/encoder.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    anchor_date = joblib.load('models/anchor_date.pkl')
except Exception as e:
    print(f"Error loading artifacts: {e}")
    raise

try:
    explainer = shap.TreeExplainer(prediction_model)
    print("SHAP explainer loaded successfully!")
except Exception as e:
    print(f"Error initializing SHAP explainer: {e}")
    raise

# Database setup
def get_db_connection():
    conn = pymysql.connect(
        host=MYSQL_CONFIG["host"],
        user=MYSQL_CONFIG["user"],
        password=MYSQL_CONFIG["password"],
        database=MYSQL_CONFIG["database"],
        port=MYSQL_CONFIG["port"],
        cursorclass=DictCursor
    )
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            email VARCHAR(255) NOT NULL UNIQUE,
            password_hash VARCHAR(255) NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS single_prediction_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_email VARCHAR(255) NOT NULL,
            customer_id VARCHAR(255) NOT NULL,
            prediction VARCHAR(50),
            probability FLOAT,
            analysis_date VARCHAR(50),
            report_path VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_email) REFERENCES users(email),
            UNIQUE (customer_id, analysis_date)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bulk_analysis_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_email VARCHAR(255) NOT NULL,
            project_name VARCHAR(255) NOT NULL,
            file_name VARCHAR(255),
            analysis_date VARCHAR(50),
            report_path VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_email) REFERENCES users(email),
            UNIQUE (project_name, analysis_date)
        )
    ''')
    cursor.execute('''
        INSERT IGNORE INTO users (email, password_hash)
        VALUES (%s, %s)
    ''', ('admin@example.com', generate_password_hash('admin123')))
    conn.commit()
    conn.close()

init_db()

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            if request.path.startswith('/predict') or request.path == '/generate_retention_strategy':
                return jsonify({'error': 'Login required'}), 401
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        print(f"Login attempt: {email}")
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        conn.close()
        if user and check_password_hash(user["password_hash"], password):
            session["user"] = user["email"]
            flash("Logged in successfully!", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid email or password", "error")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        password_hash = generate_password_hash(password)
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (email, password_hash) VALUES (%s, %s)", (email, password_hash))
            conn.commit()
            flash("Registration successful. Please log in.", "success")
            return redirect(url_for("login"))
        except pymysql.err.IntegrityError:
            flash("Email already exists. Please use a different email.", "error")
        finally:
            conn.close()
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out successfully.", "success")
    return redirect(url_for("login"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/single-prediction', methods=['GET', 'POST'])
@login_required
def single_prediction():
    return render_template("single_prediction.html")

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        data = request.json
        print(f"Received data: {data}")

        # Numerical features
        num_cols = ['Age', 'Policy_Tenure', 'Policy_Renewal_Count', 'Total_Claims_Filed', 
                    'Claim_Amount_Total', 'Missed_Payments', 'Family_Members_Insured', 
                    'Policy_Cancellation_History', 'Customer_Complaints', 
                    'Complaint_Resolution_Time', 'Customer_Support_Calls', 'Vehicle_Age', 
                    'Premium_Amount']
        num_data = [float(data[col]) for col in num_cols]

        # Date-derived features
        policy_start = datetime.strptime(data['Policy_Start_Date'], '%Y-%m-%d')
        policy_end = datetime.strptime(data['Policy_End_Date'], '%Y-%m-%d')
        last_payment = datetime.strptime(data['Last_Payment_Date'], '%Y-%m-%d')
        last_claim = datetime.strptime(data['Last_Claim_Date'], '%Y-%m-%d')
        date_features = [
            (anchor_date - policy_start).days / 365.25,
            (policy_end - policy_start).days,
            (anchor_date - last_payment).days,
            (anchor_date - last_claim).days
        ]

        num_data_full = num_data + date_features
        num_scaled = scaler.transform([num_data_full])

        # Categorical features
        cat_cols = ['Gender', 'Marital_Status', 'Region', 'Education_Level', 'Employment_Status',
                    'Policy_Type', 'Mode_of_Communication', 'Payment_Mode', 'Vehicle_Type']
        cat_data = [[data[col] for col in cat_cols]]
        cat_encoded_raw = encoder.transform(cat_data)

        if hasattr(cat_encoded_raw, 'toarray'):
            cat_encoded = cat_encoded_raw.toarray()
        else:
            cat_encoded = cat_encoded_raw

        # Combine features
        features = np.hstack((num_scaled, cat_encoded))
        feature_df = pd.DataFrame(features, columns=feature_names)
        features_ordered = feature_df[feature_names].values

        # Predict
        prediction = prediction_model.predict(features_ordered)[0]
        probability = prediction_model.predict_proba(features_ordered)[0][1]
        print(f"Prediction: {prediction}, Probability: {probability}")

        return jsonify({
            'prediction': 'Churn' if prediction == 1 else 'No Churn',
            'probability': round(probability * 100, 2)
        })

    except KeyError as e:
        print(f"Missing field: {e}")
        return jsonify({'error': f'Missing field: {e}'})
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)})

@app.route('/predict_clv', methods=['POST'])
@login_required
def predict_clv():
    try:
        data = request.json
        premium = float(data['Premium_Amount'])
        tenure = int(data['Policy_Tenure'])
        clv = premium * tenure * 12
        return jsonify({'clv': round(clv, 2)})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict_and_explain', methods=['POST'])
@login_required
def predict_and_explain():
    try:
        data = request.json 
        print(f"Received data for explanation: {data}")
        customer_id = data.get('CustomerID', 'single_customer')  # fallback name

        # Numerical features
        num_cols = ['Age', 'Policy_Tenure', 'Policy_Renewal_Count', 'Total_Claims_Filed', 
                    'Claim_Amount_Total', 'Missed_Payments', 'Family_Members_Insured', 
                    'Policy_Cancellation_History', 'Customer_Complaints', 
                    'Complaint_Resolution_Time', 'Customer_Support_Calls', 'Vehicle_Age', 
                    'Premium_Amount']
        num_data_raw = [float(data[col]) for col in num_cols]

        # Date features
        policy_start = datetime.strptime(data['Policy_Start_Date'], '%Y-%m-%d')
        policy_end = datetime.strptime(data['Policy_End_Date'], '%Y-%m-%d')
        last_payment = datetime.strptime(data['Last_Payment_Date'], '%Y-%m-%d')
        last_claim = datetime.strptime(data['Last_Claim_Date'], '%Y-%m-%d')
        date_features_raw = [
            (anchor_date - policy_start).days / 365.25,
            (policy_end - policy_start).days,
            (anchor_date - last_payment).days,
            (anchor_date - last_claim).days
        ]

        cat_cols = ['Gender', 'Marital_Status', 'Region', 'Education_Level', 'Employment_Status',
                    'Policy_Type', 'Mode_of_Communication', 'Payment_Mode', 'Vehicle_Type']
        cat_data_raw = [data[col] for col in cat_cols]

        # Scaled + Encoded Features
        num_data_full = num_data_raw + date_features_raw
        num_scaled = scaler.transform([num_data_full])
        cat_data = [cat_data_raw]
        cat_encoded_raw = encoder.transform(cat_data)
        cat_encoded = cat_encoded_raw.toarray() if hasattr(cat_encoded_raw, 'toarray') else cat_encoded_raw

        features = np.hstack((num_scaled, cat_encoded))
        feature_df = pd.DataFrame(features, columns=feature_names)
        features_ordered = feature_df[feature_names].values

        # Prediction
        prediction = prediction_model.predict(features_ordered)[0]
        probability = prediction_model.predict_proba(features_ordered)[0][1]

        # SHAP values
        shap_values = explainer.shap_values(features_ordered)
        shap_contributions = shap_values[0]

        # Map original values
        original_values = num_data_raw + date_features_raw + cat_encoded[0].tolist()
        input_map = dict(zip(feature_names[:len(num_cols) + 4], num_data_raw + date_features_raw))
        cat_start_idx = len(num_cols) + 4
        for i, col in enumerate(cat_cols):
            one_hot_cols = [f"{col}_{val}" for val in encoder.categories_[i]]
            for j, one_hot_col in enumerate(one_hot_cols):
                if j + cat_start_idx < len(feature_names) and cat_encoded[0][j] == 1:
                    input_map[one_hot_col] = cat_data_raw[i]
                    break

        # Textual Explanation
        explanation_lines = ["Why this customer might churn:"]
        feature_impact = list(zip(feature_names, shap_contributions))
        top_features = sorted(feature_impact, key=lambda x: abs(x[1]), reverse=True)[:5]

        for feature_name, shap_value in top_features:
            original_value = input_map.get(feature_name, "N/A")
            impact_level = "High" if abs(shap_value) > 0.1 else "Moderate" if abs(shap_value) > 0.05 else "Low"
            if shap_value > 0:
                if "Time_Since" in feature_name:
                    explanation_lines.append(
                        f"- {impact_level} risk: It's been {int(original_value)} days since their "
                        f"last {feature_name.split('_')[2].lower()}, which raises churn concern."
                    )
                elif feature_name in num_cols:
                    explanation_lines.append(
                        f"- {impact_level} risk: Their {feature_name.replace('_', ' ').lower()} "
                        f"of {original_value} suggests a higher chance of leaving."
                    )
                else:
                    explanation_lines.append(
                        f"- {impact_level} risk: Being {original_value} increases their likelihood of churning."
                    )

        explanation = "\n".join(explanation_lines) if len(explanation_lines) > 1 else "No significant churn factors identified."
        print(f"Explanation: {explanation}")

        # SHAP Plot (Save for history/report)
        shap_plot_url = None
        try:
            import matplotlib.pyplot as plt
            import os

            abs_contrib = np.abs(shap_contributions)
            top_idx = np.argsort(abs_contrib)[-5:][::-1]
            top_feat_names = [feature_names[i] for i in top_idx]
            top_feat_values = abs_contrib[top_idx]

            plt.figure(figsize=(10, 5))
            plt.barh(top_feat_names[::-1], top_feat_values[::-1], color='coral')
            plt.xlabel("SHAP Value (Impact)")
            plt.title("Top Churn Drivers (Individual Customer)")
            plt.tight_layout()

            safe_name = f"{customer_id}_shap.png".replace(" ", "_")
            image_path = os.path.join(app.config['PLOT_FOLDER'], safe_name)
            plt.savefig(image_path)
            plt.close()
            # Ensure forward slashes in the URL
            shap_relative_path = f"plots/{safe_name}".replace('\\', '/')
            shap_plot_url = url_for('static', filename=shap_relative_path, _external=True)
            print(f"[INFO] SHAP plot saved at: {image_path}, URL: {shap_plot_url}")
        except Exception as e:
            print(f"[SHAP IMAGE ERROR] {e}")
            shap_plot_url = None

        save_single_prediction_history(
            user_email=session['user'],
            customer_id=customer_id,
            prediction='Churn' if prediction == 1 else 'No Churn',
            probability=round(probability * 100, 2),
            analysis_date=datetime.now().strftime("%B %d, %Y")
        )

        churn_label = 'Churn' if prediction == 1 else 'No Churn'
        response = {
            'prediction': churn_label,
            'probability': round(probability * 100, 2)
        }

        if churn_label == 'Churn':
            response['explanation'] = explanation
            response['shap_plot_url'] = shap_plot_url
        else:
            response['explanation'] = ''
            response['shap_plot_url'] = ''

        return jsonify(response)


    except KeyError as e:
        print(f"Missing field: {e}")
        return jsonify({'error': f'Missing field: {e}'})
    except Exception as e:
        print(f"Explanation error: {e}")
        return jsonify({'error': str(e)})


@app.route('/generate_retention_strategy', methods=['POST'])
@login_required
def generate_retention_strategy():
    try:
        data = request.json
        print(f"Received data for retention strategy: {data}")

        # Extract prediction and probability
        predict_response = predict()
        predict_result = predict_response.get_json()
        if 'error' in predict_result:
            return jsonify({'error': predict_result['error']})
        
        churn_prediction = predict_result['prediction']
        churn_probability = predict_result['probability']

        # Only generate strategies for churned customers
        if churn_prediction != 'Churn':
            return jsonify({
                'strategies': 'No retention strategies are necessary, as the customer is predicted to remain loyal.'
            })

        # Extract raw input values
        num_cols = ['Age', 'Policy_Tenure', 'Policy_Renewal_Count', 'Total_Claims_Filed', 
                    'Claim_Amount_Total', 'Missed_Payments', 'Family_Members_Insured', 
                    'Policy_Cancellation_History', 'Customer_Complaints', 
                    'Complaint_Resolution_Time', 'Customer_Support_Calls', 'Vehicle_Age', 
                    'Premium_Amount']
        num_data_raw = [float(data[col]) for col in num_cols]

        policy_start = datetime.strptime(data['Policy_Start_Date'], '%Y-%m-%d')
        policy_end = datetime.strptime(data['Policy_End_Date'], '%Y-%m-%d')
        last_payment = datetime.strptime(data['Last_Payment_Date'], '%Y-%m-%d')
        last_claim = datetime.strptime(data['Last_Claim_Date'], '%Y-%m-%d')
        date_features_raw = [
            (anchor_date - policy_start).days / 365.25,
            (policy_end - policy_start).days,
            (anchor_date - last_payment).days,
            (anchor_date - last_claim).days
        ]

        cat_cols = ['Gender', 'Marital_Status', 'Region', 'Education_Level', 'Employment_Status',
                    'Policy_Type', 'Mode_of_Communication', 'Payment_Mode', 'Vehicle_Type']
        cat_data_raw = [data[col] for col in cat_cols]

        # Rule-based retention strategies with percentage-based calculations
        strategy_lines = ["Recommended retention strategies to prevent churn:"]

        # Rule 1: High churn probability
        if churn_probability > 70:
            strategy_lines.append(
                f"<span class='strategy'>Personalized discount offer:</span> The customer’s high churn risk ({churn_probability}%) warrants an immediate incentive. "
                f"Offer a 15% discount on their annual premium or extend coverage by 2 months at no cost, which could increase retention probability by approximately 15-20% based on industry benchmarks."
            )

        # Rule 2: High missed payments
        if num_data_raw[num_cols.index('Missed_Payments')] >= 2:
            strategy_lines.append(
                f"<span class='strategy'>Flexible payment plan:</span> With {int(num_data_raw[num_cols.index('Missed_Payments')])} missed payments, payment difficulties are evident. "
                f"Propose a monthly installment plan splitting the annual premium into 12 equal payments (approximately 8.33% of the premium per month), potentially reducing churn by 8-15%."
            )

        # Rule 3: Long complaint resolution time
        if num_data_raw[num_cols.index('Complaint_Resolution_Time')] > 10:
            strategy_lines.append(
                f"<span class='strategy'>Expedited complaint resolution:</span> The average complaint resolution time of {int(num_data_raw[num_cols.index('Complaint_Resolution_Time')])} days is excessive. "
                f"Prioritize resolving open complaints within 5 days and schedule a follow-up call, which could improve retention by 10%."
            )

        # Rule 4: High customer complaints
        if num_data_raw[num_cols.index('Customer_Complaints')] >= 3:
            strategy_lines.append(
                f"<span class='strategy'>Dedicated support agent:</span> The customer has lodged {int(num_data_raw[num_cols.index('Customer_Complaints')])} complaints, indicating dissatisfaction. "
                f"Assign a dedicated support agent to address their concerns, projecting a 12% reduction in churn likelihood."
            )

        # Rule 5: Recent policy cancellation history
        if num_data_raw[num_cols.index('Policy_Cancellation_History')] >= 1:
            strategy_lines.append(
                f"<span class='strategy'>Renewal incentive:</span> With {int(num_data_raw[num_cols.index('Policy_Cancellation_History')])} prior cancellations, the customer may hesitate to renew. "
                f"Offer a 5% discount on the renewal premium or a free policy add-on, expecting a 15% retention boost."
            )

        # Rule 6: Low policy tenure
        if num_data_raw[num_cols.index('Policy_Tenure')] < 1:
            strategy_lines.append(
                f"<span class='strategy'>Onboarding support:</span> As a new customer with a policy tenure of {num_data_raw[num_cols.index('Policy_Tenure')]} years, they may lack familiarity with benefits. "
                f"Provide access to onboarding webinars or a policy guide, anticipating an 8% increase in retention."
            )

        # Rule 7: High time since last claim
        if date_features_raw[3] > 365:
            strategy_lines.append(
                f"<span class='strategy'>Policy review engagement:</span> It has been {int(date_features_raw[3])} days since the last claim, suggesting low engagement. "
                f"Schedule a policy review or vehicle wellness check to re-engage the customer, aiming for a 7% churn reduction."
            )

        # Rule 8: Region-specific (Rural)
        if cat_data_raw[cat_cols.index('Region')] == 'Rural':
            strategy_lines.append(
                f"<span class='strategy'>Localized engagement:</span> As a rural customer, in-person interaction may strengthen loyalty. "
                f"Arrange a local workshop or agent visit to discuss their needs, targeting a 10% retention uplift."
            )

        # Rule 9: Payment mode (Cash)
        if cat_data_raw[cat_cols.index('Payment_Mode')] == 'Cash':
            strategy_lines.append(
                f"<span class='strategy'>Digital payment transition:</span> The customer’s reliance on cash payments may cause inconvenience. "
                f"Assist with setting up digital payment options (e.g., credit card or online banking), expecting a 5% churn reduction."
            )

        # Default message if no specific rules triggered
        if len(strategy_lines) == 1:
            strategy_lines.append(
                "<span class='strategy'>Personalized engagement:</span> No specific churn drivers were identified, but proactive outreach is recommended. "
                "Contact the customer to discuss their needs and offer tailored benefits, aiming to maintain loyalty."
            )

        strategies = "\n".join(strategy_lines)
        gemini_response = explain_retention_strategy(strategies, data)
        return jsonify({
            'strategies': gemini_response
        })

    except KeyError as e:
        print(f"Missing field: {e}")
        return jsonify({'error': f'Missing field: {e}'})
    except Exception as e:
        print(f"Retention strategy error: {e}")
        return jsonify({'error': str(e)})


ALLOWED_EXTENSIONS = {'csv'}
REQUIRED_COLUMNS = [
    'Customer_ID', 'Age', 'Gender', 'Marital_Status', 'Region',
    'Education_Level', 'Employment_Status', 'Policy_Type', 'Policy_Tenure',
    'Policy_Renewal_Count', 'Total_Claims_Filed', 'Claim_Amount_Total',
    'Missed_Payments', 'Family_Members_Insured', 'Policy_Cancellation_History',
    'Mode_of_Communication', 'Payment_Mode', 'Customer_Complaints',
    'Complaint_Resolution_Time', 'Customer_Support_Calls',
    'Policy_Start_Date', 'Policy_End_Date', 'Last_Claim_Date',
    'Last_Payment_Date', 'Vehicle_Age', 'Vehicle_Type',
    'Premium_Amount', 'Churn'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Ensure logging is set up
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.route('/bulk_upload', methods=['GET', 'POST'])
@login_required
def bulk_upload():
    default_grouping_columns = [
        col for col in REQUIRED_COLUMNS if col not in NON_GROUPING_COLUMNS
    ]

    if request.method == 'POST':
        project_name = request.form.get('project_name')
        file = request.files.get('file')
        group_feature = request.form.get('group_feature')
        analysis_type = request.form.get('analysis_type', 'pattern')

        logger.debug(f"Received form data: project_name={project_name}, group_feature={group_feature}, analysis_type={analysis_type}")

        if not project_name or not file:
            flash('Project name and file are required.', 'error')
            return redirect(url_for('bulk_upload'))

        if not file.filename.endswith('.csv'):
            flash('Only CSV files are allowed.', 'error')
            return redirect(url_for('bulk_upload'))

        file_path = save_uploaded_file(file, app.config['UPLOAD_FOLDER'])
        df, error = read_csv_file(file_path)

        if error:
            flash(f'Error reading CSV file: {error}', 'error')
            return redirect(url_for('bulk_upload'))

        if df.empty:
            flash('Uploaded file is empty.', 'error')
            return redirect(url_for('bulk_upload'))

        if group_feature and group_feature not in df.columns:
            flash(f"Selected feature '{group_feature}' not found in CSV.", 'error')
            return redirect(url_for('bulk_upload'))

        relative_plot_paths = {}

        # Generate KPI & pie overview (shared)
        summary_result = generate_summary_insights(df, app.config['PLOT_FOLDER'], project_name)
        summary_kpis = summary_result['summary']
        if 'Churn Rate Overview' in summary_result['plot']:
            relative_plot_paths['Churn Rate Overview'] = summary_result['plot']['Churn Rate Overview']

        if analysis_type == 'pattern':
            logger.debug("Running pattern analysis...")
            relative_plot_paths.update({
                **{k: f'plots/{os.path.basename(v)}' for k, v in perform_analysis(df, app.config['PLOT_FOLDER'], project_name).items() if not k.endswith('_plot_data')},
                **{k: f'plots/{os.path.basename(v)}' for k, v in generate_pattern_plots(df, project_name, output_dir=app.config['PLOT_FOLDER']).items() if not k.endswith('_plot_data')},
                **{k: f'plots/{os.path.basename(v)}' for k, v in generate_time_based_plots(df, project_name, group_features=None, output_dir=app.config['PLOT_FOLDER']).items() if not k.endswith('_plot_data')},
                **{k: f'plots/{os.path.basename(v)}' for k, v in generate_risk_group_analysis(df, project_name, output_dir=app.config['PLOT_FOLDER']).items() if not k.endswith('_plot_data')},
                **{k: f'plots/{os.path.basename(v)}' for k, v in generate_top_churn_drivers_plot(df, project_name, output_dir=app.config['PLOT_FOLDER']).items() if not k.endswith('_plot_data')}
            })

        elif analysis_type == 'grouping' and group_feature:
            logger.debug(f"Running grouping analysis by: {group_feature}")
            relative_plot_paths.update({
                **{k: f'plots/{os.path.basename(v)}' for k, v in generate_grouping_plot(df, project_name, group_features=[group_feature], output_dir=app.config['PLOT_FOLDER']).items() if not k.endswith('_plot_data')}
            })

        # Save cleaned CSV and session data
        df_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{project_name}_cleaned.csv")
        df.to_csv(df_path, index=False)

        # After all plots added to relative_plot_paths
        plot_explanations = {}
        # for title, path in relative_plot_paths.items():
        #     if title.endswith("_plot_data"): continue  # Skip metadata

        #     summary = f"This plot titled '{title}' shows trends related to customer churn."
        #     explanation = explain_plot(title, summary)
        #     plot_explanations[title] = explanation


        session['bulk_analysis'] = {
    'project_name': project_name,
    'file_path': file_path,
    'df_path': df_path,
    'df': df.to_json(),
    'plot_paths': relative_plot_paths,
    'dataset_summary': {
        'row_count': len(df),
        'columns': list(df.columns),
        'churn_rate': df['Churn'].mean() * 100
    },
    'summary_kpis': summary_kpis,
    'analysis_type': analysis_type,         # 
    'group_feature': group_feature if analysis_type == 'grouping' else None  # 
}


        flash('File uploaded and analyzed successfully!', 'success')
        return render_template(
            'bulk_upload.html',
            plot_paths=relative_plot_paths,
            project_name=project_name,
            dataset_summary=session['bulk_analysis']['dataset_summary'],
            grouping_columns=get_grouping_columns(df),
            summary_kpis=summary_kpis,
            analysis_type=analysis_type,
            plot_explanations=plot_explanations  # 

        )

    return render_template(
        'bulk_upload.html',
        plot_paths=None,
        project_name=None,
        dataset_summary=None,
        grouping_columns=default_grouping_columns,
        summary_kpis=None,
        analysis_type=None,
        plot_explanations=None  # 

    )

    
#
@app.route('/explain_plot', methods=['POST'])
@login_required
def explain_plot_route():
    try:
        data = request.get_json()
        title = data.get("title", "")
        summary = data.get("summary", "")

        if not title or not summary:
            return jsonify({"error": "Missing title or summary"}), 400

        explanation = explain_plot(title, summary)
        return jsonify({"explanation": explanation})

    except Exception as e:
        print(f"[EXPLAIN ERROR] {e}")
        return jsonify({"explanation": "Unable to generate explanation."})

@app.route('/explain_churn_reason_ai', methods=['POST'])
@login_required
def explain_churn_reason_ai():
    try:
        data = request.json
        top_features_text = data.get("top_features_text", "")
        raw_data = data.get("raw_data", {})
        explanation = explain_top_churn_drivers(top_features_text, raw_data)
        return jsonify({"explanation": explanation})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/explain_retention_strategy_ai', methods=['POST'])
@login_required
def explain_retention_strategy_ai():
    try:
        data = request.json
        strategy_text = data.get("strategy_text", "")
        raw_data = data.get("raw_data", {})
        explanation = explain_retention_strategy(strategy_text, raw_data)
        return jsonify({"explanation": explanation})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

import pdfkit
from flask import send_file, render_template
from datetime import datetime

# In app.py, update the /generate_report route
from datetime import datetime  # Add this import at the top

@app.route('/generate_report')
@login_required
def generate_report():
    try:
        if 'bulk_analysis' not in session:
            flash("No analysis data found for report.", "error")
            return redirect(url_for('bulk_upload'))

        project_name = session['bulk_analysis']['project_name']
        summary = session['bulk_analysis']['summary_kpis']
        plot_paths = session['bulk_analysis']['plot_paths']
        file_name = session['bulk_analysis'].get('file_path', 'unknown.csv')
        file_name = os.path.basename(file_name)
        
        analysis_date = datetime.now().strftime("%B %d, %Y")
        summary['analysis_date'] = analysis_date

        analysis_type = session['bulk_analysis'].get('analysis_type', 'pattern')
        group_feature = session['bulk_analysis'].get('group_feature', '')
        pdf_io = generate_pdf_report(project_name, summary, plot_paths, analysis_type, group_feature)


        report_filename = f"{project_name}_report.pdf"
        report_path = os.path.join(app.config['REPORT_FOLDER'], report_filename)
        with open(report_path, 'wb') as f:
            f.write(pdf_io.getvalue())
        relative_report_path = f"reports/{report_filename}".replace('\\', '/')
        print(f"[DEBUG] Saving report path: {relative_report_path}")

        save_bulk_analysis_history(
            user_email=session['user'],
            project_name=project_name,
            file_name=file_name,
            analysis_date=analysis_date,
            report_path=relative_report_path
        )

        return send_file(
            pdf_io,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"{project_name}_ChurnShield_Report.pdf"
        )

    except Exception as e:
        print(f"[REPORT ERROR] {e}")
        flash("Report generation failed.", "error")
        return redirect(url_for('bulk_upload'))

# from report_generator import generate_single_prediction_pdf_report
@app.route('/generate_single_report', methods=['POST'])
@login_required
def generate_single_report():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data received'}), 400

        customer_id = data.get('customer_id')
        prediction = data.get('prediction')
        probability = data.get('probability')
        explanation = data.get('explanation')
        shap_plot_url = data.get('shap_plot_url')
        strategies = data.get('retention_strategies')
        analysis_date = data.get('analysis_date')
        clv = float(data.get('clv').replace('₹', '').replace(',', '')) if isinstance(data.get('clv'), str) else data.get('clv')
        form_inputs = data.get('form_inputs', {})

        pdf_io = generate_single_prediction_pdf_report(
            customer_id=customer_id,
            prediction=prediction,
            probability=probability,
            explanation=explanation,
            shap_plot_url=shap_plot_url,
            retention_strategies=strategies,
            analysis_date=analysis_date,
            form_inputs=form_inputs,
            clv=clv
        )
        report_filename = f"{customer_id}_Churn_Report.pdf"
        report_path = os.path.join(app.config['REPORT_FOLDER'], report_filename)
        with open(report_path, 'wb') as f:
            f.write(pdf_io.getvalue())
        relative_report_path = f"reports/{report_filename}".replace('\\', '/')
        print(f"[DEBUG] Saving report path: {relative_report_path}")

        save_single_prediction_history(
            user_email=session['user'],
            customer_id=customer_id,
            prediction=prediction,
            probability=probability,
            analysis_date=analysis_date,
            report_path=relative_report_path
        )

        return send_file(
            pdf_io,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"{customer_id}_Churn_Report.pdf"
        )

    except Exception as e:
        print(f"[SINGLE REPORT ERROR] {e}")
        return jsonify({'error': f'Failed to generate PDF report: {str(e)}'}), 500

# Add this route to your existing app.py
@app.route('/calculate_clv', methods=['POST'])
def calculate_clv():
    try:
        data = request.get_json()
        premium_amount = data.get('premium_amount', 0)
        policy_tenure = data.get('policy_tenure', 0)
        churn_probability = data.get('churn_probability', 0)
        
        # Simple CLV calculation: Premium_Amount * Policy_Tenure * (1 - Churn_Probability/100)
        clv = premium_amount * policy_tenure * (1 - churn_probability / 100)
        
        return jsonify({'clv': max(clv, 0)})  # Ensure non-negative CLV
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
def save_single_prediction_history(user_email, customer_id, prediction, probability, analysis_date, report_path=None):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO single_prediction_history (
                user_email, customer_id, prediction, probability, analysis_date, report_path
            ) VALUES (%s, %s, %s, %s, %s, %s)
        ''', (
            user_email, customer_id, prediction, probability, analysis_date, report_path
        ))
        conn.commit()
        conn.close()
        print(f"[INFO] Saved single prediction history for customer {customer_id}")
    except Exception as e:
        print(f"[SAVE HISTORY ERROR] {e}")

def get_single_prediction_history(user_email, limit=10, offset=0):
    print(f"[DEBUG] Fetching history for user_email={user_email}, limit={limit}, offset={offset}")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()  # DictCursor is set in get_db_connection
        cursor.execute('''
            SELECT id, customer_id, prediction, probability, analysis_date, report_path
            FROM single_prediction_history
            WHERE user_email = %s
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        ''', (user_email, limit, offset))
        history = cursor.fetchall()
        conn.close()
        print(f"[INFO] Retrieved {len(history)} history records for {user_email}: {history}")
        return history
    except Exception as e:
        print(f"[GET HISTORY ERROR] for {user_email}: {e}")
        return []

@app.route('/history')
@login_required
def history():
    user_email = session['user']
    print(f"[DEBUG] User email in session: {user_email}")
    history_data = get_single_prediction_history(user_email)
    return render_template('history.html', history_data=history_data)

@app.route('/delete_history/<int:record_id>', methods=['POST'])
@login_required
def delete_history(record_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            DELETE FROM single_prediction_history
            WHERE id = %s AND user_email = %s
        ''', (record_id, session['user']))
        affected_rows = cursor.rowcount
        conn.commit()
        conn.close()
        if affected_rows > 0:
            flash('Prediction record deleted successfully.', 'success')
        else:
            flash('Record not found or not authorized.', 'error')
        print(f"[DEBUG] Delete attempted for record_id={record_id}, user={session['user']}, affected_rows={affected_rows}")
    except Exception as e:
        print(f"[DELETE HISTORY ERROR] {e}")
        flash('Failed to delete prediction record.', 'error')
    return redirect(url_for('history'))

def save_bulk_analysis_history(user_email, project_name, file_name, analysis_date, report_path=None):
    print(f"[DEBUG] Saving bulk history: user_email={user_email}, project_name={project_name}, file_name={file_name}, analysis_date={analysis_date}, report_path={report_path}")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO bulk_analysis_history (
                user_email, project_name, file_name, analysis_date, report_path
            ) VALUES (%s, %s, %s, %s, %s)
        ''', (user_email, project_name, file_name, analysis_date, report_path))
        conn.commit()
        print(f"[INFO] Saved bulk analysis history for project {project_name}")
    except Exception as e:
        print(f"[SAVE BULK HISTORY ERROR] {e}")
    finally:
        conn.close()

def get_bulk_analysis_history(user_email, limit=10, offset=0):
    print(f"[DEBUG] Fetching bulk history for user_email={user_email}, limit={limit}, offset={offset}")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, project_name, file_name, analysis_date, report_path
            FROM bulk_analysis_history
            WHERE user_email = %s
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        ''', (user_email, limit, offset))
        history = cursor.fetchall()
        conn.close()
        print(f"[INFO] Retrieved {len(history)} bulk history records for {user_email}: {history}")
        return history
    except Exception as e:
        print(f"[GET BULK HISTORY ERROR] for {user_email}: {e}")
        return []
    
@app.route('/bulk_history')
@login_required
def bulk_history():
    user_email = session['user']
    print(f"[DEBUG] User email in session: {user_email}")
    bulk_history_data = get_bulk_analysis_history(user_email)
    return render_template('bulk_history.html', bulk_history_data=bulk_history_data)
@app.route('/delete_bulk_history/<int:record_id>', methods=['POST'])
@login_required
def delete_bulk_history(record_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            DELETE FROM bulk_analysis_history
            WHERE id = %s AND user_email = %s
        ''', (record_id, session['user']))
        affected_rows = cursor.rowcount
        conn.commit()
        conn.close()
        if affected_rows > 0:
            flash('Bulk analysis record deleted successfully.', 'success')
        else:
            flash('Record not found or not authorized.', 'error')
        print(f"[DEBUG] Delete bulk attempted for record_id={record_id}, user={session['user']}, affected_rows={affected_rows}")
    except Exception as e:
        print(f"[DELETE BULK HISTORY ERROR] {e}")
        flash('Failed to delete bulk analysis record.', 'error')
    return redirect(url_for('bulk_history'))
    
if __name__ == "__main__":
    app.run(debug=True)