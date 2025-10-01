# utils/report_generator.py

import pdfkit
import io
import os
from flask import render_template_string, url_for

PDF_CONFIG = pdfkit.configuration(
    wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
)

def generate_pdf_report(project_name, summary, plot_paths, analysis_type='pattern', group_feature=''):
    try:
        sections_patterns = {
            'Churn vs No-Churn': ['Churn Rate Distribution'],
            'Time-based Churn Trends': ['Monthly Churn Rate Over Time', 'Yearly Churn Rate Over Time'],
            'Top Churn Drivers': ['Top Churn Drivers']
        }
        sections_groups = {
            'Churn Rate X Selected Option Analysis': ['Churn Rate by ']
        }

        section_html_blocks = []
        for section, keywords in sections_patterns.items():
            section_plots = []
            for title, path in plot_paths.items():
                for keyword in keywords:
                    if title == keyword or title.startswith(keyword):
                        plot_url = url_for('static', filename=path, _external=True)
                        section_plots.append(f"""
                            <div>
                                <h3>{title}</h3>
                                <img src="{plot_url}" style="max-width: 100%; height: auto;">
                            </div>
                        """)
            if section_plots:
                section_html_blocks.append(f"""
                    <h2>{section}</h2>
                    {' '.join(section_plots)}
                """)
        
        if analysis_type == 'grouping' and group_feature:
            section = f"Churn Rate by {group_feature}"
            section_plots = []
            for title, path in plot_paths.items():
                if title.strip() == section:
                    plot_url = url_for('static', filename=path, _external=True)
                    section_plots.append(f"""
                        <div>
                            <h3>{title}</h3>
                            <img src="{plot_url}" style="max-width: 100%; height: auto;">
                        </div>
                    """)
            if section_plots:
                section_html_blocks.append(f"""
                    <h2>{section}</h2>
                    {' '.join(section_plots)}
                """)



        report_html = render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ChurnShield Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20mm; }
                h1 { text-align: center; }
                h2 { margin-top: 20px; }
                .summary { margin-bottom: 20px; }
                .summary p { margin: 5px 0; }
            </style>
        </head>
        <body>
            <h1>ChurnShield Analysis Report</h1>
            <div class="summary">
                <p><strong>Project:</strong> {{ project_name }}</p>
                <p><strong>Analysis Date:</strong> {{ summary.analysis_date }}</p>
                <p><strong>Total Customers:</strong> {{ summary.total_customers }}</p>
                <p><strong>Churned:</strong> {{ summary.total_churned }}</p>
                <p><strong>Churn Rate:</strong> {{ summary.churn_rate }}%</p>
            </div>
            <h2>Analysis Plots</h2>
            {{ plots|safe }}
        </body>
        </html>
        """, project_name=project_name, summary=summary, plots="\n".join(section_html_blocks))

        debug_path = os.path.join("frontend", "static", "debug_report.html")
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(report_html)

        pdf_options = {
            'enable-local-file-access': None,
            'quiet': '',
            'page-size': 'A4',
            'margin-top': '20mm',
            'margin-bottom': '20mm',
            'margin-left': '15mm',
            'margin-right': '15mm'
        }
        pdf_bytes = pdfkit.from_string(report_html, False, configuration=PDF_CONFIG, options=pdf_options)
        return io.BytesIO(pdf_bytes)

    except Exception as e:
        print(f"[PDF GENERATION ERROR] {str(e)}")
        raise

def generate_single_prediction_pdf_report(
    customer_id,
    prediction,
    probability,
    explanation,
    shap_plot_url,
    retention_strategies,
    analysis_date,
    form_inputs=None,
    clv=None
):
    try:
        import os
        from flask import url_for

        # Extract filename and normalize path
        shap_filename = os.path.basename(shap_plot_url) if shap_plot_url else None
        shap_relative_path = f"plots/{shap_filename}".replace('\\', '/') if shap_filename else None
        shap_file_path = os.path.abspath(os.path.join("frontend", "static", "plots", shap_filename)) if shap_filename else None

        # Generate image tag using absolute URL
        shap_img_tag = "No SHAP plot available."
        if shap_file_path and os.path.exists(shap_file_path):
            shap_img_url = url_for('static', filename=shap_relative_path, _external=True)
            shap_img_tag = f'<img src="{shap_img_url}" alt="SHAP Plot" style="max-width: 100%; height: auto;">'
            print(f"[INFO] Using SHAP plot URL: {shap_img_url}")
        else:
            print(f"[WARNING] SHAP plot file not found at: {shap_file_path}")

        # Strategy lines
        strategy_lines = retention_strategies.split('\n') if retention_strategies else []
        strategy_list_html = "".join(
            f"    {line.strip().replace('-', '').replace('AI suggests', 'Based on studies')}<br>"
            for line in strategy_lines if line.strip()
        )

        # Explanation lines
        explanation_lines = explanation.split('\n') if explanation else []
        explanation_list_html = "".join(
            f"    {line.strip().replace('-', '').replace('High risk', 'Analysis indicates')}<br>"
            for line in explanation_lines if line.strip()
        )

        # Form inputs
        inputs_html = ""
        if form_inputs:
            inputs_html = "<ul>" + "".join(
                f"    <li>{k}: {v}</li>" for k, v in form_inputs.items()
            ) + "</ul>"

        # Render HTML
        report_html = render_template_string("""
        <html>
        <head>
            <title>ChurnShield Prediction Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #2c3e50; }
                h2 { color: #34495e; }
                ul { list-style-type: disc; margin-left: 20px; }
                .section { margin-bottom: 20px; }
                img { display: block; margin: 10px 0; }
            </style>
        </head>
        <body>
            <h1>ChurnShield Prediction Report</h1>
            <div class="section">
                <h2>Customer Summary</h2>
                <ul>
                    <li><strong>Customer ID:</strong> {{ customer_id }}</li>
                    <li><strong>Analysis Date:</strong> {{ analysis_date }}</li>
                    <li><strong>Churn Prediction:</strong> {{ prediction }}</li>
                    <li><strong>Churn Probability:</strong> {{ probability }}%</li>
                    <li><strong>Customer Lifetime Value:</strong> &#8377;{{ clv|round(2) }}</li>
                </ul>
            </div>
            <div class="section">
                <h2>Customer Input Details</h2>
                {{ inputs_html|safe }}
            </div>
            <div class="section">
                <h2>Top Churn Drivers (SHAP)</h2>
                {{ shap_img_tag|safe }}
            </div>
            <div class="section">
                <h2>Churn Explanation</h2>
                <ul>
                    {{ explanation_list_html|safe }}
                </ul>
            </div>
            <div class="section">
                <h2>AI-Suggested Retention Strategies</h2>
                <ul>
                    {{ strategy_list_html|safe }}
                </ul>
            </div>
        </body>
        </html>
        """, customer_id=customer_id,
           prediction=prediction,
           probability=probability,
           clv=clv,
           analysis_date=analysis_date,
           inputs_html=inputs_html,
           shap_img_tag=shap_img_tag,
           explanation_list_html=explanation_list_html,
           strategy_list_html=strategy_list_html)

        # Save debug HTML
        debug_path = os.path.join("frontend", "static", f"debug_single_prediction_{customer_id}.html")
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(report_html)

        # Convert to PDF
        pdf_options = {
            'enable-local-file-access': None,
            'quiet': '',
            'page-size': 'A4',
            'margin-top': '20mm',
            'margin-bottom': '20mm',
            'margin-left': '15mm',
            'margin-right': '15mm'
        }

        pdf_bytes = pdfkit.from_string(report_html, False, configuration=PDF_CONFIG, options=pdf_options)
        if not pdf_bytes:
            raise ValueError("PDF generation resulted in an empty file.")
        return io.BytesIO(pdf_bytes)

    except Exception as e:
        print(f"[SINGLE PREDICTION PDF ERROR] {str(e)}")
        raise