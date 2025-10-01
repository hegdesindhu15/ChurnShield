def prepare_features_for_model(df, scaler, encoder, feature_names, anchor_date):
    import pandas as pd
    import numpy as np

    df = df.copy()
    # Ensure datetime conversion FIRST
    # Ensure engineered features exactly match training
    df['Policy_Start_Date'] = pd.to_datetime(df['Policy_Start_Date'], errors='coerce')
    df['Policy_End_Date'] = pd.to_datetime(df['Policy_End_Date'], errors='coerce')
    df['Last_Claim_Date'] = pd.to_datetime(df['Last_Claim_Date'], errors='coerce')
    df['Last_Payment_Date'] = pd.to_datetime(df['Last_Payment_Date'], errors='coerce')

    df['Policy_Duration'] = (df['Policy_End_Date'] - df['Policy_Start_Date']).dt.days.fillna(0)
    df['Policy_Age'] = ((anchor_date - df['Policy_Start_Date']).dt.days / 365.25).fillna(0)
    df['Time_Since_Last_Claim'] = (anchor_date - df['Last_Claim_Date']).dt.days.fillna(0)
    df['Time_Since_Last_Payment'] = (anchor_date - df['Last_Payment_Date']).dt.days.fillna(0)

    # Numerical features
    num_cols = [
        'Age', 'Policy_Tenure', 'Policy_Renewal_Count', 'Total_Claims_Filed', 'Claim_Amount_Total',
        'Missed_Payments', 'Family_Members_Insured', 'Policy_Cancellation_History',
        'Customer_Complaints', 'Complaint_Resolution_Time', 'Customer_Support_Calls',
        'Vehicle_Age', 'Premium_Amount','Policy_Duration' ,
        'Policy_Age', 'Time_Since_Last_Claim', 'Time_Since_Last_Payment'
    ]

    num_scaled = scaler.transform(df[num_cols])

    # Categorical features
    cat_cols = [
        'Gender', 'Marital_Status', 'Region', 'Education_Level', 'Employment_Status',
        'Policy_Type', 'Mode_of_Communication', 'Payment_Mode', 'Vehicle_Type'
    ]
    cat_encoded = encoder.transform(df[cat_cols])
    if hasattr(cat_encoded, 'toarray'):
        cat_encoded = cat_encoded.toarray()

    # Combine
    features = np.hstack((num_scaled, cat_encoded))
    feature_df = pd.DataFrame(features, columns=feature_names)

    return feature_df
