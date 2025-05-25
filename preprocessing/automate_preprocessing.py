import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

def auto_preprocess(data_path):
    # Load data
    df = pd.read_csv(data_path)
    
    # Handle missing values
    df['TotalCharges'] = df['TotalCharges'].replace(' ', '0').astype(float)
    
    # Drop irrelevant column
    df = df.drop('customerID', axis=1)
    
    # Encode target variable
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Define features
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = ['gender', 'Partner', 'InternetService', 'Contract']
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])
    
    # Fit-transform
    X = preprocessor.fit_transform(df.drop('Churn', axis=1))
    y = df['Churn']
    
    # Get feature names
    numeric_feature_names = numeric_features
    categorical_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = list(numeric_feature_names) + list(categorical_feature_names)
    
    # Create DataFrame with proper column names
    X_df = pd.DataFrame(X.toarray() if hasattr(X, "toarray") else X, columns=all_feature_names)
    
    # Combine features and target
    combined_data = pd.concat([X_df, y.rename('Churn')], axis=1)
    
    # Save file
    output_path = r'C:\Project\Eksperimen_SML_Haiqel-Aziizul-Hakeem\preprocessing\telco_preprocessed.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_data.to_csv(output_path, index=False)
    
    return X, y

if __name__ == "__main__":
    auto_preprocess("C:\Project\Eksperimen_SML_Haiqel-Aziizul-Hakeem\WA_Fn-UseC_-Telco-Customer-Churn.csv")