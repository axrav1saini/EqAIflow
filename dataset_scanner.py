import pandas as pd
from fairlearn.metrics import demographic_parity_ratio, demographic_parity_difference
import json

def scan_for_bias(csv_path: str, sensitive_feature_col: str, target_col: str) -> str:
    """
    Scans a dataset for disparate impact and representation bias.
    Returns a JSON string of the metrics for the Narrative Oracle.
    """
    try:
        # 1. Load the dataset
        df = pd.read_csv(csv_path)
        
        if sensitive_feature_col not in df.columns or target_col not in df.columns:
            raise ValueError("Sensitive feature or target column not found in dataset.")

        # 2. Extract features
        y_true = df[target_col]
        sensitive_features = df[sensitive_feature_col]

        # 3. Calculate Bias Metrics (Disparate Impact)
        dp_ratio = demographic_parity_ratio(y_true, y_true, sensitive_features=sensitive_features)
        dp_diff = demographic_parity_difference(y_true, y_true, sensitive_features=sensitive_features)
        
        # Calculate representation breakdown
        representation = df[sensitive_feature_col].value_counts(normalize=True).to_dict()

        metrics = {
            "demographic_parity_ratio": dp_ratio,
            "demographic_parity_difference": dp_diff,
            "group_representation": representation,
            "sensitive_feature_analyzed": sensitive_feature_col
        }
        
        return json.dumps(metrics, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})