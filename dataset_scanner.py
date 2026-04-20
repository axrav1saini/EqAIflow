import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate
import json
import os
import itertools
from google import genai
from dotenv import load_dotenv

# Load env for the mapping agent
load_dotenv()


def map_columns_with_llm(
    user_sensitive: str, user_target: str, dataset_description: str, available_columns: list
) -> tuple:
    """Uses Gemini to semantically map user input (with typos/synonyms) to actual dataset columns."""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    prompt = f"""
    You are a data-mapping agent. The user typed the following column names (which may contain typos or be synonyms):
    User Sensitive Column: '{user_sensitive}'
    User Target Column: '{user_target}'
    Dataset Description (Context): '{dataset_description}'

    The actual available columns in the dataset are:
    {available_columns}

    Map the user's inputs to the exact column names from the dataset.
    Respond ONLY with a JSON object containing exactly two keys: "sensitive_columns" (a list of strings) and "target_column" (a string).
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        )
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]

        mapping = json.loads(raw_text.strip())
        return mapping.get("sensitive_columns", [user_sensitive] if user_sensitive else []), mapping.get(
            "target_column", user_target
        )
    except Exception as e:
        print(f"[-] LLM Mapping Error: {str(e)}")
        return [user_sensitive] if user_sensitive else [], user_target


def scan_for_bias(csv_path: str, actual_sensitive_list: list, actual_target: str, dataset_description: str = "") -> str:
    """
    Scans a dataset for disparate impact and representation bias.
    Returns a JSON string of the metrics for the Narrative Oracle.
    """
    try:
        # 1. Load the dataset
        df = pd.read_csv(csv_path)

        if actual_target not in df.columns:
            raise ValueError(
                f"Could not reliably map target column. Inferred: '{actual_target}'."
            )

        # 2. Extract features
        y_true = df[actual_target]
        metrics = {}
        columns_to_check = []

        if actual_sensitive_list:
            for col in actual_sensitive_list:
                if col not in df.columns:
                    raise ValueError(
                        f"Could not reliably map sensitive column. Inferred: '{col}'."
                    )
            columns_to_check = actual_sensitive_list
        else:
            # Auto-detect: Check categorical or low-cardinality columns
            columns_to_check = [
                col
                for col in df.columns
                if col != actual_target and df[col].nunique() <= 15
            ]

        if not columns_to_check:
            raise ValueError("No suitable sensitive columns found to analyze.")

        # Prepare binned features to avoid high cardinality blowups
        sensitive_df = pd.DataFrame()
        for col in columns_to_check:
            if df[col].nunique() > 15:
                sensitive_df[col] = pd.qcut(df[col], q=4, duplicates="drop").astype(str)
            else:
                sensitive_df[col] = df[col].astype(str)

        # 3. Calculate Bias Metrics using MetricFrame
        for col in columns_to_check:
            mf = MetricFrame(
                metrics=selection_rate,
                y_true=y_true,
                y_pred=y_true,
                sensitive_features=sensitive_df[col]
            )
            
            rates = mf.by_group.to_dict()
            min_rate = mf.group_min()
            max_rate = mf.group_max()

            metrics[col] = {
                "demographic_parity_ratio": min_rate / max_rate if max_rate > 0 else 0.0,
                "demographic_parity_difference": max_rate - min_rate,
                "group_selection_rates": {str(k): v for k, v in rates.items()},
                "group_representation": sensitive_df[col].value_counts(normalize=True).to_dict(),
            }

        # Calculate Intersectional Bias if multiple columns exist
        intersectional_metrics = {}
        if len(columns_to_check) > 1:
            for pair in itertools.combinations(columns_to_check, 2):
                pair_name = f"{pair} + {pair}"
                mf_intersect = MetricFrame(
                    metrics=selection_rate,
                    y_true=y_true,
                    y_pred=y_true,
                    sensitive_features=sensitive_df[list(pair)]
                )
                
                rates = mf_intersect.by_group.to_dict()
                min_rate = mf_intersect.group_min()
                max_rate = mf_intersect.group_max()
                
                rep = sensitive_df.groupby(list(pair)).size() / len(sensitive_df)
                
                intersectional_metrics[pair_name] = {
                    "demographic_parity_ratio": min_rate / max_rate if max_rate > 0 else 0.0,
                    "demographic_parity_difference": max_rate - min_rate,
                    "group_selection_rates": {str(k): v for k, v in rates.items()},
                    "group_representation": {str(k): v for k, v in rep.to_dict().items()}
                }
                
        # Prepare final JSON payload
        final_output = {}
        if actual_sensitive_list and len(actual_sensitive_list) == 1:
            final_output = metrics[actual_sensitive_list]
            final_output["sensitive_feature_analyzed"] = actual_sensitive_list
        else:
            final_output["auto_detected_bias_metrics"] = metrics
            
        if intersectional_metrics:
            final_output["intersectional_bias_metrics"] = intersectional_metrics

        return json.dumps(final_output, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})
