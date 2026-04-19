import os
import pandas as pd
from dataset_scanner import scan_for_bias
from narrative_oracle import generate_plain_english_report, generate_recommendations

def create_dummy_data(file_path: str):
    """Creates a sample dataset with deliberate representation bias."""
    # In this dataset, 'M' has an 80% hire rate, and 'F' has a 20% hire rate.
    data = {
        'gender': ['M', 'M', 'M', 'M', 'M', 'F', 'F', 'F', 'F', 'F'],
        'hired':  [ 1,   1,   1,   1,   0,   0,   0,   0,   0,   1 ]
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"[*] Created dummy dataset at '{file_path}'")

def main():
    csv_path = "sample_bias_data.csv"
    sensitive_col = "gender"
    target_col = "hired"

    # 1. Setup Data
    if not os.path.exists(csv_path):
        create_dummy_data(csv_path)

    # 2. Run Scanner
    print("\n[+] Running Dataset Scanner...")
    metrics_json = scan_for_bias(csv_path, sensitive_col, target_col)
    print("\n--- Raw Statistical Metrics ---")
    print(metrics_json)

    # 3. Run Narrative Oracle
    print("\n[+] Passing metrics to EquiLens Narrative Engine (Gemini)...")
    if "error" not in metrics_json.lower():
        assessment = generate_plain_english_report(metrics_json)
        
        print("\n--- EquiLens Narrative Report: Assessment ---\n")
        print(assessment)
        
        show_fixes = input("\n[?] Would you like to generate and view recommended fixes? (Y/N): ").strip().upper()
        if show_fixes == 'Y':
            print("\n[+] Generating recommendations...")
            recommendations = generate_recommendations(metrics_json)
            print("\n--- EquiLens Narrative Report: Recommendations ---\n")
            print(recommendations)
    else:
        print("Skipping report generation due to scanning errors.")

if __name__ == "__main__":
    main()