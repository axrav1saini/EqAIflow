import os
import pandas as pd
from dataset_scanner import scan_for_bias
from narrative_oracle import generate_plain_english_report, generate_recommendations

def main():
    csv_path = input("[?] Enter the path to the dataset CSV (e.g., datasets/recruitment_data.csv): ").strip()
    sensitive_col = input("[?] Enter the sensitive column name (e.g., Gender, Age): ").strip()
    target_col = input("[?] Enter the target column name (e.g., HiringDecision): ").strip()

    # 1. Setup Data
    if not os.path.exists(csv_path):
        print(f"[-] No dataset found at '{csv_path}'")
        return

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