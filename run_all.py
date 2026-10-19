"""Run the complete Musanze Harvest Intelligence assignment pipeline."""
import argparse
from src.pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-id", required=True, help="Your INES institutional username/email")
    parser.add_argument("--seed", required=True, type=int, help="Last four digits of your registration number")
    args = parser.parse_args()
    info = run_pipeline(args.student_id, args.seed)
    print("Musanze Harvest Intelligence Service completed.")
    print(f"Student ID: {info['student_id']}")
    print(f"Seed: {info['seed']}")
    print(f"Verification code: {info['verification_code']}")
    print(f"Dataset fingerprint (SHA-256): {info['dataset_sha256']}")
    print("Generated artifacts: 11 files/items")
    print(f"Selected regression model: {info['regression_model']}")
    print(f"Selected classification model: {info['classification_model']}")
    print(f"Selected cluster count: {info['selected_cluster_count']}")
    print(f"Report: docs/AI_A2_{'2528159'}_REPORT.pdf")
if __name__ == "__main__": main()
