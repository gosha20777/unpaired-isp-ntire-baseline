import os
import argparse
import subprocess
import re


def get_model_numbers(output_dir):
    model_files = [f for f in os.listdir(output_dir) if re.match(r'baseline_model_\d+\.pth$', f)]
    model_numbers = [int(re.search(r'baseline_model_(\d+)\.pth', f).group(1)) for f in model_files]
    return model_numbers


def main(a, b, output_dir, script_path):
    model_numbers = get_model_numbers(output_dir)

    for model_number in model_numbers:
        if model_number % a == b:
            print(f"Running inference for model {model_number} (satisfies {model_number} % {a} == {b})")
            subprocess.run(["python3", script_path, str(model_number)], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('a', type=int, help="The divisor for the condition")
    parser.add_argument('b', type=int, help="The remainder for the condition")
    parser.add_argument('--output_dir', type=str, default="output", help="Directory containing model files")
    parser.add_argument('--script_path', type=str, default="inference.py", help="Path to the inference script")
    args = parser.parse_args()

    main(args.a, args.b, args.output_dir, args.script_path)
