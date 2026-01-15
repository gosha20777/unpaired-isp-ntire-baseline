import os
import argparse
import time
import subprocess

def main(a, step, process_script, output_dir, script_path):
    for b in range(0, a, step):
        session_name = f"sess_{b}"
        command = (
            f"tmux new-session -d -s {session_name} "
            f"'python3 {process_script} {a} {b} --output_dir {output_dir} --script_path {script_path}; exec bash'"
        )

        print(f"Starting tmux session '{session_name}' for remainder {b}...")
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error starting tmux session '{session_name}': {e}")
        time.sleep(5)

    print("All tmux sessions launched.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, help="The divisor for the condition")
    parser.add_argument('--step', type=int, default=10, help="Step size for the range (default: 10)")
    parser.add_argument('--process_script', type=str, default="process_subset.py", help="Path to the process_subset script")
    parser.add_argument('--output_dir', type=str, default="output", help="Directory containing model files")
    parser.add_argument('--script_path', type=str, default="inference.py", help="Path to the inference script")
    args = parser.parse_args()

    main(args.n, args.step, args.process_script, args.output_dir, args.script_path)