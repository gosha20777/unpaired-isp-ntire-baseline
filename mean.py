import os
import pandas as pd

def find_best_psnr(csv_folder):
    best_file = None
    best_avg_psnr = float('-inf')

    for file in os.listdir(csv_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(csv_folder, file)
            try:
                df = pd.read_csv(file_path)

                if 'PSNR' in df.columns:
                    avg_psnr = df['PSNR'].mean()

                    if avg_psnr > best_avg_psnr:
                        best_avg_psnr = avg_psnr
                        best_file = file
                else:
                    print(f"Warning: 'PSNR' column not found in {file}")
            except Exception as e:
                print(f"Error processing file {file}: {e}")

    return best_file, best_avg_psnr

if __name__ == "__main__":
    csv_folder = "csv"
    if not os.path.exists(csv_folder):
        print(f"Error: Folder '{csv_folder}' does not exist.")
        exit(1)

    best_file, best_avg_psnr = find_best_psnr(csv_folder)
    if best_file:
        print(f"\nThe file with the best average PSNR is '{best_file}' with a PSNR of {best_avg_psnr:.2f}")
    else:
        print("No valid CSV files found.")
