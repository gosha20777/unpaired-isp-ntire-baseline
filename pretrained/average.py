import pandas as pd

csv_file = 'baseline_supervised.csv'
output_file = 'result.csv'

try:
    df = pd.read_csv(csv_file)
    print(f"Loaded CSV file: {csv_file}")
except FileNotFoundError:
    print(f"Error: File '{csv_file}' not found.")
    exit()

required_columns = ['PSNR', 'SSIM', 'MS-SSIM', 'LPIPS-ALEX']
if not all(col in df.columns for col in required_columns):
    print(f"Error: CSV file must contain the following columns: {', '.join(required_columns)}")
    exit()

averages = {
    'Average PSNR': df['PSNR'].mean(),
    'Average SSIM': df['SSIM'].mean(),
    'Average MS-SSIM': df['MS-SSIM'].mean(),
    'Average LPIPS': df['LPIPS-ALEX'].mean()
}

print("\n--- Averaged Metrics ---")
for metric, value in averages.items():
    print(f"{metric}: {value:.4f}")

avg_df = pd.DataFrame([averages])
avg_df.to_csv(output_file, index=False)
print(f"\nAveraged metrics saved to '{output_file}' successfully.")
