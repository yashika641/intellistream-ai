import pandas as pd

# Load the data from the sheet
df = pd.read_csv(r'C:\Users\palya\Desktop\intellistream\intellistream-ai\docs\netflix_news.csv')

# Convert the 'date' column to datetime objects
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Drop rows where 'date' is NaT (invalid dates)
df.dropna(subset=['date'], inplace=True)

# Define the stock columns
stock_cols = ['stock_open', 'stock_close', 'stock_high', 'stock_low', 'stock_volume']

# Convert stock columns to numeric, coercing errors to NaN
for col in stock_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows that have NaN in any of the stock_cols after conversion
df.dropna(subset=stock_cols, inplace=True)

# Group by 'date' and 'sentiment', calculate the average of stock columns and aggregate headlines
grouped_df = df.groupby(['date', 'sentiment']).agg(
    avg_stock_open=('stock_open', 'mean'),
    avg_stock_close=('stock_close', 'mean'),
    avg_stock_high=('stock_high', 'mean'),
    avg_stock_low=('stock_low', 'mean'),
    avg_stock_volume=('stock_volume', 'mean'),
    news_headlines=('headline', lambda x: '; '.join(x.tolist()))  # Join all headlines for each group
).reset_index()

# Make a copy to safely modify
final_output = grouped_df.copy()

# Convert 'date' column to string for display
final_output['date'] = final_output['date'].dt.strftime('%Y-%m-%d')

# Save the full aggregated data to CSV
final_output.to_csv('aggregated_netflix_news_stock.csv', index=False)

print("✅ Aggregation complete for full CSV!")
