import pandas as pd

chunksize = 10**6  # Adjust based on your memory limits
output_file = 'comments_filtered/worldnews_comments.csv'
total_rows = 0

for i, file in enumerate(['comments_filtered/worldnews_comments_1_4_2024.csv', 
                            'comments_filtered/worldnews_comments_5_7_2024.csv',
                            'comments_filtered/worldnews_comments_8_2024.csv', 
                            'comments_filtered/worldnews_comments_9_2024.csv']):
    # Process each file in chunks
    for chunk in pd.read_csv(file, chunksize=chunksize):
        total_rows += len(chunk)

print(f'Total number of rows in the output file: {total_rows}')