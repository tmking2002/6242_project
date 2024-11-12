import pandas as pd

comments = pd.read_pickle('reddit_data.pkl')

# Ensure comments is a DataFrame
if isinstance(comments, list):
    comments = pd.DataFrame(comments)

# get first ten comments
comments = comments.head(10)

comments.to_csv('test.csv', index=False)