import pandas as pd

data = pd.read_pickle('worldnews_comments.pkl')

data = pd.DataFrame(data)

print(data.head(5))
