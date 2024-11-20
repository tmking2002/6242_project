import pandas as pd

data = pd.read_csv('palestine_sentiment.csv')

# randomly select half of the rows

# data = data.sample(frac=0.5)

# sort by date (created column)
data['created'] = pd.to_datetime(data['created'])

data = data.sort_values(by='created')

data.to_csv('palestine_sentiment.csv', index=False)
