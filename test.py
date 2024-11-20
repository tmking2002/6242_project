import pandas as pd

data = pd.read_csv('palestine_sentiment.csv')

# randomly select half of the rows

data = data.sample(frac=0.5)

data.to_csv('palesine_sentiment.csv', index=False)
