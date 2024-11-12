import pandas as pd
from transformers import pipeline
from tqdm import tqdm

classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

comments = pd.read_pickle('worldnews_comments.pkl')
comments['score'] = pd.to_numeric(comments['score'], errors='coerce')
comments.dropna(subset=['score'], inplace=True)

comments = comments[comments['score'] > 50]


pattern = r'/r/worldnews/comments/\w+/([^/]+/\w+)'

comments['og_post'] = comments['link'].str.extract(pattern, expand=False)
comments['og_post'] = comments['og_post'].str.replace('_', ' ')

# Remove rows where no match was found
comments = comments.dropna(subset=['og_post'])

comments['full_text'] = comments['og_post'] + ' ' + comments['body']
comments.dropna(subset=['full_text'], inplace=True)

def adjust_sentiment_data(sentiment):
    flattened_data = {item['label']: item['score'] for item in sentiment[0]}
    return pd.DataFrame([flattened_data])

def analyze_sentiment(comments, text):
    # filter comments to only include those that contain the text regardless of case
    text = text.lower()
    comments = comments[comments['full_text'].str.lower().str.contains(text)]
    comments['body'] = comments['body'].str[:512]

    sentiment = pd.DataFrame()

    for i in tqdm(range(len(comments))):
        try:
            cur_sentiment = classifier(comments['body'].iloc[i])
            cur_sentiment = adjust_sentiment_data(cur_sentiment)
            cur_sentiment['comment'] = comments['body'].iloc[i]
            cur_sentiment['created'] = pd.to_datetime(comments['created'].iloc[i])

            sentiment = pd.concat([sentiment, cur_sentiment])
        except Exception as e:
            print(e)
            print(comments['body'].iloc[i])
            continue


    return sentiment

sentiment = analyze_sentiment(comments, 'trump')

sentiment.to_csv("trump_sentiment.csv")


