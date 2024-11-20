import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import torch
from datasets import Dataset

classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None, device=0 if torch.cuda.is_available() else -1, batch_size = 32)

comments = pd.read_pickle('worldnews_comments.pkl')
comments['score'] = pd.to_numeric(comments['score'], errors='coerce')
comments.dropna(subset=['score'], inplace=True)

comments = comments[comments['score'] > 10]


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

def analyze_sentiment(comments, words):
    # filter comments to only include those that contain the text regardless of case
    comments = comments[comments['full_text'].str.lower().apply(lambda x: any(word.lower() in x for word in words))]
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

biden_keywords = ['biden', 'joe']
biden = {"keywords": biden_keywords, "filename": "biden_sentiment.csv"}

china_keywords = ['china', 'chinese', 'jinping', 'beijing', 'taiwan']
china = {"keywords": china_keywords, "filename": "china_sentiment.csv"}

musk_keywords = ['musk', 'elon', 'tesla', 'spacex']
musk = {"keywords": musk_keywords, "filename": "musk_sentiment.csv"}

lebanon_keywords = ['lebanon', 'beirut', 'hariri', 'hezbollah']
lebanon = {"keywords": lebanon_keywords, "filename": "lebanon_sentiment.csv"}

palestine_keywords = ['israel', 'palestine', 'gaza', 'hamas']
palestine = {"keywords": palestine_keywords, "filename": "palestine_sentiment.csv"}

trump_keywords = ['trump', 'donald']
trump = {"keywords": trump_keywords, "filename": "trump_sentiment.csv"}

ukraine_keywords = ['ukraine', 'kiev', 'kyiv', 'zelenksy']
ukraine = {"keywords": ukraine_keywords, "filename": "ukraine_sentiment.csv"}

for topic in [biden, china, musk, lebanon, palestine, trump, ukraine]:
    print(f"Analyzing sentiment for {topic}")
    sentiment = analyze_sentiment(comments, topic['keywords'])
    sentiment.to_csv(topic['filename'])