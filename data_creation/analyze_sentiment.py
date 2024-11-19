import pandas as pd
from transformers import pipeline
from tqdm import tqdm

model_id = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

classifier = pipeline("sentiment-analysis", model=model_id)

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

def analyze_sentiment(comments, text):
    # filter comments to only include those that contain the text regardless of case
    text = text.lower()
    comments = comments[comments['full_text'].str.lower().str.contains(text)]
    comments['body'] = comments['body'].str[:512]

    sentiment = pd.DataFrame()

    for i in tqdm(range(len(comments))):
        try:
            cur_sentiment = classifier(comments['body'].iloc[i])

            cur_sentiment = pd.DataFrame(cur_sentiment)
            cur_sentiment['comment'] = comments['body'].iloc[i]
            cur_sentiment['created'] = pd.to_datetime(comments['created'].iloc[i])

            sentiment = pd.concat([sentiment, cur_sentiment])
        except Exception as e:
            print(e)
            print(comments['body'].iloc[i])
            continue


    return sentiment

sentiment = analyze_sentiment(comments, 'ukraine')

# first_comment = comments['full_text'].iloc[0]
# print(first_comment)

# sentiment = classifier(first_comment)
# print(sentiment)

sentiment["positive"] = sentiment.apply(lambda row: row["score"] if row["label"] == "POSITIVE" else 1 - row["score"], axis=1)
sentiment["negative"] = sentiment.apply(lambda row: row["score"] if row["label"] == "NEGATIVE" else 1 - row["score"], axis=1)

# Keep only the desired columns
sentiment = sentiment[["positive", "negative", "comment", "created"]]

sentiment.to_csv("ukraine_sentiment.csv")


