import praw

# Initialize the Reddit API client
reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    user_agent="YOUR_USER_AGENT",
    username="YOUR_REDDIT_USERNAME",
    password="YOUR_REDDIT_PASSWORD"
)

# Verify that we've successfully logged in
print(f"Authenticated as: {reddit.user.me()}")

# Now you can use the 'reddit' object to interact with the Reddit API
# For example, to get the top posts from a subreddit:
# subreddit = reddit.subreddit("python")
# for post in subreddit.hot(limit=10):
#     print(post.title)
