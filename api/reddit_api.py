# src/data/reddit_api.py

import praw
import time

# Ersetze mit deinen eigenen Reddit API Anmeldedaten
CLIENT_ID = '0aMNpyoCTXiRFguwg3w61w'
CLIENT_SECRET = 'MwFkNHDYYowE3hbzsvxGmxZGyCNvFg'
USER_AGENT = 'bot_detection_project_v1.0_by_u/Substantial_Salad992'

reddit = praw.Reddit(client_id=CLIENT_ID,
                     client_secret=CLIENT_SECRET,
                     user_agent=USER_AGENT)

def fetch_reddit_data(subreddit_name, limit=10):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for post in subreddit.new(limit=limit):
        posts.append({
            'title': post.title,
            'url': post.url,
            'score': post.score,
            'id': post.id,
            'comments': post.num_comments,
            'created': post.created_utc
        })
    return posts

def fetch_reddit_data_with_retry(subreddit_name, limit=10):
    try:
        posts = fetch_reddit_data(subreddit_name, limit)
    except Exception as e:
        print(f"Error fetching data: {e}")
        time.sleep(5)
        posts = fetch_reddit_data(subreddit_name, limit)
    return posts

if __name__ == "__main__":
    # Beispielaufruf
    posts = fetch_reddit_data('learnpython', limit=10)
    for post in posts:
        print(post['title'])
