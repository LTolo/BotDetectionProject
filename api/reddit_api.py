import praw
import time
import pandas as pd

# Ersetze mit deinen eigenen Reddit API Anmeldedaten
CLIENT_ID = '0aMNpyoCTXiRFguwg3w61w'
CLIENT_SECRET = 'MwFkNHDYYowE3hbzsvxGmxZGyCNvFg'
USER_AGENT = 'bot_detection_project_v1.0_by_u/Substantial_Salad992'
 
reddit = praw.Reddit(client_id=CLIENT_ID,
                     client_secret=CLIENT_SECRET,
                     user_agent=USER_AGENT)

def fetch_reddit_data(subreddit_name, limit=10):
    """
    Ruft neue Beiträge eines Subreddits ab und formatiert diese.
    Die Spalte 'title' wird in 'Tweet' umbenannt, um mit unserem Modell übereinzustimmen.
    """
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for post in subreddit.new(limit=limit):
        posts.append({
            'Tweet': post.title,  # Umbenennung: title -> Tweet
            'url': post.url,
            'score': post.score,
            'id': post.id,
            'comments': post.num_comments,
            'created': post.created_utc
        })
    return posts

def fetch_reddit_data_with_retry(subreddit_name, limit=10):
    """
    Versucht, Reddit-Daten abzurufen und wiederholt den Abruf im Fehlerfall.
    """
    try:
        posts = fetch_reddit_data(subreddit_name, limit)
    except Exception as e:
        print(f"Error fetching data: {e}")
        time.sleep(5)
        posts = fetch_reddit_data(subreddit_name, limit)
    return posts

def get_reddit_data(subreddit_name='learnpython', limit=10):
    """
    Ruft die Reddit-Daten ab und gibt sie als DataFrame zurück.
    Da keine 'Bot Label'-Information vorliegt, erfolgt hier nur die Datenausgabe.
    """
    posts = fetch_reddit_data_with_retry(subreddit_name, limit)
    df = pd.DataFrame(posts)
    return df

if __name__ == "__main__":
    df = get_reddit_data('learnpython', limit=10)
    print(df[['Tweet']])
