import praw
import time
import pandas as pd

# Ersetze mit deinen eigenen Reddit API Anmeldedaten
CLIENT_ID = 'YourClientID'                                          #You need your own Reddit API
CLIENT_SECRET = 'YourClientSecret'                                  #You need your own Reddit API
USER_AGENT = 'YourUseerAgent'                                       #You need your own Reddit API
 
reddit = praw.Reddit(client_id=CLIENT_ID,
                     client_secret=CLIENT_SECRET,
                     user_agent=USER_AGENT)

def fetch_reddit_data(subreddit_name, limit=10):
    """
    Ruft Beiträge eines Subreddits ab.
    Es werden 10 Beiträge mit .new() und 10 Beiträge mit .hot() abgefragt,
    jeweils versehen mit einem 'source'-Feld, das angibt, ob es sich um
    neue oder heiße Beiträge handelt.
    """
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    
    # 10 neueste Beiträge
    for post in subreddit.new(limit=limit):
        posts.append({
            'Tweet': post.title,  # Umbenennung: title -> Tweet
            'url': post.url,
            'score': post.score,
            'id': post.id,
            'comments': post.num_comments,
            'created': post.created_utc,
            'source': 'new'
        })
    
    # 10 Hot-Beiträge
    for post in subreddit.hot(limit=limit):
        posts.append({
            'Tweet': post.title,
            'url': post.url,
            'score': post.score,
            'id': post.id,
            'comments': post.num_comments,
            'created': post.created_utc,
            'source': 'hot'
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
    Ruft Reddit-Daten ab und gibt sie als DataFrame zurück.
    Dabei werden 10 neue und 10 Hot-Beiträge abgefragt.
    """
    posts = fetch_reddit_data_with_retry(subreddit_name, limit)
    df = pd.DataFrame(posts)
    return df

if __name__ == "__main__":
    df = get_reddit_data('learnpython', limit=10)
    # Ausgabe der gruppierten Ergebnisse
    if 'source' in df.columns:
        for source, group in df.groupby("source"):
            print(f"\n--- Beiträge aus '{source}' ---")
            print(group[['Tweet']])
    else:
        print(df[['Tweet']])
