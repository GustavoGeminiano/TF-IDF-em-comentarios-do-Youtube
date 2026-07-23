import nltk
import time
import random
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Settings ---
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

STOPWORDS_EN = set(stopwords.words('english'))

API_KEY = "YOUR_API_KEY_HERE"
YOUTUBE = build("youtube", "v3", developerKey=API_KEY)

# --- Functions ---
def clean_text(text):
    """Removes stopwords and converts text to lowercase."""
    return " ".join(word for word in text.lower().split() if word not in STOPWORDS_EN)

def fetch_comments(video_id):
    """Fetches all top-level comments and replies from a video."""
    comments = []
    next_page_token = None

    print("Downloading comments...")

    while True:
        try:
            req = YOUTUBE.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText"
            )
            res = req.execute()

            for item in res["items"]:
                # Top-level comment
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)

                # First-level replies
                for reply in item.get("replies", {}).get("comments", []):
                    comments.append(reply["snippet"]["textDisplay"])

            next_page_token = res.get("nextPageToken")
            if not next_page_token:
                break

            # Delay to avoid hitting rate limits
            time.sleep(random.uniform(0.5, 1.5))

        except Exception as e:
            print(f"Error: {e}. Retrying in 5 seconds...")
            time.sleep(5)

    print(f"{len(comments)} comments downloaded.\n")
    return comments

def search_relevant_comments(comments, query):
    """Calculates similarity and returns the most relevant comments."""
    clean_comments = [clean_text(c) for c in comments]
    clean_query = [clean_text(query)]

    vectorizer = TfidfVectorizer()
    tfidf_docs = vectorizer.fit_transform(clean_comments)
    tfidf_query = vectorizer.transform(clean_query)

    similarities = cosine_similarity(tfidf_query, tfidf_docs)[0]

    results = [
        (c, s) for c, s in zip(comments, similarities) if s > 0
    ]
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# --- Main Execution ---
if __name__ == "__main__":
    link = input("Enter YouTube video link: ").strip()
    video_id = link.replace("https://www.youtube.com/watch?v=", "").split("&")[0]
    query = input("Enter your search query: ").strip()

    comments = fetch_comments(video_id)
    relevant_comments = search_relevant_comments(comments, query)

    if relevant_comments:
        print("\nMost relevant comments:")
        for i, (comment, score) in enumerate(relevant_comments[:5], 1):
            print(f"{i}. [{score:.3f}] {comment}")
    else:
        print("\nNo relevant comments found.")
