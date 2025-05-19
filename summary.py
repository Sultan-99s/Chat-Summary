import argparse
from typing import List, Tuple, Dict
import re
from collections import Counter
from nltk.corpus import stopwords
import torch
from transformers import pipeline


def parse_chat(path: str):
    """
    Reads the .txt file and returns a list of (speaker, message) tuples.
    """
    messages = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            if line.startswith("User:"):
                text = line[len("User:"):].strip()
                messages.append(("User", text))
            elif line.startswith("AI:"):
                text = line[len("AI:"):].strip()
                messages.append(("AI", text))
            else:
                if messages:
                    speaker, prev = messages[-1]
                    messages[-1] = (speaker, prev + " " + line)
    return messages


def build_pairs(messages):
    """
    Groups the flat list into [(user_msg, ai_msg), …].
    """
    pairs = []
    user_msg = None
    for speaker, text in messages:
        if speaker == "User":
            user_msg = text
        elif speaker == "AI" and user_msg is not None:
            pairs.append((user_msg, text))
            user_msg = None
    return pairs


def compute_stats(pairs):
    """
    Returns a dict with:
      - total_exchanges: number of (user, ai) pairs
      - total_user_msgs: number of User messages
      - total_ai_msgs: number of AI messages
    """
    return {
        "total_exchanges": len(pairs),
        "total_user_msgs": len(pairs),
        "total_ai_msgs": len(pairs)
    }



STOPWORDS = set(stopwords.words("english"))

def top_keywords(pairs, n=5):
    """
    Tokenize all messages, filter stopwords, return top-n words.
    """
    all_text = " ".join(u + " " + a for u, a in pairs).lower()
    # simple tokenization on words
    words = re.findall(r"\b[a-z']+\b", all_text)
    filtered = [w for w in words if w not in STOPWORDS]
    counts = Counter(filtered)
    return [word for word, _ in counts.most_common(n)]


def format_summary(stats, keywords):
    return (
        f"Summary:\n"
        f"- The conversation had {stats['total_exchanges']} exchanges.\n"
        f"- User messages: {stats['total_user_msgs']}, AI messages: {stats['total_ai_msgs']}.\n"
        f"- Most common keywords: {', '.join(keywords)}."
    )



summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",  
    device=0 if torch.cuda.is_available() else -1
)

def llm_summarize(text: str, max_length: int = 50) -> str:
    """
    Returns a very short summary (headline) of the given text.
    """
    result = summarizer(text, max_length=max_length, min_length=5, do_sample=False)
    return result[0]["summary_text"].strip()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Summarize a chat log")
    parser.add_argument("input_file", help="chat.txt")
    args = parser.parse_args()


    messages = parse_chat(args.input_file)
    pairs = build_pairs(messages)
    raw_text = " ".join(f"User: {u} AI: {a}" for u, a in pairs)
    headline = llm_summarize(raw_text, max_length=10)

    stats = compute_stats(pairs)
    keywords = top_keywords(pairs)

    print(format_summary(stats, keywords))
    print("\nTopic & Short Summary:")
    print(f"- {headline}")

