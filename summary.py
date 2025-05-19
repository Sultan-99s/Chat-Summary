import argparse
from typing import List, Tuple, Dict
import re
from collections import Counter
from nltk.corpus import stopwords
import torch
from transformers import pipeline


def parse_chat(path: str):
    messages = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            low = line.lower()
            if low.startswith("user:"):
                text = line[line.index(":")+1:].strip()
                messages.append(("User", text))
            elif low.startswith("ai:"):
                text = line[line.index(":")+1:].strip()
                messages.append(("AI", text))
            else:
                # skip completely invalid lines
                continue
    return messages


def build_pairs(messages):
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
    Tokenize all messages, filter stopwords and short tokens, return top-n words.
    """
    # combine user+AI text
    all_text = " ".join(u + " " + a for u, a in pairs)
    # find only real words of length >=2
    words = re.findall(r"[a-zA-Z]{2,}", all_text)
    # lowercase and filter stopwords
    filtered = [w.lower() for w in words if w.lower() not in STOPWORDS]
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

def llm_overview(text: str) -> tuple[str, str]:
    prompt = (
        "Please produce:\n"
        "1. A concise title (3-4 words).\n"
        "2. A very short summary (2-3 sentences).\n\n"
        "Conversation:\n"
        "<<<\n"
        f"{text}\n"
        ">>>\n"
    )
    # Generate combined title + summary
    result = summarizer(
        prompt,
        max_length=100,
        min_length=50,
        do_sample=False
    )[0]["summary_text"].strip()

    # Split into lines and extract parts
    lines = [line.strip() for line in result.split("\n") if line.strip()]
    if len(lines) >= 2:
        title = lines[0].rstrip('.')
        summary = ' '.join(lines[1:])
    else:
        # Fallback: split by first period
        parts = result.split('.', 1)
        title = parts[0].strip().rstrip('.')
        summary = parts[1].strip() if len(parts) > 1 else ''

    return title, summary



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Summarize a chat log")
    parser.add_argument("input_file", help="chat.txt")
    args = parser.parse_args()


    messages = parse_chat(args.input_file)
    pairs = build_pairs(messages)
    # build raw_text as before
    raw_text = " ".join(f"User: {u} AI: {a}" for u, a in pairs)
    title, summary = llm_overview(raw_text)
    stats = compute_stats(pairs)
    keywords = top_keywords(pairs)

    print(format_summary(stats, keywords))
    print("\nTopic & Short Summary:")
    print("\nThe conversation is about " + title + "\n" + summary)

