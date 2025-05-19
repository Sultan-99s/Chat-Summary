from flask import Flask, render_template, request
from summary import parse_chat, build_pairs, compute_stats, top_keywords, llm_overview

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = {}
    if request.method == "POST":
        convo_text = request.form["conversation"]
        # write to temp file or parse lines directly
        # we’ll parse lines from the textarea:
        lines = [line for line in convo_text.splitlines() if line.strip()]
        # simulate parse_chat output:
        messages = []
        for line in lines:
            if line.lower().startswith("user:"):
                messages.append(("User", line.split(":",1)[1].strip()))
            elif line.lower().startswith("ai:"):
                messages.append(("AI",   line.split(":",1)[1].strip()))
        pairs = build_pairs(messages)

        if pairs:
            stats     = compute_stats(pairs)
            keywords  = top_keywords(pairs)
            raw_text  = " ".join(f"User: {u} AI: {a}" for u, a in pairs)
        else:
            # fallback on messages
            stats    = None
            keywords = top_keywords_from_text(" ".join(t for _,t in messages))
            raw_text = " ".join(t for _,t in messages)

        title, summary = llm_overview(raw_text)

        result = {
            "stats":    stats,
            "keywords": keywords,
            "title":    title,
            "summary":  summary
        }

    return render_template("index.html", result=result)
