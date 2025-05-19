# Chat Summary & Thematic Analyzer

A simple Python + Flask app that ingests a User–AI conversation, parses and analyzes it, and presents:

- **Message Statistics**: counts of User vs. AI messages  
- **Keyword Extraction**: top‑5 keywords (NLTK stop‑word filtering)  
- **LLM‑Powered Overview**: concise topic heading + 2–3 sentence summary (using `facebook/bart-large-cnn`)  
- **Web Frontend**: paste your conversation into a single‑page HTML/CSS form and see results instantly

---

## Features

1. **Chat Parsing & Stats**  
   - Parses lines beginning with `User:` and `AI:`  
   - Counts total exchanges, User messages, and AI messages  

2. **Keyword Extraction**  
   - Tokenizes text with regex  
   - Filters out NLTK stop words  
   - Ranks the top 5 most frequent words  

3. **LLM Overview**  
   - Uses Hugging Face’s `transformers` pipeline with `facebook/bart-large-cnn`  
   - Prompts the model to produce:
     1. A 3–4 word **Title**  
     2. A 2–3 sentence **Summary**  

4. **Flask Web UI**  
   - Single page with a textarea for your conversation  
   - Displays stats, keywords, title, and summary below  

---

## Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourusername/Chat-Summary.git
   cd Chat-Summary
