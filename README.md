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


# Usage Examples

# 1. Command-Line Summary
python summary.py chat.txt

# 2. Run Web Interface
export FLASK_APP=app.py
flask run

# Then open in browser:
# http://127.0.0.1:5000

# Command-Line output
![Screenshot from 2025-05-19 13-10-50](https://github.com/user-attachments/assets/578a74a6-11df-4619-8188-f0b7254eac88)


# Web Interface
![Screenshot from 2025-05-19 13-07-30](https://github.com/user-attachments/assets/484219ef-800f-40bf-9c91-1cda3e32f1b1)
