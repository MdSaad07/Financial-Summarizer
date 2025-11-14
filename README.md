# Financial Summarizer

Financial Summarizer is a local Streamlit application that downloads, processes and analyzes company annual/quarterly PDF reports and generates professional, investor-grade financial assessment reports.

Core features
- Automatic PDF scraping (multi-URL support)
- PDF text extraction and content filtering to keep financial tables and statements
- FAISS vector store for semantic retrieval of relevant document chunks
- Chat interface for asking questions and generating full structured financial reports
- Anti-hallucination rules: segment validation, numeric verification, and AI Consistency Check
- Uses OpenAI (GPT-3.5/GPT-4o) for accurate extraction and analysis (configurable)

Table of contents
- [Quick start](#quick-start)
- [Requirements](#requirements)
- [Setup](#setup)
- [Run the app](#run-the-app)
- [How it works](#how-it-works)
- [Templates & prompts](#templates--prompts)
- [Files in this repo](#files-in-this-repo)
- [Troubleshooting](#troubleshooting)
- [Security & costs](#security--costs)
- [Contributing](#contributing)
- [Author](#author)

## Quick start

1. Create and activate a Python virtual environment (recommended).
2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Add your OpenAI API key to `.env` (see `OPENAI_SETUP_GUIDE.md`).
4. Place company PDF reports into the `Files/` folder or run the scraper.
5. Start the app:

```bash
streamlit run chatpdf.py
```

Open the app at `http://localhost:8501` and use the sidebar to select and process PDFs first.

## Requirements
- Python 3.11+
- See `requirements.txt` for Python packages. Key packages:
	- streamlit
	- openai, langchain-openai, tiktoken
	- PyPDF2
	- faiss-cpu
	- selenium (for the scraper)

## Setup

1. Copy `.env` and fill values (or edit the existing `.env`):

```env
OPENAI_API_KEY=sk-...
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=you@example.com
SENDER_PASSWORD=app_password_here
```

2. If you plan to scrape PDFs from websites, install a Chrome WebDriver or use `webdriver-manager` (already in `requirements.txt`).

3. Recommended: create a small test set of PDFs in `Files/` and click **Process Selected PDFs** in the app sidebar to build the FAISS index.

## Run the app

1. Start Streamlit (see Quick start):

```bash
streamlit run chatpdf.py
```

2. In the sidebar:
- Select PDFs (or Select All)
- Click **Process Selected PDFs** to extract text, create chunks and build the FAISS index

3. In the main area:
- Use the multi-line query box for complex, formatted prompts (see `quick_report_template.txt`)
- Short questions work fine in the same box

4. For comprehensive reports, open the expander **"📋 Generate Comprehensive Report"** for instructions and paste the template.

## How it works

High-level flow:

1. PDF text extraction (PyPDF2) with cleaning and filtering to keep financial sections
2. Text splitting into chunks (RecursiveCharacterTextSplitter) and filtering for relevance
3. Embeddings created with OpenAI `text-embedding-3-small` and stored in FAISS
4. User queries are converted to embeddings and similarity-searched against FAISS
5. Top relevant chunks are passed to ChatGPT (GPT-3.5 for quick queries / GPT-4o for comprehensive reports)
6. Model responses are validated with anti-hallucination rules and returned with an **AI Consistency Check**

## Templates & prompts

- `quick_report_template.txt` - a compact template to paste for generating full assessment notes
- `report_template.txt` - original, more detailed template
- `test_prompts.txt` - useful prompts for validating correctness and anti-hallucination checks

Best practice: Always click **Process Selected PDFs** after adding or updating PDFs. Clear the cache if you reprocess documents.

## Files in this repo
- `chatpdf.py` - Streamlit app and main logic
- `pdfscrapper.py` - Selenium-based scraper (download reports from `urls.txt`)
- `main.py` - Orchestrator script to run scraper + app
- `requirements.txt` - Python dependencies
- `Files/` - Folder for downloaded PDFs
- `faiss_index/` - Saved FAISS index (created after processing)
- `response_cache.json` - Cached chat responses
- `quick_report_template.txt` - Template to paste into chatbot
- `test_prompts.txt` - Prompts for testing
- `OPENAI_SETUP_GUIDE.md` - How to get and configure OpenAI key
- `ANTI_HALLUCINATION_IMPROVEMENTS.md` - Details on anti-hallucination rules and checks
- `INTELLIGENT_ANALYSIS_FEATURES.md` - Feature notes

## Troubleshooting

- If Streamlit shows **OpenAI API Key Required**: set `OPENAI_API_KEY` in `.env` and restart the app
- If results seem incomplete:
	- Ensure PDFs were processed (FAISS index exists)
	- Increase `k_docs` (number of chunks retrieved) in `chatpdf.py` if needed
	- Reprocess PDFs after any updates and clear cache
- If you hit OpenAI rate limits or billing issues: check your OpenAI dashboard and set usage limits

## Security & costs

- Keep the `.env` file private and out of source control (add to `.gitignore`)
- Monitor OpenAI usage in your account and set a monthly spending cap
- Use GPT-3.5 for cheap quick queries; reserve GPT-4o for important reports

## Contributing

Contributions welcome. Please open issues or pull requests on GitHub. For major changes, open an issue to discuss first.

## Author

Created by Mohammed Saad Fazal — mdsaad7803@gmail.com

# Financial-Summarizer