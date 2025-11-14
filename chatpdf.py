import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import glob
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import hashlib
import json
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Path to PDF files
PDF_FOLDER = "/Users/msf/Desktop/Projects/Financial Summarizer/Files"
CACHE_FILE = "response_cache.json"

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
    st.error("⚠️ OpenAI API key not configured. Please add your API key to the .env file.")

# Email configuration (loaded from .env)
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")  # App-specific password

# Response cache
def load_cache():
    """Load response cache from file"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_cache(cache):
    """Save response cache to file"""
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def get_cache_key(question):
    """Generate a cache key from the question"""
    return hashlib.md5(question.lower().strip().encode()).hexdigest()

# Keywords to identify valuable financial content
VALUABLE_KEYWORDS = [
    'revenue', 'profit', 'loss', 'balance sheet', 'income statement', 
    'cash flow', 'assets', 'liabilities', 'equity', 'earnings',
    'ebitda', 'operating income', 'net income', 'gross profit',
    'financial position', 'consolidated', 'standalone',
    'key financial', 'performance', 'ratio', 'margin',
    'dividend', 'shareholder', 'capital', 'debt',
    'risk', 'management discussion', 'md&a', 'auditor',
    'notes to accounts', 'significant accounting', 'segment'
]

# Keywords to identify sections to ignore
IGNORE_KEYWORDS = [
    'chairman', 'ceo letter', 'message from', 'sustainability',
    'corporate social responsibility', 'csr', 'environment',
    'disclaimer', 'forward-looking', 'corporate governance',
    'board of directors', 'directors report', 'notice',
    'proxy', 'voting', 'general meeting', 'registered office',
    'registrar', 'transfer agent', 'contact', 'branch'
]

def clean_text(text):
    """Remove unwanted characters and clean the text"""
    # Remove page numbers, headers, footers patterns
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Remove standalone page numbers
    text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Remove very short lines (likely headers/footers)
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if len(line.strip()) > 20]
    
    return '\n'.join(cleaned_lines)

def is_valuable_section(text_chunk):
    """Check if a text chunk contains valuable financial information"""
    text_lower = text_chunk.lower()
    
    # Check if it contains ignore keywords
    ignore_count = sum(1 for keyword in IGNORE_KEYWORDS if keyword in text_lower)
    if ignore_count > 2:  # If multiple ignore keywords, skip it
        return False
    
    # Check if it contains valuable keywords or numbers
    valuable_count = sum(1 for keyword in VALUABLE_KEYWORDS if keyword in text_lower)
    has_numbers = bool(re.search(r'\d+[,\d]*\.?\d*', text_chunk))  # Contains numbers
    has_currency = bool(re.search(r'₹|Rs\.|USD|\$|€|£', text_chunk))  # Contains currency
    has_percentage = bool(re.search(r'\d+\.?\d*\s*%', text_chunk))  # Contains percentages
    
    # Keep if has valuable keywords AND (numbers OR currency OR percentage)
    if valuable_count > 0 and (has_numbers or has_currency or has_percentage):
        return True
    
    # Keep if has multiple valuable keywords
    if valuable_count >= 3:
        return True
    
    # Keep if it looks like a financial table (multiple numbers in structured format)
    numbers_count = len(re.findall(r'\d+[,\d]*\.?\d*', text_chunk))
    if numbers_count > 10:  # Likely a financial table
        return True
    
    return False






def get_pdf_text(pdf_paths):
    """Extract and filter text from PDF files given their paths"""
    text = ""
    total_pages = 0
    processed_pages = 0
    
    for pdf_path in pdf_paths:
        try:
            pdf_reader = PdfReader(pdf_path)
            total_pages += len(pdf_reader.pages)
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                
                # Clean the text
                page_text = clean_text(page_text)
                
                # Only include valuable sections
                if is_valuable_section(page_text):
                    text += page_text + "\n\n"
                    processed_pages += 1
                    
        except Exception as e:
            st.warning(f"Error reading {os.path.basename(pdf_path)}: {e}")
    
    st.info(f"📊 Filtered: {processed_pages}/{total_pages} pages contain valuable financial data")
    return text


def load_pdfs_from_folder(folder_path):
    """Load all PDF files from the specified folder"""
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    # Filter out .crdownload files (incomplete downloads)
    pdf_files = [f for f in pdf_files if not f.endswith('.crdownload')]
    return pdf_files



def get_text_chunks(text):
    """Split text into optimized chunks for comprehensive analysis"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,  # Larger chunks to capture complete financial tables
        chunk_overlap=400,  # More overlap to preserve context
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    
    # Further filter chunks to keep only those with financial data
    valuable_chunks = [chunk for chunk in chunks if is_valuable_section(chunk)]
    
    st.info(f"📦 Created {len(valuable_chunks)} relevant chunks (filtered from {len(chunks)} total)")
    return valuable_chunks


def get_vector_store(text_chunks):
    """Create vector store using OpenAI embeddings"""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  # Cost-effective, high quality
        openai_api_key=OPENAI_API_KEY
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain(is_comprehensive=False):
    """Get ChatGPT model for financial analysis"""
    # Use GPT-4 for comprehensive reports, GPT-3.5 for quick queries
    model_name = "gpt-4o" if is_comprehensive else "gpt-3.5-turbo"
    max_tokens = 4096 if is_comprehensive else 2048
    
    # Temperature set to 0 for deterministic, consistent responses
    model = ChatOpenAI(
        model=model_name,
        temperature=0.0,  # Zero temperature = consistent, deterministic responses
        max_tokens=max_tokens,
        openai_api_key=OPENAI_API_KEY,
        model_kwargs={
            "top_p": 0.1,  # Reduce randomness
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
    )
    return model


def user_input(user_question):
    # Check cache first
    cache = load_cache()
    cache_key = get_cache_key(user_question)
    
    if cache_key in cache:
        st.success("✅ Retrieved from cache (instant response)")
        st.write("**Reply:**", cache[cache_key])
        return
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Detect if this is a comprehensive report request
    is_comprehensive_report = any(keyword in user_question.lower() for keyword in [
        'comprehensive', 'assessment report', 'financial assessment', 'detailed report',
        'generate a report', 'create a report', 'full report', 'complete report'
    ])
    
    # For comprehensive reports, retrieve more documents
    k_docs = 20 if is_comprehensive_report else 5
    score_threshold = 2.5 if is_comprehensive_report else 1.5
    
    if is_comprehensive_report:
        st.warning("🎯 Comprehensive Report Mode Activated - Using GPT-4o for detailed, accurate analysis")
    
    # Get more relevant documents with score threshold for better accuracy
    docs_with_scores = new_db.similarity_search_with_score(user_question, k=k_docs)
    
    # Filter by relevance score (lower score = more relevant in FAISS)
    relevant_docs = [doc for doc, score in docs_with_scores if score < score_threshold]
    
    # If no highly relevant docs, use top documents
    if not relevant_docs:
        top_n = 10 if is_comprehensive_report else 3
        relevant_docs = [doc for doc, score in docs_with_scores[:top_n]]
    
    st.info(f"🔍 Using {len(relevant_docs)} relevant document chunks")
    
    model = get_conversational_chain(is_comprehensive=is_comprehensive_report)
    
    # Create context from documents with clear separation
    context_parts = []
    for i, doc in enumerate(relevant_docs, 1):
        context_parts.append(f"--- Source {i} ---\n{doc.page_content}")
    
    context = "\n\n".join(context_parts)
    
    # Enhanced prompt for comprehensive financial reports
    if is_comprehensive_report:
        prompt = f"""You are an expert financial analyst and investment research professional. Your ABSOLUTE PRIORITY is DATA FIDELITY - only use exact figures from the source documents.

🚨 CRITICAL ANTI-HALLUCINATION RULES:

1. 🎯 BUSINESS SEGMENT ACCURACY (MANDATORY)
   - Extract segment names EXACTLY as written in the Annual Report
   - DO NOT invent segments that don't exist in the documents
   - For ITC: Only use segments explicitly mentioned (FMCG, Hotels, Agri Business, Paperboards & Packaging)
   - NEVER include: Cement, Electronics, Engineering, or any segment not in source documents
   - If a segment is not mentioned, DO NOT create data for it
   - Match official terminology (e.g., "Agri Business" not "Agriculture")

2. 📊 FINANCIAL DATA FIDELITY (ZERO TOLERANCE FOR ERRORS)
   - Extract ONLY actual published numbers from the provided context
   - If FY 2025 data is not in the documents, state: "FY 2025 data not yet published. Using FY 2024 actual data."
   - NEVER generate, estimate, or hallucinate revenue/profit figures
   - Cross-verify: Revenue > EBITDA > PBT > PAT (this hierarchy must hold)
   - If a number seems wrong, flag it: "Data requires verification - appears inconsistent"

3. 📈 RATIO AND MARGIN CONSISTENCY (MATHEMATICAL VALIDATION)
   - Calculate margins: EBITDA Margin = (EBITDA ÷ Revenue) × 100
   - Calculate margins: PAT Margin = (PAT ÷ Revenue) × 100
   - Verify YoY % = ((Current - Previous) ÷ Previous) × 100
   - For ITC specifically: Debt-Equity should be ~0.00 to 0.05 (virtually debt-free)
   - If extracted ratio contradicts known financial profile, add note: "[Ratio appears unusual for this company's profile]"

4. 🗣️ ENHANCED DISCLAIMER REQUIREMENT
   - If any data is estimated/projected, add clear disclaimer
   - State explicitly: "Note: [Specify which figures] are estimates/projections, NOT published results"
   - If fiscal year data unavailable, state: "Official FY[Year] results not available in source documents"

5. 🔍 DATA VALIDATION PROTOCOL (EXECUTE BEFORE OUTPUT)
   Step 1: Verify all segment names exist in source documents
   Step 2: Check Revenue > EBITDA > PBT > PAT hierarchy
   Step 3: Validate margin calculations match provided numbers
   Step 4: Confirm ratios align with company's known financial health
   Step 5: Flag any inconsistencies with [DATA VERIFICATION NEEDED] tag

REASONING PROCESS - Follow this step-by-step:

1. DATA EXTRACTION FIRST
   - Locate the Consolidated Statement of Profit & Loss in the context
   - Extract EXACT figures for Revenue from Operations, EBITDA, PBT, PAT
   - Find Balance Sheet for ratios: Debt-Equity, Current Ratio, Net Worth
   - Locate segment revenue table for business-wise performance
   - Extract previous year comparisons from the same tables

2. SEGMENT VALIDATION
   - List all segments mentioned in "Segment Information" or "Segment Revenue" section
   - Use ONLY these segment names - do not add others
   - Extract segment-wise revenue, EBIT, margins from the table
   - If segment data incomplete, state "Segment details not fully disclosed"

3. FINANCIAL HEALTH ASSESSMENT
   - For debt-free companies (like ITC): Debt-Equity near 0.00
   - For FMCG companies: High margins (EBITDA 30-40%, PAT 20-30%)
   - For capital-intensive: Lower margins, higher debt
   - Match your extracted ratios against these sector norms

4. PLAUSIBILITY CHECK
   - ITC typical revenue range: ₹60,000-70,000 Cr (FY2024-2025)
   - If you see ₹1,43,000 Cr or ₹2,53,000 Cr - THIS IS WRONG
   - Cross-check: Does EBITDA exceed Revenue? IMPOSSIBLE - flag error
   - Does PAT exceed Revenue? IMPOSSIBLE - flag error

5. REPORT STRUCTURE - Follow this EXACT format:

# Assessment Note on FY[Year]

**Date:** November 7, 2025  
**Company:** [Extract exact name from document - e.g., "ITC Limited"]  
**Credit Rating:** [Extract if available, else "Not Disclosed"]  
**Report Status:** [✅ Based on Published Annual Report / ⚠️ Contains Estimates]

## 1. Summary / Overview
- Provide 3-4 bullet points based ONLY on data found in documents
- Include total revenue and profit trends with EXACT numbers
- Mention key developments ONLY if stated in context
- Note macroeconomic factors ONLY if mentioned in MD&A

## 2. Key Positives & Negatives

**Positive Factors:**
- [List 4-5 specific strengths FOUND in the report - not generic]
- Use exact quotes or data points from context

**Negative Factors:**
- [List 3-4 challenges MENTIONED in the documents]
- Reference specific sections (e.g., "As per MD&A...")

## 3. Consolidated Financial Performance (YoY)

| Particulars | FY [Current] | FY [Previous] | YoY % |
|-------------|--------------|---------------|-------|
| Revenue from Operations | ₹ [EXACT] Cr | ₹ [EXACT] Cr | [CALCULATED]% |
| EBITDA | ₹ [EXACT] Cr | ₹ [EXACT] Cr | [CALCULATED]% |
| PBT (Profit Before Tax) | ₹ [EXACT] Cr | ₹ [EXACT] Cr | [CALCULATED]% |
| PAT (Profit After Tax) | ₹ [EXACT] Cr | ₹ [EXACT] Cr | [CALCULATED]% |
| EBITDA Margin (%) | [CALCULATED]% | [CALCULATED]% | +/- [X] bps |
| PAT Margin (%) | [CALCULATED]% | [CALCULATED]% | +/- [X] bps |

**Comments:**
- Use ONLY information from the documents to explain trends
- If reason not stated, write "Specific drivers not disclosed in available documents"

## 4. Financial Ratios

| Metric | FY [Current] | FY [Previous] | Interpretation |
|--------|--------------|---------------|----------------|
| Debt-Equity Ratio | [EXACT or "Not disclosed"] | [EXACT or "Not disclosed"] | [Based on extracted value] |
| Interest Coverage Ratio | [EXACT] | [EXACT] | [Based on actual figures] |
| Current Ratio | [EXACT] | [EXACT] | [Liquidity assessment from data] |
| Return on Equity (%) | [EXACT]% | [EXACT]% | [ROE analysis] |
| Net Worth | ₹ [EXACT] Cr | ₹ [EXACT] Cr | [Capital base commentary] |

**Analysis:**
Provide 2-3 sentences STRICTLY based on the extracted ratios. Do not speculate.

## 5. Standalone Financial Performance (If applicable)

[ONLY include if "Standalone" financials are separately provided in the documents]
[If not found, write: "Standalone financial data not separately disclosed in available documents"]

## 6. Segment Information

⚠️ CRITICAL: Use ONLY segments explicitly named in the "Segment Revenue" or "Business Segment" section

| Segment | FY [Current] Revenue | FY [Previous] Revenue | YoY % | EBIT Margin | Key Drivers |
|---------|---------------------|----------------------|-------|-------------|-------------|
| [EXACT NAME] | ₹ [EXACT] Cr | ₹ [EXACT] Cr | [CALC]% | [EXACT]% | [From context ONLY] |

**Segment Validation Check:**
✓ All segment names verified from source document section: [Page/Section reference]
✓ No segments added beyond those officially reported

## 7. Management Discussion Highlights

Extract ONLY from MD&A or Director's Report sections found in context:
- **Business Environment:** [Quote or paraphrase from document]
- **Cost Dynamics:** [ONLY if mentioned in context]
- **Strategic Initiatives:** [ONLY stated initiatives, not assumptions]
- **Operational Challenges:** [ONLY if discussed in report]
- **Capital Allocation:** [ONLY if dividend/capex mentioned]

[If section not found: "Management Discussion section not available in provided context"]

## 8. Future Outlook / Guidance

[ONLY include forward-looking statements FOUND in the documents]
[If no guidance section exists, state: "Formal guidance not provided in available documents"]

Based on available commentary:
1. [Quote or reference actual statement]
2. [Do not speculate - use document text only]

## 9. References

- Annual Report FY[Year]: [If mentioned in context]
- Company: [Exact company name]
- Report Type: Consolidated Financial Statements

---

## ⚠️ DATA FIDELITY DISCLAIMER

**Source Data Status:**
- ✅ All financial figures extracted directly from source documents
- ⚠️ [If applicable] FY[Year] data not yet published - using FY[Previous Year] actual data
- ⚠️ [If applicable] The following figures are ESTIMATES/PROJECTIONS, NOT published results: [List any]

**Validation Summary:**
- Segment names verified against official Annual Report terminology
- Mathematical consistency checked: Revenue > EBITDA > PBT > PAT ✓
- Margin calculations verified: EBITDA Margin = (EBITDA/Revenue)×100 ✓
- Ratios cross-checked against company's known financial profile ✓
- No hallucinated or estimated figures included without explicit disclosure ✓

**Important Notice:**
This report is based solely on information available in the provided financial documents. Any figures marked as estimates or projections are NOT official published results and should be treated with appropriate caution. Users should verify all data against official company filings before making investment decisions.

---

## AI Consistency Check

[Provide 3-4 sentences summarizing]:
- ✓ Data source: [Confirm all data from provided context only]
- ✓ Segment validation: [List segments used and confirm they match source]
- ✓ Calculation verification: [Confirm YoY%, margins calculated correctly]
- ✓ Plausibility check: [Confirm no impossible figures like EBITDA > Revenue]
- ✓ Known company profile: [E.g., "ITC's debt-free status confirmed - Debt-Equity ratio [X] aligns with expectations"]
- ⚠️ [Any data gaps or missing information noted]

---

6. QUALITY ASSURANCE RULES
   - Extract EXACT numbers from context - do not round or estimate
   - Use ₹ Cr (Crores) for Indian companies
   - If data not found, write "Data not disclosed in available documents" - DO NOT GUESS
   - Verify segment total revenue ≤ consolidated revenue
   - For ITC: Debt-Equity should be ~0.00 (virtually debt-free company)
   - Remove outdated references (COVID impact beyond FY2023 unless mentioned)

Context from Financial Reports (Use ONLY this data - NOTHING ELSE):
{context}

User's Report Request:
{user_question}

Generate the COMPLETE professional assessment report. REMEMBER: Data fidelity is paramount - only use figures from the context. Flag any inconsistencies. Add disclaimers for any estimated data:"""
    else:
        # Standard prompt for regular questions
        prompt = f"""You are a financial analyst. Answer the question based ONLY on the provided context from financial reports.

Instructions:
- Be precise and factual
- Use exact numbers and dates from the context
- If asked for financial data, present it in a clear format using markdown tables
- If information is not in the context, say "Information not available in the documents"
- Do not make assumptions or estimates
- Always cite the source number when presenting data

Context:
{context}

Question: {user_question}

Answer:"""
    
    response = model.invoke(prompt)
    answer = response.content
    
    # Cache the response
    cache[cache_key] = answer
    save_cache(cache)
    
    st.write("**Reply:**", answer)




def main():
    st.set_page_config("Chat PDF")
    st.header("📊 Financial Summarizer - Powered by ChatGPT 🤖")
    
    # Show API key status
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
        st.error("⚠️ **OpenAI API Key Required!** Please add your API key to the `.env` file. See `OPENAI_SETUP_GUIDE.md` for instructions.")
        st.info("� Get your API key at: https://platform.openai.com/api-keys")
        return
    
    # Check if vector store exists
    vector_store_exists = os.path.exists("faiss_index")
    
    # Sidebar information
    with st.sidebar:
        st.title("📚 PDF Management")
        
        # Load PDFs from folder
        all_pdf_files = load_pdfs_from_folder(PDF_FOLDER)
        
        if all_pdf_files:
            st.success(f"✅ Found {len(all_pdf_files)} PDF files")
            
            # PDF Selection
            st.subheader("Select PDFs to Process")
            selected_pdfs = []
            
            # Add "Select All" checkbox
            select_all = st.checkbox("Select All PDFs", value=True)
            
            if select_all:
                selected_pdfs = all_pdf_files
            else:
                # Individual checkboxes for each PDF
                for pdf in all_pdf_files:
                    pdf_name = os.path.basename(pdf)
                    if st.checkbox(pdf_name, key=pdf_name):
                        selected_pdfs.append(pdf)
            
            st.divider()
            st.info(f"📄 Selected: {len(selected_pdfs)} PDF(s)")
            
            # Process button
            if st.button("🔄 Process Selected PDFs", type="primary"):
                if selected_pdfs:
                    with st.spinner(f"Processing {len(selected_pdfs)} PDF(s)... Please wait."):
                        try:
                            raw_text = get_pdf_text(selected_pdfs)
                            text_chunks = get_text_chunks(raw_text)
                            get_vector_store(text_chunks)
                            st.success(f"✅ {len(selected_pdfs)} PDF(s) processed successfully!")
                            st.balloons()
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                else:
                    st.warning("⚠️ Please select at least one PDF!")
            
            st.divider()
            
            if vector_store_exists:
                st.success("✅ PDFs processed and ready!")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🗑️ Clear Data"):
                        import shutil
                        if os.path.exists("faiss_index"):
                            shutil.rmtree("faiss_index")
                        st.success("Cleared!")
                        st.rerun()
                
                with col2:
                    if st.button("🧹 Clear Cache"):
                        if os.path.exists(CACHE_FILE):
                            os.remove(CACHE_FILE)
                        st.success("Cache cleared!")
                        st.rerun()
        else:
            st.warning("⚠️ No PDF files found!")
            st.info("💡 Run `python pdfscrapper.py` to download PDFs")
    
    # Main area - Question input
    if vector_store_exists:
        st.info("💬 Ask me anything about the financial reports!")
        
        # Add tip for comprehensive reports
        with st.expander("📋 Generate Comprehensive Report", expanded=False):
            st.markdown("""
            **For detailed financial assessment reports:**
            1. Open `quick_report_template.txt` file in your project folder
            2. Copy the entire template
            3. Paste it in the query box below
            4. The system will automatically use **GPT-4o** for accurate analysis
            5. Wait 10-30 seconds for the complete report
            
            **Features:**
            - ✅ Uses **OpenAI GPT-4o** - Superior accuracy for financial data
            - ✅ Retrieves 20+ relevant document chunks
            - ✅ Generates up to 4096 tokens for detailed reports
            - ✅ Extracts EXACT numbers from source documents
            - ✅ Much faster and more accurate than Ollama models
            - ✅ Better at understanding financial terminology and context
            
            **Quick Queries:** Use GPT-3.5 Turbo (fast, cost-effective)
            **Comprehensive Reports:** Use GPT-4o (most accurate, detailed)
            
            **Tip:** You MUST re-process PDFs after downloading for best results.
            Click "Process Selected PDFs" in the sidebar before generating reports.
            """)
        
        user_question = st.text_area(
            "Your Question:", 
            placeholder="e.g., What was the total revenue in 2024?\n\nOr paste the comprehensive report template here...",
            height=150,
            help="Enter your question or paste a formatted query. Supports multiple lines."
        )
        
        if user_question:
            # Detect comprehensive report
            is_comprehensive = any(keyword in user_question.lower() for keyword in [
                'comprehensive', 'assessment report', 'generate a report'
            ])
            
            if is_comprehensive:
                with st.spinner("📊 Generating comprehensive report... This may take 30-60 seconds..."):
                    user_input(user_question)
            else:
                with st.spinner("🤔 Analyzing documents..."):
                    user_input(user_question)
    else:
        st.warning("⚠️ Please select and process PDFs from the sidebar to start chatting.")
    
    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; padding: 20px; color: #666;'>
            <p style='margin: 5px 0; font-size: 14px;'>
                <strong>Created by Mohammed Saad Fazal</strong>
            </p>
            <p style='margin: 5px 0; font-size: 12px;'>
                📧 <a href='mailto:mdsaad7803@gmail.com' style='color: #0066cc; text-decoration: none;'>mdsaad7803@gmail.com</a>
            </p>
            <p style='margin: 5px 0; font-size: 11px; color: #999;'>
                Financial Summarizer | AI-Powered PDF Analysis
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )



if __name__ == "__main__":
    main()