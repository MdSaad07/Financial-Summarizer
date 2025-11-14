from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time
import re

# Keywords to identify latest annual/quarterly reports
LATEST_KEYWORDS = [
    '2025', '2024', '2023',  # Recent years
    'latest', 'recent', 'current',
    'annual report 2024', 'annual report 2025',
    'q4 2024', 'q3 2024', 'q2 2024', 'q1 2024',
    'fy 2024', 'fy 2025', 'fy24', 'fy25'
]

def extract_year_from_text(text):
    """Extract year from text/link"""
    years = re.findall(r'20\d{2}', text)
    if years:
        return max([int(y) for y in years])  # Return the most recent year
    return 0

def is_latest_report(link_text, href):
    """Check if the link is for a latest annual/quarterly report"""
    combined_text = (link_text + " " + href).lower()
    
    # Check for latest keywords
    for keyword in LATEST_KEYWORDS:
        if keyword in combined_text:
            return True
    
    return False

def get_latest_pdf_link(pdf_links):
    """Filter and return only the latest PDF link"""
    if not pdf_links:
        return None
    
    # Score each link based on recency
    scored_links = []
    
    for link in pdf_links:
        try:
            href = link.get_attribute('href') or ""
            text = link.text or ""
            combined = (href + " " + text).lower()
            
            # Extract year
            year = extract_year_from_text(combined)
            
            # Calculate score
            score = year
            
            # Bonus points for specific keywords
            if 'annual report' in combined:
                score += 10
            if 'latest' in combined or 'current' in combined:
                score += 5
            if 'q4' in combined or 'fourth quarter' in combined:
                score += 3
            
            scored_links.append((score, year, link, text))
        except:
            continue
    
    if not scored_links:
        return pdf_links[0] if pdf_links else None
    
    # Sort by score (highest first)
    scored_links.sort(reverse=True, key=lambda x: (x[0], x[1]))
    
    # Return the top link
    top_link = scored_links[0]
    print(f"  📊 Selected: '{top_link[3][:50]}' (Year: {top_link[1]}, Score: {top_link[0]})")
    
    return top_link[2]

# Read URLs from text file
def read_urls(file_path):
    """Read URLs from a text file, one URL per line"""
    try:
        with open(file_path, 'r') as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        return urls
    except FileNotFoundError:
        print(f"Error: {file_path} not found!")
        return []

# Configure Chrome options
options = webdriver.ChromeOptions()
options.add_experimental_option('prefs', {
    "download.default_directory": "/Users/msf/Desktop/Projects/Financial Summarizer/Files",
    "download.prompt_for_download": False,
    "plugins.always_open_pdf_externally": True
})

# Add stealth options to avoid bot detection
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

# Remove webdriver property
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

# Read URLs from file
urls_file = "urls.txt"
urls_list = read_urls(urls_file)

if not urls_list:
    print("No URLs found in urls.txt file!")
    driver.quit()
    exit()

print(f"Found {len(urls_list)} URLs to process")

total_pdfs_downloaded = 0

# Loop through each URL
for url_index, url in enumerate(urls_list, 1):
    print(f"\n{'='*60}")
    print(f"Processing URL {url_index}/{len(urls_list)}: {url}")
    print(f"{'='*60}")
    
    try:
        driver.get(url)
        driver.maximize_window()
        time.sleep(5)  # Increased wait time for bot detection/CAPTCHA
        
        # Check if it's Nestle website (has CAPTCHA)
        if 'nestle' in url.lower():
            print("  ⚠️  CAPTCHA detected! Please solve it manually in the browser window.")
            print("  Waiting 30 seconds for you to solve CAPTCHA...")
            time.sleep(30)  # Give user time to solve CAPTCHA
        
        # Scroll down to load dynamic content
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(2)
        
        # Try to handle cookie notification
        try:
            cookie_div = driver.find_element(By.XPATH, '//div[@id="js-cookie-notification"]')
            if cookie_div:
                driver.find_element(By.XPATH, '//div[@id="js-cookie-notification"]/button').click()
                time.sleep(1)
        except:
            pass
        
        # Find all links
        all_a_tags = driver.find_elements(By.XPATH, "//a")
        pdf_links_found = []
        
        # First pass: collect all potential PDF links
        for a in all_a_tags:
            try:
                href = a.get_attribute('href')
                if href and '.pdf' in href.lower():
                    pdf_links_found.append(a)
            except:
                continue
        
        print(f"  Found {len(pdf_links_found)} potential PDF links")
        
        # Get only the LATEST PDF link
        latest_pdf_link = get_latest_pdf_link(pdf_links_found)
        
        if latest_pdf_link:
            try:
                driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", latest_pdf_link)
                time.sleep(1)
                driver.execute_script("arguments[0].click();", latest_pdf_link)
                time.sleep(3)  # Wait for download
                link_text = latest_pdf_link.text[:50] if latest_pdf_link.text else 'PDF'
                print(f"  ✅ Downloaded latest report: {link_text}")
                total_pdfs_downloaded += 1
            except Exception as e:
                print(f"  ❌ Error downloading: {e}")
        else:
            print(f"  ⚠️  No suitable PDF found")
        
        print(f"Total PDFs downloaded from this URL: {1 if latest_pdf_link else 0}")
        
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        continue

print(f"\n{'='*60}")
print(f"SUMMARY: Total PDFs downloaded from all URLs: {total_pdfs_downloaded}")
print(f"{'='*60}")

time.sleep(5)  # Wait for downloads to complete
driver.quit()