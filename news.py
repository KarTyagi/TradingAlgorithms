import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

def is_relevant_article(title, link):
    paywalled_domains = ["wsj.com", "barrons.com", "ft.com", "seekingalpha.com", "marketwatch.com"]
    if any(domain in link for domain in paywalled_domains):
        return False
    exclude_keywords = ["opinion", "sponsored", "press release", "advertisement", "ad:", "recap", "summary"]
    title_lower = title.lower()
    link_lower = link.lower()
    if any(keyword in title_lower or keyword in link_lower for keyword in exclude_keywords):
        return False
    if not link or link == "N/A":
        return False
    return True

def fetch_article_text(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        main_div = soup.find("div", class_="caas-body") or soup.find("div", class_="article-body")
        if main_div:
            paragraphs = main_div.find_all("p")
            text = " ".join(p.get_text() for p in paragraphs)
            return text.strip()
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text() for p in paragraphs)
        return text.strip()
    except Exception as e:
        print(f"[DEBUG] Failed to fetch article text from {url}: {e}")
        return None

def scrape_yf(target_date):
    url = "https://finance.yahoo.com/news/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.find_all("li", class_="js-stream-content") or soup.find_all("li", class_=lambda x: x and "stream-item" in x)
        if not articles:
            print("No articles found with current selectors.")
            return []
        article_data = []
        cutoff_date = target_date - timedelta(days=1)
        for article in articles[:20]:
            title_elem = article.find("h3")
            if not title_elem:
                continue
            title = title_elem.text.strip()
            link_elem = article.find("a", href=True)
            link = link_elem["href"] if link_elem else "N/A"
            if not link.startswith("http"):
                link = f"https://finance.yahoo.com{link}"
            time_elem = article.find("time")
            article_date = None
            if time_elem:
                time_text = time_elem.get("datetime", "")
                try:
                    article_date = datetime.strptime(time_text[:10], "%Y-%m-%d").date()
                    if article_date < cutoff_date:
                        continue
                except ValueError:
                    pass
            if is_relevant_article(title, link):
                article_data.append({"title": title, "link": link})
            if len(article_data) >= 10:
                break
        if not article_data:
            print("No relevant articles matched the date criteria or were found.")
        return article_data
    except Exception as e:
        print(f"Error scraping articles: {e}")
        return []
