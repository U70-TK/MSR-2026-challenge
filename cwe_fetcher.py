import requests
import time
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE_URL = "https://cwe.mitre.org/data/definitions"


def fetch_html(url: str) -> str:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.text


def extract_child_cwes(top_html: str):
    soup = BeautifulSoup(top_html, "html.parser")
    child_ids = set()

    for a in soup.select("a[href]"):
        href = a["href"]
        if href.startswith("/data/definitions/") and href.endswith(".html"):
            try:
                cid = int(href.split("/")[-1].replace(".html", ""))
                child_ids.add(cid)
            except:
                pass

    return sorted(child_ids)


def parse_cwe_page(html: str, cwe_id: int):
    soup = BeautifulSoup(html, "html.parser")

    # Extract CWE Title (usually <h1>)
    h1 = soup.find("h1")
    cwe_title = h1.get_text(strip=True) if h1 else None

    # Extract description using nested structure:
    # div id="Description" > div id="detail" > div id="indent"
    desc_container = soup.find("div", id="Description")
    cwe_description = None

    if desc_container:
        detail = desc_container.find("div", id="detail")
        if detail:
            indent = detail.find("div", id="indent")
            if indent:
                cwe_description = indent.get_text(separator="\n", strip=True)

    return {
        "cwe_id": cwe_id,
        "cwe_title": cwe_title,
        "cwe_description": cwe_description
    }



def scrape_children_to_parquet(top_cwe_id: int, output_path: str):
    # Load the parent page
    top_url = f"{BASE_URL}/{top_cwe_id}.html"
    top_html = fetch_html(top_url)

    # Extract child CWEs
    child_ids = extract_child_cwes(top_html)
    print(f"Found {len(child_ids)} child nodes for CWE-{top_cwe_id}")

    results = []

    for cid in tqdm(child_ids, desc=f"Scraping CWE-{top_cwe_id} children"):
        try:
            html = fetch_html(f"{BASE_URL}/{cid}.html")
            record = parse_cwe_page(html, cid)
            results.append(record)
        except Exception as e:
            print(f"Error fetching CWE-{cid}: {e}")
        time.sleep(0.3)  # polite delay

    # Convert to DF
    df = pd.DataFrame(results)

    # Save as Parquet
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path)

    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    scrape_children_to_parquet(1000, "cwe_children_1000.parquet")
