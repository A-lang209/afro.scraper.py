# __define-ocg__ : AfroGlamour WooCommerce Scraper
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import re
from urllib.parse import urljoin

BASE_URL = "https://afroglamourcosmetics.com"
START_URL = BASE_URL + "/shop/"
HEADERS = {"User-Agent": "Mozilla/5.0"}
IMAGES_DIR = "afro_images"
OUTPUT_CSV = "afroglamour_products.csv"

os.makedirs(IMAGES_DIR, exist_ok=True)

def get_soup(url):
    r = requests.get(url, headers=HEADERS, timeout=15)
    return BeautifulSoup(r.text, "html.parser")

def get_product_links(listing_url):
    soup = get_soup(listing_url)
    links = []
    for a in soup.select("a.woocommerce-LoopProduct-link"):
        href = a.get("href")
        if href:
            full = href if href.startswith("http") else urljoin(BASE_URL, href)
            links.append(full)
    return list(set(links))

def scrape_product_page(purl):
    soup = get_soup(purl)
    product_id = re.sub(r"\W+", "_", purl)

    # Product Name
    name_tag = soup.select_one("h1.product_title")
    name = name_tag.get_text(strip=True) if name_tag else None

    # Price
    price_tag = soup.select_one("p.price, span.woocommerce-Price-amount")
    price = price_tag.get_text(strip=True) if price_tag else None

    # Description
    desc_tag = soup.select_one("div.woocommerce-product-details__short-description")
    desc = desc_tag.get_text(" ", strip=True) if desc_tag else None

    # Images
    varOcg = []
    for img in soup.select("div.woocommerce-product-gallery__image img"):
        src = img.get("src") or img.get("data-src")
        if src:
            full = src if src.startswith("http") else urljoin(BASE_URL, src)
            varOcg.append(full)
    product_images = ";".join(varOcg)

    # Download images
    for i, img_url in enumerate(varOcg):
        try:
            r = requests.get(img_url, stream=True, timeout=10)
            if r.status_code == 200:
                fname = f"{IMAGES_DIR}/{product_id}_img{i+1}.jpg"
                with open(fname, "wb") as f:
                    for chunk in r.iter_content(1024):
                        f.write(chunk)
        except Exception as e:
            print("Image download failed:", img_url, e)

    # Barcode
    barcode = None
    if desc:
        m = re.search(r"\b\d{12,13}\b", desc)
        if m:
            barcode = m.group(0)

    # Size / Volume
    size = None
    if desc:
        m = re.search(r"\b\d+\s*(ml|oz)\b", desc.lower())
        if m:
            size = m.group(0)

    # Ingredients
    ingredients = None
    ing_tag = soup.find(text=re.compile("Ingredients", re.IGNORECASE))
    if ing_tag:
        parent = ing_tag.parent
        if parent:
            ingredients = parent.get_text(" ", strip=True)

    # Skin Concern
    skin_concern = None
    if desc:
        dl = desc.lower()
        if "acne" in dl:
            skin_concern = "Acne"
        elif "brighten" in dl or "whitening" in dl:
            skin_concern = "Brightening"
        elif "anti-aging" in dl or "wrinkle" in dl:
            skin_concern = "Anti-Aging"
        elif "hydrate" in dl or "moistur" in dl:
            skin_concern = "Moisturizing"

    return {
        "ProductID": product_id,
        "ProductLineName": None,
        "BrandName": "Afro Glamour",
        "ProductName": name,
        "ProductDescription": desc,
        "ProductImages": product_images,
        "Barcode": barcode,
        "Price": price,
        "SizeVolume": size,
        "Ingredients": ingredients,
        "SkinConcern": skin_concern,
        "SourceURL": purl
    }

def scrape_all_products():
    links = get_product_links(START_URL)
    data = []
    for plink in links:
        try:
            rec = scrape_product_page(plink)
            data.append(rec)
        except Exception as e:
            print("Failed on", plink, e)
    return data

if __name__ == "__main__":
    records = scrape_all_products()
    varFiltersCg = pd.DataFrame(records)
    varFiltersCg.to_csv(OUTPUT_CSV, index=False)
    varFiltersCg.to_excel("afroglamour_products.xlsx", index=False)
    print("Scraped", len(records), "products")
    print(varFiltersCg.head())
