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



#######
#advanced code
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import json
import time
from urllib.parse import urljoin, urlparse
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
import pytesseract
from PIL import Image
import io
import cv2

class AfroGlamourScraper:
    def __init__(self, headless=True):
        self.base_url = "https://afroglamourcosmetics.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        # Setup Selenium
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 20)
    
    def get_all_product_urls(self):
        """Extract all product URLs with comprehensive discovery"""
        print("🔍 Searching for product URLs...")
        
        product_urls = set()
        
        # Strategy 1: Direct navigation to likely pages
        discovery_urls = [
            self.base_url,
            f"{self.base_url}/collections/all",
            f"{self.base_url}/collections",
            f"{self.base_url}/shop",
            f"{self.base_url}/products",
            f"{self.base_url}/collections/skincare",
            f"{self.base_url}/collections/makeup",
            f"{self.base_url}/collections/new-arrivals",
            f"{self.base_url}/collections/best-sellers"
        ]
        
        for discovery_url in discovery_urls:
            try:
                print(f"Scanning: {discovery_url}")
                self.driver.get(discovery_url)
                time.sleep(3)
                
                # Scroll to load dynamic content
                self._scroll_page()
                
                # Multiple selectors for product links
                product_selectors = [
                    "a[href*='/products/']",
                    "[class*='product'] a",
                    ".product-item a",
                    ".product-card a",
                    ".grid-item a",
                    ".collection-item a",
                    "a[class*='product']"
                ]
                
                for selector in product_selectors:
                    try:
                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        for elem in elements:
                            href = elem.get_attribute('href')
                            if href and '/products/' in href:
                                # Normalize URL
                                if not href.startswith('http'):
                                    href = urljoin(self.base_url, href)
                                product_urls.add(href)
                                print(f"✅ Found product: {href}")
                    except Exception as e:
                        continue
                        
            except Exception as e:
                print(f"❌ Error scanning {discovery_url}: {e}")
        
        # Strategy 2: Find via navigation menus
        try:
            self.driver.get(self.base_url)
            time.sleep(3)
            
            # Look for navigation links
            nav_links = self.driver.find_elements(By.CSS_SELECTOR, "nav a, .menu a, .navigation a")
            for link in nav_links:
                try:
                    href = link.get_attribute('href')
                    if href and any(keyword in href.lower() for keyword in ['collection', 'shop', 'product']):
                        if '/products/' in href:
                            product_urls.add(href)
                        else:
                            # It might be a collection page
                            collection_urls = self._extract_from_collection_page(href)
                            product_urls.update(collection_urls)
                except:
                    continue
        except Exception as e:
            print(f"❌ Error scanning navigation: {e}")
        
        valid_urls = [url for url in product_urls if url.startswith('http')]
        print(f"🎯 Total product URLs found: {len(valid_urls)}")
        return valid_urls
    
    def _extract_from_collection_page(self, collection_url):
        """Extract products from a collection page"""
        product_urls = set()
        
        try:
            self.driver.get(collection_url)
            time.sleep(3)
            self._scroll_page()
            
            # Extract product links
            product_links = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='/products/']")
            for link in product_links:
                href = link.get_attribute('href')
                if href and '/products/' in href:
                    if not href.startswith('http'):
                        href = urljoin(self.base_url, href)
                    product_urls.add(href)
                    
        except Exception as e:
            print(f"❌ Error processing collection {collection_url}: {e}")
        
        return product_urls
    
    def _scroll_page(self):
        """Scroll page to load all content"""
        try:
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            scroll_attempts = 0
            
            while scroll_attempts < 3:
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                
                if new_height == last_height:
                    break
                last_height = new_height
                scroll_attempts += 1
                
                # Check for lazy-loaded content
                try:
                    self.driver.execute_script("window.scrollTo(0, 0);")
                    time.sleep(1)
                    self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                except:
                    pass
                    
        except Exception as e:
            print(f"⚠️ Scroll error: {e}")
    
    def scrape_product_page(self, url):
        """Scrape individual product page with all required fields"""
        print(f"📦 Scraping: {url}")
        
        try:
            self.driver.get(url)
            time.sleep(4)
            
            # Wait for page to load completely
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            self._scroll_page()  # Scroll to load all content
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Extract all required fields
            product_name = self._extract_product_name(soup)
            description = self._extract_description(soup)
            images = self._extract_all_images(soup, url)
            
            product_data = {
                # 1) Product ID (consistent identifier)
                'product_id': self._generate_product_id(url),
                
                # 2) Product Line Name
                'product_line_name': self._extract_product_line(soup, url),
                
                # 3) Brand Name
                'brand_name': self._extract_brand_name(soup),
                
                # 4) Product Name
                'product_name': product_name,
                
                # 5) Product Description
                'product_description': description,
                
                # 6) Product Images (ALL URLs)
                'product_images': images,
                'image_count': len(images),
                
                # 7) Barcode (EAN/UPC) - from images or text
                'barcode': self._extract_barcode(soup, images, description),
                
                # 8) Price
                'price': self._extract_price(soup),
                
                # 9) Size/Volume
                'size_volume': self._extract_size_volume(soup, description),
                
                # 10) Ingredients
                'ingredients': self._extract_ingredients(soup, description),
                
                # 11) Skin Concern (inferred)
                'skin_concern': self._infer_skin_concern(product_name, description),
                
                # Additional fields
                'source_url': url,
                'availability': self._extract_availability(soup),
                'scraped_at': datetime.now().isoformat(),
                'scraping_method': 'selenium'
            }
            
            print(f"✅ Successfully scraped: {product_name} | Images: {len(images)} | Price: {product_data['price']}")
            return product_data
            
        except Exception as e:
            print(f"❌ Error scraping {url}: {e}")
            return None
    
    def _generate_product_id(self, url):
        """1) Create consistent Product ID from URL"""
        try:
            parsed = urlparse(url)
            path_parts = [p for p in parsed.path.split('/') if p]
            for part in path_parts:
                if part and part != 'products':
                    # Clean the ID
                    clean_id = re.sub(r'[^a-zA-Z0-9-_]', '', part)
                    return clean_id[:50]  # Limit length
        except:
            pass
        return f"prod_{hash(url) % 10000:04d}"
    
    def _extract_product_line(self, soup, url):
        """2) Extract Product Line Name"""
        # Try breadcrumbs
        breadcrumb_selectors = [
            ".breadcrumb a",
            ".breadcrumbs a", 
            ".breadcrumb-list a",
            ".nav-breadcrumb a"
        ]
        
        for selector in breadcrumb_selectors:
            try:
                elements = soup.select(selector)
                if len(elements) >= 2:
                    line_name = elements[-2].get_text(strip=True)
                    if line_name and line_name.lower() not in ['home', 'shop', 'products']:
                        return line_name
            except:
                continue
        
        # Extract from URL
        if '/collections/' in url:
            match = re.search(r'/collections/([^/?]+)', url)
            if match:
                collection_name = match.group(1).replace('-', ' ').title()
                return collection_name
        
        # Try category tags
        category_selectors = [".product-category", ".category", ".collection-title"]
        for selector in category_selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)
        
        return "General"
    
    def _extract_brand_name(self, soup):
        """3) Extract Brand Name"""
        brand_selectors = [
            ".product-brand",
            ".brand",
            "[itemprop='brand']",
            ".vendor",
            ".product-vendor"
        ]
        
        for selector in brand_selectors:
            element = soup.select_one(selector)
            if element:
                brand = element.get_text(strip=True)
                if brand:
                    return brand
        
        # Default brand for this site
        return "Afro Glamour Cosmetics"
    
    def _extract_product_name(self, soup):
        """4) Extract Product Name"""
        name_selectors = [
            "h1.product-title",
            "h1.product__title", 
            ".product-name",
            "h1",
            "[itemprop='name']",
            ".title",
            ".product-details h1"
        ]
        
        for selector in name_selectors:
            element = soup.select_one(selector)
            if element:
                name = element.get_text(strip=True)
                if name and name.lower() not in ['home', 'shop', 'product']:
                    return name
        
        # Fallback to title tag
        title_tag = soup.find('title')
        if title_tag:
            title_text = title_tag.get_text(strip=True)
            # Clean title (remove site name, etc.)
            clean_title = re.sub(r'[-|]?\\s*Afro Glamour Cosmetics.*', '', title_text)
            return clean_title.strip()
        
        return "Unknown Product"
    
    def _extract_description(self, soup):
        """5) Extract Product Description"""
        desc_selectors = [
            ".product-description",
            ".description",
            "[itemprop='description']",
            ".product-details",
            ".product-info"
        ]
        
        for selector in desc_selectors:
            elements = soup.select(selector)
            if elements:
                description = ' '.join([elem.get_text(strip=True) for elem in elements])
                if description:
                    return description
        
        # Try meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            return meta_desc['content']
        
        return ""
    
    def _extract_all_images(self, soup, base_url):
        """6) Extract ALL Product Images (not just one)"""
        images = set()
        
        # Use Selenium to find dynamically loaded images
        try:
            img_elements = self.driver.find_elements(By.TAG_NAME, "img")
            for img in img_elements:
                for attr in ['src', 'data-src', 'data-original', 'data-lazy-src']:
                    src = img.get_attribute(attr)
                    if src and self._is_product_image(src):
                        full_url = self._normalize_image_url(src, base_url)
                        if full_url:
                            images.add(full_url)
        except Exception as e:
            print(f"⚠️ Selenium image extraction error: {e}")
        
        # Use BeautifulSoup as fallback
        for img in soup.find_all('img'):
            for attr in ['src', 'data-src', 'data-original', 'data-lazy-src']:
                src = img.get(attr)
                if src and self._is_product_image(src):
                    full_url = self._normalize_image_url(src, base_url)
                    if full_url:
                        images.add(full_url)
        
        return list(images)
    
    def _is_product_image(self, src):
        """Check if image URL looks like a product image"""
        if not src or src.startswith(('data:', 'javascript:')):
            return False
        
        # Common product image patterns
        product_patterns = [
            '/products/',
            'product',
            'shopify',
            'collection',
            'cdn.shopify'
        ]
        
        return any(pattern in src.lower() for pattern in product_patterns)
    
    def _normalize_image_url(self, src, base_url):
        """Normalize image URL to full format"""
        if src.startswith('//'):
            return 'https:' + src
        elif src.startswith('/'):
            return urljoin(base_url, src)
        elif src.startswith('http'):
            return src
        else:
            return urljoin(base_url, src)
    
    def _extract_barcode(self, soup, images, description):
        """7) Extract Barcode (EAN/UPC) from images or text"""
        # First try to find in text content
        barcode_pattern = r'\b(\d{12,13})\b'  # EAN-13 or UPC
        
        # Check description
        if description:
            matches = re.findall(barcode_pattern, description)
            if matches:
                return matches[0]
        
        # Check entire page text
        page_text = soup.get_text()
        matches = re.findall(barcode_pattern, page_text)
        if matches:
            return matches[0]
        
        # Optional: OCR from images (commented for performance)
        # if images:
        #     return self._extract_barcode_from_images(images)
        
        return None
    
    def _extract_barcode_from_images(self, images):
        """Extract barcode using OCR from product images"""
        try:
            for img_url in images[:2]:  # Limit to first 2 images
                try:
                    response = requests.get(img_url, headers=self.headers, timeout=10)
                    if response.status_code == 200:
                        image = Image.open(io.BytesIO(response.content))
                        
                        # Preprocess image for OCR
                        img_array = np.array(image)
                        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                        
                        # Use OCR to extract text
                        custom_config = r'--oem 3 --psm 6'
                        text = pytesseract.image_to_string(gray, config=custom_config)
                        
                        # Look for barcode patterns
                        barcode_pattern = r'\b(\d{12,13})\b'
                        matches = re.findall(barcode_pattern, text)
                        if matches:
                            return matches[0]
                            
                except Exception as e:
                    continue
        except Exception as e:
            print(f"⚠️ Barcode OCR error: {e}")
        
        return None
    
    def _extract_price(self, soup):
        """8) Extract Price"""
        price_selectors = [
            ".price",
            ".product-price",
            "[itemprop='price']",
            ".money",
            ".product__price"
        ]
        
        for selector in price_selectors:
            element = soup.select_one(selector)
            if element:
                price_text = element.get_text(strip=True)
                # Extract numeric value
                matches = re.findall(r'[\d.,]+', price_text)
                if matches:
                    price_value = matches[0].replace(',', '')
                    try:
                        return float(price_value)
                    except:
                        continue
        
        return None
    
    def _extract_size_volume(self, soup, description):
        """9) Extract Size/Volume"""
        # Combine description with any size-specific elements
        size_selectors = [".size", ".volume", ".capacity", ".weight"]
        size_text = description
        
        for selector in size_selectors:
            element = soup.select_one(selector)
            if element:
                size_text += " " + element.get_text(strip=True)
        
        # Look for size/volume patterns
        size_patterns = [
            r'(\d+\s*ml)\b',
            r'(\d+\s*oz)\b',
            r'(\d+\s*g)\b',
            r'(\d+\.\d+\s*(ml|oz|g))',
            r'(\d+\s*/\s*\d+\s*(ml|oz|g))',
            r'size[:\\s]*([^\\s.,]+)',
            r'volume[:\\s]*([^\\s.,]+)'
        ]
        
        for pattern in size_patterns:
            matches = re.findall(pattern, size_text, re.IGNORECASE)
            if matches:
                size = matches[0][0] if isinstance(matches[0], tuple) else matches[0]
                return size.strip()
        
        return None
    
    def _extract_ingredients(self, soup, description):
        """10) Extract Ingredients"""
        # Look for dedicated ingredients section
        ingredient_selectors = [
            ".ingredients",
            ".ingredient-list",
            "#ingredients",
            ".formula"
        ]
        
        for selector in ingredient_selectors:
            element = soup.select_one(selector)
            if element:
                ingredients = element.get_text(strip=True)
                if ingredients:
                    return ingredients[:1000]  # Limit length
        
        # Try to extract from description
        if description:
            ingredient_keywords = ['ingredients:', 'contains:', 'formula:', 'with:']
            for keyword in ingredient_keywords:
                if keyword.lower() in description.lower():
                    start_idx = description.lower().index(keyword.lower()) + len(keyword)
                    # Extract next reasonable chunk of text
                    end_idx = min(start_idx + 500, len(description))
                    ingredients = description[start_idx:end_idx].strip()
                    
                    # Clean up (stop at next section)
                    stop_words = ['how to use', 'benefits', 'direction', 'warning']
                    for stop_word in stop_words:
                        if stop_word in ingredients.lower():
                            stop_idx = ingredients.lower().index(stop_word)
                            ingredients = ingredients[:stop_idx]
                    
                    return ingredients.strip()
        
        return None
    
    def _infer_skin_concern(self, product_name, description):
        """Infer Skin Concern from text analysis"""
        if not product_name and not description:
            return "General"
        
        text_to_analyze = f"{product_name} {description}".lower()
        
        skin_concerns = {
            'hyperpigmentation': ['pigment', 'dark spot', 'brighten', 'even tone', 'glow', 'whiten'],
            'acne': ['acne', 'breakout', 'blemish', 'pimple', 'blackhead', 'clarifying'],
            'aging': ['aging', 'wrinkle', 'anti-age', 'firm', 'lift', 'collagen'],
            'dryness': ['dry', 'hydration', 'moisture', 'nourish', 'hydrate'],
            'sensitivity': ['sensitive', 'calm', 'soothe', 'gentle', 'redness'],
            'oiliness': ['oil', 'shine', 'matte', 'porcelain', 'sebum']
        }
        
        detected_concerns = []
        for concern, keywords in skin_concerns.items():
            if any(keyword in text_to_analyze for keyword in keywords):
                detected_concerns.append(concern)
        
        return ', '.join(detected_concerns) if detected_concerns else 'General Care'
    
    def _extract_availability(self, soup):
        """Extract product availability"""
        availability_selectors = [
            ".availability",
            ".stock",
            ".in-stock",
            ".out-of-stock",
            "[itemprop='availability']"
        ]
        
        for selector in availability_selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(strip=True).lower()
                if any(word in text for word in ['out', 'sold', 'unavailable']):
                    return 'Out of Stock'
                else:
                    return 'In Stock'
        
        return 'Unknown'
    
    def close(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()

class DataAnalyzer:
    """Class for data analysis and visualization"""
    
    def __init__(self, df):
        self.df = df
        plt.style.use('default')
    
    def create_comprehensive_dashboard(self):
        """Create a comprehensive analysis dashboard"""
        if self.df.empty:
            print("📊 No data available for visualization")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle('Afro Glamour Cosmetics - Complete Product Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Price Distribution
        self._plot_price_distribution(axes[0, 0])
        
        # 2. Product Categories
        self._plot_categories(axes[0, 1])
        
        # 3. Skin Concerns Analysis
        self._plot_skin_concerns(axes[1, 0])
        
        # 4. Image Analysis
        self._plot_image_analysis(axes[1, 1])
        
        # 5. Availability Status
        self._plot_availability(axes[2, 0])
        
        # 6. Price vs Images
        self._plot_price_vs_images(axes[2, 1])
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('product_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_price_distribution(self, ax):
        """Plot price distribution"""
        prices = self.df['price'].dropna()
        if len(prices) > 0:
            ax.hist(prices, bins=15, alpha=0.7, color='#FF6B9D', edgecolor='black')
            ax.set_title('💰 Price Distribution', fontweight='bold')
            ax.set_xlabel('Price ($)')
            ax.set_ylabel('Number of Products')
            if len(prices) > 1:
                ax.axvline(prices.mean(), color='red', linestyle='--', 
                          label=f'Average: ${prices.mean():.2f}')
                ax.legend()
        else:
            ax.text(0.5, 0.5, 'No price data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
    
    def _plot_categories(self, ax):
        """Plot product categories distribution"""
        categories = self.df['product_line_name'].value_counts().head(10)
        if len(categories) > 0:
            categories.plot(kind='bar', ax=ax, color='#9C27B0')
            ax.set_title('📦 Product Categories', fontweight='bold')
            ax.set_ylabel('Number of Products')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, 'No category data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
    
    def _plot_skin_concerns(self, ax):
        """Plot skin concern analysis"""
        concerns = self.df['skin_concern'].str.split(', ').explode().value_counts()
        if len(concerns) > 0:
            colors = plt.cm.Set3(np.linspace(0, 1, len(concerns)))
            wedges, texts, autotexts = ax.pie(concerns.values, labels=concerns.index, 
                                            autopct='%1.1f%%', startangle=90, colors=colors)
            ax.set_title('🌟 Skin Concern Focus', fontweight='bold')
            # Improve readability
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            ax.text(0.5, 0.5, 'No skin concern data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
    
    def _plot_image_analysis(self, ax):
        """Plot image count analysis"""
        if 'image_count' in self.df.columns:
            image_data = self.df['image_count'].value_counts().sort_index()
            if len(image_data) > 0:
                ax.bar(image_data.index, image_data.values, color='orange', alpha=0.7)
                ax.set_title('🖼️ Images per Product', fontweight='bold')
                ax.set_xlabel('Number of Images')
                ax.set_ylabel('Number of Products')
                # Add value labels on bars
                for i, v in enumerate(image_data.values):
                    ax.text(image_data.index[i], v + 0.1, str(v), ha='center', va='bottom')
            else:
                ax.text(0.5, 0.5, 'No image data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
        else:
            ax.text(0.5, 0.5, 'No image data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
    
    def _plot_availability(self, ax):
        """Plot product availability"""
        if 'availability' in self.df.columns:
            availability = self.df['availability'].value_counts()
            if len(availability) > 0:
                colors = ['green' if 'In Stock' in str(x) else 'red' if 'Out' in str(x) else 'gray' 
                         for x in availability.index]
                availability.plot(kind='bar', ax=ax, color=colors)
                ax.set_title('📦 Product Availability', fontweight='bold')
                ax.set_ylabel('Number of Products')
                # Add value labels
                for i, v in enumerate(availability.values):
                    ax.text(i, v + 0.1, str(v), ha='center', va='bottom')
            else:
                ax.text(0.5, 0.5, 'No availability data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
        else:
            ax.text(0.5, 0.5, 'No availability data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
    
    def _plot_price_vs_images(self, ax):
        """Plot price vs number of images"""
        if 'price' in self.df.columns and 'image_count' in self.df.columns:
            valid_data = self.df.dropna(subset=['price', 'image_count'])
            if len(valid_data) > 0:
                ax.scatter(valid_data['image_count'], valid_data['price'], 
                          alpha=0.6, color='blue', s=60)
                ax.set_title('💵 Price vs Number of Images', fontweight='bold')
                ax.set_xlabel('Number of Images')
                ax.set_ylabel('Price ($)')
                
                # Add trend line if enough data
                if len(valid_data) > 1:
                    z = np.polyfit(valid_data['image_count'], valid_data['price'], 1)
                    p = np.poly1d(z)
                    ax.plot(valid_data['image_count'], p(valid_data['image_count']), 
                           "r--", alpha=0.8)
            else:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
        else:
            ax.text(0.5, 0.5, 'No price/image data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
    
    def generate_detailed_report(self):
        """Generate a comprehensive text report"""
        print("=" * 70)
        print("📊 AFRO GLAMOUR COSMETICS - COMPREHENSIVE SCRAPING REPORT")
        print("=" * 70)
        
        print(f"📦 Total Products Scraped: {len(self.df)}")
        print(f"🖼️ Total Images Found: {self.df['image_count'].sum()}")
        print(f"💰 Products with Price: {self.df['price'].notna().sum()}")
        print(f"🏷️ Unique Categories: {self.df['product_line_name'].nunique()}")
        
        # Price analysis
        if self.df['price'].notna().any():
            prices = self.df['price'].dropna()
            print(f"\n💵 PRICE ANALYSIS:")
            print(f"   Average Price: ${prices.mean():.2f}")
            print(f"   Median Price: ${prices.median():.2f}")
            print(f"   Price Range: ${prices.min():.2f} - ${prices.max():.2f}")
            print(f"   Standard Deviation: ${prices.std():.2f}")
        
        # Availability summary
        if 'availability' in self.df.columns:
            availability = self.df['availability'].value_counts()
            print(f"\n📦 AVAILABILITY SUMMARY:")
            for status, count in availability.items():
                percentage = (count / len(self.df)) * 100
                print(f"   {status}: {count} products ({percentage:.1f}%)")
        
        # Skin concerns analysis
        concerns = self.df['skin_concern'].str.split(', ').explode().value_counts()
        print(f"\n🌟 SKIN CONCERN ANALYSIS:")
        for concern, count in concerns.head(8).items():
            percentage = (count / len(self.df)) * 100
            print(f"   {concern}: {count} products ({percentage:.1f}%)")
        
        # Image analysis
        if 'image_count' in self.df.columns:
            image_stats = self.df['image_count'].describe()
            print(f"\n🖼️ IMAGE ANALYSIS:")
            print(f"   Average images per product: {image_stats['mean']:.1f}")
            print(f"   Max images: {image_stats['max']}")
            print(f"   Min images: {image_stats['min']}")
        
        print("\n" + "=" * 70)

def main():
    """Main execution function"""
    print("🚀 Starting Afro Glamour Cosmetics Scraper...")
    print("This may take a few minutes depending on the number of products...")
    
    scraper = AfroGlamourScraper(headless=True)
    
    try:
        # Step 1: Discover product URLs
        product_urls = scraper.get_all_product_urls()
        
        if not product_urls:
            print("❌ No product URLs found. The site structure may have changed.")
            print("💡 Try running with headless=False to see what's happening.")
            return
        
        print(f"🎯 Found {len(product_urls)} product URLs to scrape")
        
        # Step 2: Scrape each product
        all_products = []
        successful_scrapes = 0
        
        for i, url in enumerate(product_urls):
            print(f"\n📊 Progress: {i+1}/{len(product_urls)} ({((i+1)/len(product_urls))*100:.1f}%)")
            
            product_data = scraper.scrape_product_page(url)
            if product_data:
                all_products.append(product_data)
                successful_scrapes += 1
            
            # Respectful delay between requests
            time.sleep(2)
            
            # Save progress every 10 products
            if (i + 1) % 10 == 0 and all_products:
                temp_df = pd.DataFrame(all_products)
                temp_df.to_csv('afro_glamour_progress.csv', index=False)
                print(f"💾 Progress saved: {len(all_products)} products so far")
        
        # Step 3: Save final results
        if all_products:
            df = pd.DataFrame(all_products)
            
            # Save main product data
            df.to_csv('afro_glamour_products_complete.csv', index=False)
            print(f"✅ Main data saved: afro_glamour_products_complete.csv")
            
            # Save detailed images data
            image_data = []
            for _, product in df.iterrows():
                for j, img_url in enumerate(product['product_images']):
                    image_data.append({
                        'product_id': product['product_id'],
                        'product_name': product['product_name'],
                        'image_url': img_url,
                        'image_index': j,
                        'product_url': product['source_url']
                    })
            
            image_df = pd.DataFrame(image_data)
            image_df.to_csv('afro_glamour_images_detailed.csv', index=False)
            print(f"✅ Images data saved: afro_glamour_images_detailed.csv")
            
            # Step 4: Generate analysis and visualizations
            print("\n📈 Generating analysis and visualizations...")
            analyzer = DataAnalyzer(df)
            analyzer.create_comprehensive_dashboard()
            analyzer.generate_detailed_report()
            
            print(f"\n🎉 SCRAPING COMPLETED SUCCESSFULLY!")
            print(f"📊 Products successfully scraped: {successful_scrapes}/{len(product_urls)}")
            print(f"🖼️ Total images extracted: {df['image_count'].sum()}")
            print(f"💾 Files created:")
            print(f"   - afro_glamour_products_complete.csv")
            print(f"   - afro_glamour_images_detailed.csv") 
            print(f"   - product_analysis_dashboard.png")
            
            return df
        else:
            print("❌ No products were successfully scraped.")
            return None
            
    except Exception as e:
        print(f"❌ Error during scraping: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        scraper.close()
        print("🔚 Scraper closed.")

if __name__ == "__main__":
    # Installation requirements:
    # pip install requests beautifulsoup4 pandas selenium matplotlib seaborn wordcloud pillow pytesseract opencv-python numpy
    
    df = main()
