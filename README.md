# afro.scraper.py

# AfroGlamour Cosmetics Scraper

This project is part of a data analyst interview task for **Veefyed Ltd**.  
It demonstrates my ability to collect, clean, and structure product data from an online store (AfroGlamour Cosmetics), while extracting **all product images**.

---

## 🚀 Deliverables

- **afro_scraper.py** → Python script for scraping
- **AfroGlamour_Scraper.ipynb** → Jupyter notebook with step-by-step logic
- **AfroGlamour_Scraper_Logic.pdf** → Short PDF explaining approach
- **AfroGlamour_Sample_Output.xlsx** → Example dataset (mocked data)
- **data/** → CSV and Excel files generated when running the scraper
- **images/** → Downloaded product images

---

## 🛠 Tools Used
- Python 3
- [Requests](https://docs.python-requests.org/) (HTTP requests)
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) (HTML parsing)
- [Pandas](https://pandas.pydata.org/) (data structuring + CSV/Excel export)
- Regex (extracting barcodes, sizes, keywords)
- Optional: Selenium (if dynamic content is needed)

---

## ⚡ How It Works
1. Collect product URLs from `/shop/` using WooCommerce selector `a.woocommerce-LoopProduct-link`.
2. For each product:
   - Extract product name, description, price
   - Capture **all images** and download them locally
   - Extract barcodes, sizes, ingredients (if available)
   - Infer skin concern keywords (Acne, Moisturizing, Brightening, Anti-Aging)
3. Save structured data into CSV + Excel
4. Store all product images in `/images/`

---

## 📦 Outputs
- `afroglamour_products.csv`
- `afroglamour_products.xlsx`
- Downloaded images (`images/`)
- Ready-to-share logic note (`AfroGlamour_Scraper_Logic.pdf`)

---

## 📧 Author
Developed by Abdelrahman Agha
