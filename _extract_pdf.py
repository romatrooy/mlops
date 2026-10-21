import sys
from pypdf import PdfReader

path = sys.argv[1]
reader = PdfReader(path)
print("PAGES:", len(reader.pages))
for i, page in enumerate(reader.pages, 1):
    print(f"--- page {i} ---")
    print(page.extract_text())