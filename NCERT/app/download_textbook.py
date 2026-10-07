import os
import ssl
import urllib.request
import concurrent.futures

CHAPTER_NAMES = {
    1: "Chemical Reactions and Equations",
    2: "Acids, Bases and Salts",
    3: "Metals and Non-metals",
    4: "Carbon and its Compounds",
    5: "Life Processes",
    6: "Control and Coordination",
    7: "How do Organisms Reproduce?",
    8: "Heredity",
    9: "Light – Reflection and Refraction",
    10: "The Human Eye and the Colourful World",
    11: "Electricity",
    12: "Magnetic Effects of Electric Current",
    13: "Our Environment"
}

PDF_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw_pdfs")

def download_chapter(ch_num):
    os.makedirs(PDF_DIR, exist_ok=True)
    pdf_path = os.path.join(PDF_DIR, f"chapter_{ch_num:02d}.pdf")
    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 100000:
        print(f"Chapter {ch_num} already downloaded ({os.path.getsize(pdf_path)} bytes).")
        return pdf_path

    url = f"https://ncert.nic.in/textbook/pdf/jesc1{ch_num:02d}.pdf"
    print(f"Downloading Chapter {ch_num}: {CHAPTER_NAMES[ch_num]} from {url}...")
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
        content = resp.read()
        with open(pdf_path, "wb") as f:
            f.write(content)
            
    print(f"Saved Chapter {ch_num} ({len(content)} bytes).")
    return pdf_path

def download_all():
    print(f"Starting parallel download for {len(CHAPTER_NAMES)} NCERT Class 10 Science chapters...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_chapter, ch): ch for ch in CHAPTER_NAMES}
        for future in concurrent.futures.as_completed(futures):
            ch = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error downloading chapter {ch}: {e}")

if __name__ == "__main__":
    download_all()
