import os
import io
import fitz  # PyMuPDF
import base64
from PIL import Image
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

load_dotenv()

PROMPT = (
    "Anda adalah asisten AI ahli ekstraksi data dokumen. "
    "Ubah gambar halaman dokumen ini menjadi teks Markdown yang sangat rapi.\n\n"
    "ATURAN KETAT:\n"
    "1. Teks Biasa: Ekstrak semua teks narasi dengan akurat. Hapus spasi ganda yang aneh akibat teks rata kanan-kiri (justified). "
    "Jadikan kalimat mengalir natural dengan spasi tunggal.\n"
    "2. Tabel: JIKA TERDAPAT TABEL, Anda WAJIB mengubahnya menjadi format Markdown Table (menggunakan | Kolom | Kolom |). "
    "Jika ada teks tabel yang terpotong/multibaris di gambar aslinya, gabungkan menjadi satu baris panjang yang rapi di format Markdown.\n"
    "3. Gambar/Screenshot UI: Jika ada screenshot antarmuka web, ekstrak teks yang ada di dalam gambar tersebut dan "
    "tambahkan deskripsi singkat mengenai fitur/tombol yang terlihat (misal: '[Screenshot UI menunjukkan tombol Submit dan menu Dropdown]').\n"
    "4. Jangan berikan kalimat pembuka/penutup (seperti 'Berikut hasilnya'), langsung berikan output Markdown-nya saja."
)

def pdf_page_to_base64_image(page) -> str:
    pix = page.get_pixmap(dpi=150)
    img_data = pix.tobytes("png")
    return base64.b64encode(img_data).decode("utf-8")

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
def invoke_llm_with_retry(llm, message):
    return llm.invoke([message])

def parse_pdf(file_path: str):
    """
    Parses a PDF file page by page, using Gemini Multimodal to extract text and image descriptions.
    Returns a list of dictionaries with 'page' (1-indexed) and 'text' content.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)
    
    doc = fitz.open(file_path)
    parsed_pages = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        base64_image = pdf_page_to_base64_image(page)
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ]
        )
        
        response = invoke_llm_with_retry(llm, message)
        
        # Handle cases where response.content is a list of blocks
        content = response.content
        if isinstance(content, list):
            text_blocks = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
            extracted_text = "\n".join(text_blocks)
        else:
            extracted_text = str(content)
            
        parsed_pages.append({
            "page": page_num + 1,
            "text": extracted_text
        })
        
    return parsed_pages
