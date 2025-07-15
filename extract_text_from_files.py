import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from PIL import Image
from pathlib import Path
import os

# All Supported image types
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]

def extract_text_from_pdf(pdf_path):
    #text is empty string which will holds all text which is extracted
    text = ""
    reader = PdfReader(pdf_path)

    # extracted text is embedded
    #it reads all pages and if texts are there it will embed them and add  ---Page header so that each page is separated
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += f"\n--- Page {page_num + 1} (text layer) ---\n"
            text += page_text + "\n"

    # OCR for each page image
    #it will convert each pdf to images and apply OCR on each pages 
    try:
        images = convert_from_path(pdf_path)
        for i, img in enumerate(images):
            ocr_text = pytesseract.image_to_string(img)
            if ocr_text.strip():
                text += f"\n--- Page {i + 1} (OCR) ---\n"
                text += ocr_text + "\n"
    except Exception as e:
        print(f"OCR failed for {pdf_path.name}: {e}")

    #returns all text extracted from pdf and OCR 
    return text.strip()

#open images , run OCR on every images and if text are visible it will extract them
def extract_text_from_image(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"Failed to OCR image {image_path.name}: {e}")
        return ""

def convert_all_files(input_folder="data", output_folder="text_data"):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    files = list(input_path.glob("*"))
    if not files:
        print("No files found in 'data/' folder.")
        return

    for file in files:
        try:
            if file.suffix.lower() == ".pdf":
                print(f"Extracting from PDF: {file.name}")
                text = extract_text_from_pdf(file)
            elif file.suffix.lower() in IMAGE_EXTENSIONS:
                print(f"Extracting from Image: {file.name}")
                text = extract_text_from_image(file)
            else:
                print(f"Skipping unsupported file: {file.name}")
                continue

            if text:
                output_file = output_path / (file.stem + ".txt")
                output_file.write_text(text, encoding="utf-8")
                print(f"Saved: {output_file.name}")
            else:
                print(f"No text found in: {file.name}")
        except Exception as e:
            print(f"Error processing {file.name}: {e}")

if __name__ == "__main__":
    convert_all_files()
