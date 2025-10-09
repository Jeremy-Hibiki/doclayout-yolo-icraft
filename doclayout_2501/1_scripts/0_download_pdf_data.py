import httpx
import pymupdf

if __name__ == "__main__":
    pdf_path = "https://arxiv.org/pdf/1706.03762"
    response = httpx.get(pdf_path, verify=False, proxy="http://127.0.0.1:7890")
    doc = pymupdf.open(stream=response.content)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        output_image_path = f"./imgs/page_{page_num + 1}.png"
        pix.save(output_image_path)
        print(f"Saved {output_image_path}")
    doc.close()
