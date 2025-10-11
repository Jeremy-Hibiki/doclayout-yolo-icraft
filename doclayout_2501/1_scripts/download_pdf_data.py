import httpx
import pymupdf

ATTENTION_IS_ALL_YOU_NEED = "https://arxiv.org/pdf/1706.03762"
QWEN3_OMNI = "https://arxiv.org/pdf/2509.17765"
KIMI_K2 = "https://arxiv.org/pdf/2507.20534"

if __name__ == "__main__":
    response = httpx.get(
        KIMI_K2,
        verify=False,
        proxy="http://127.0.0.1:7890",
    )
    doc = pymupdf.open(stream=response.content)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        output_image_path = f"./imgs/kimi_k2_{page_num + 1}.png"
        pix.save(output_image_path)
        print(f"Saved {output_image_path}")
    doc.close()
