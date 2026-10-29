def chunk_pages(pages: list[dict], chunk_size: int = 1200, overlap: int = 180) -> list[dict]:
    if chunk_size < 100 or not 0 <= overlap < chunk_size:
        raise ValueError("chunk_size must be at least 100 and overlap smaller than it.")
    chunks: list[dict] = []
    for page in pages:
        text, start = page["text"], 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                boundary = max(text.rfind(". ", start, end), text.rfind(" ", start, end))
                if boundary > start + chunk_size // 2:
                    end = boundary + 1
            value = text[start:end].strip()
            if value:
                chunks.append({"text": value, "page": page.get("page"), "section": page.get("section")})
            if end == len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks
