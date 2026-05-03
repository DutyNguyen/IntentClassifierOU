"""
preprocessor.py
NLP Preprocessing Pipeline cho tieng Viet
"""

import re
import unicodedata
from pathlib import Path

try:
    from underthesea import word_tokenize
    UNDERTHESEA_AVAILABLE = True
except ImportError:
    UNDERTHESEA_AVAILABLE = False
    print("underthesea chua cai. Dung fallback tokenizer.")


# ------------------------------------------------------------------ #
# Stopwords tiếng Việt tùy chỉnh (domain tuyển sinh)
# ------------------------------------------------------------------ #
# Stopwords chat (chi bo cum tu khong mang y nghia nganh)
# Giu lai nhung tu co tinh phan biet intent: hoc, phi, diem, ho so, han...
STOPWORDS = {
    "va", "cua", "trong", "theo", "nhu", "hay", "hoac", "vi", "do", "boi",
    "da", "se", "dang", "van", "con", "them", "bi", "ra", "vao", "len",
    "xuong", "lai", "di", "den", "rat", "qua", "kha", "hoi", "cang",
    "nhieu", "it", "chung", "moi", "ai",
}

# ------------------------------------------------------------------ #
# Preprocessing Pipeline
# ------------------------------------------------------------------ #
class VietnamesePreprocessor:
    """
    Pipeline:
      1. Lowercase & normalize unicode
      2. Xóa ký tự đặc biệt
      3. Word segmentation (underthesea)
      4. Loại bỏ stopwords
    """

    def __init__(self, use_word_segment: bool = True, remove_stopwords: bool = True):
        self.use_word_segment = use_word_segment and UNDERTHESEA_AVAILABLE
        self.remove_stopwords = remove_stopwords

    def normalize(self, text: str) -> str:
        """Lowercase, chuan hoa unicode, bo ky tu thua."""
        text = text.lower().strip()
        # Giu lai chu, so, dau tieng Viet, khoang trang
        text = re.sub(r"[^\w\s\u00C0-\u024F\u1E00-\u1EFF]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def strip_accents(self, text: str) -> str:
        """Chuyen tieng Viet co dau -> khong dau de match voi query khong dau."""
        # NFD decompose -> bo combining chars -> NFC
        nfd = unicodedata.normalize("NFD", text)
        stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
        return unicodedata.normalize("NFC", stripped)

    def tokenize(self, text: str) -> list[str]:
        """Phân đoạn từ tiếng Việt."""
        if self.use_word_segment:
            tokens = word_tokenize(text, format="text").split()
        else:
            tokens = text.split()
        return tokens

    def filter_stopwords(self, tokens: list[str]) -> list[str]:
        """Lọc stopwords và token quá ngắn."""
        return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    def process(self, text: str) -> str:
        """Xu ly day du, tra ve chuoi da xu ly.
        
        - Normalize + lowercase
        - Strip accents tren ban sao de match ca co dau lan khong dau
        - Word segment neu co underthesea  
        - Filter stopwords nhe
        """
        text = self.normalize(text)
        # Tao them ban khong dau de tang coverage khi query khong dau
        no_accent = self.strip_accents(text)
        combined = text if text == no_accent else f"{text} {no_accent}"
        tokens = self.tokenize(combined)
        if self.remove_stopwords:
            tokens = self.filter_stopwords(tokens)
        return " ".join(tokens)

    def process_batch(self, texts: list[str]) -> list[str]:
        """Xử lý nhiều câu cùng lúc."""
        return [self.process(t) for t in texts]


# ------------------------------------------------------------------ #
# Quick test
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    preprocessor = VietnamesePreprocessor()

    test_sentences = [
        "Xin chào, tôi muốn hỏi về điểm chuẩn ngành CNTT năm 2026?",
        "Học từ xa có cần thi đầu vào không?",
        "Hạn nộp hồ sơ xét tuyển học bạ là khi nào?",
        "Trường xét tuyển theo những tổ hợp môn nào?",
        "Có IELTS 6.5 thì được cộng mấy điểm ưu tiên?",
    ]

    print("=== NLP Preprocessing Pipeline ===\n")
    for sent in test_sentences:
        processed = preprocessor.process(sent)
        print(f"Input:  {sent}")
        print(f"Output: {processed}")
        print()
