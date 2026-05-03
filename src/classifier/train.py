import json
import joblib
import random
import unicodedata
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

# Path
ROOT = Path(__file__).parent.parent.parent
DATA_FILE = ROOT / "data" / "intents" / "intents.json"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

console = Console()


# Load dữ liệu
def load_data(path: Path) -> tuple[list[str], list[str]]:
    """Load intents.json → (sentences, labels)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    texts, labels = [], []
    for intent in data["intents"]:
        for example in intent["examples"]:
            texts.append(example)
            labels.append(intent["tag"])

    console.print(f"[cyan]Tong mau: {len(texts)} | So intent: {len(set(labels))}[/cyan]")
    return texts, labels


# Preprocessing 
def preprocess(texts: list[str]) -> list[str]:
    """Tien xu ly tieng Viet."""
    try:
        import sys
        sys.path.append(str(ROOT / "src"))
        from nlp.preprocessor import VietnamesePreprocessor
        preprocessor = VietnamesePreprocessor()
        return preprocessor.process_batch(texts)
    except Exception as e:
        console.print(f"[yellow]Fallback preprocessor: {e}[/yellow]")
        return [t.lower().strip() for t in texts]


# Data Augmentation
def strip_diacritics(text: str) -> str:
    """Chuyen tieng Viet co dau thanh khong dau."""
    nfd = unicodedata.normalize("NFD", text)
    # Bo tat ca combining diacritical marks (U+0300..U+036F) va
    # combining half marks, dang co the xuat hien trong tieng Viet
    result = "".join(
        c for c in nfd
        if unicodedata.category(c) not in ("Mn",)  # Mn = Mark, Nonspacing
        and c != "\u0111"  # d-gach (d with stroke) -> d
        and c != "\u0110"
    )
    # xu ly 'd gach' rieng (khong phai combining mark)
    result = result.replace("đ", "d").replace("Đ", "D")
    # Normalize lai NFD -> NFC de dam bao sach
    return unicodedata.normalize("NFC", result)


# Cac cap ki tu hay nham khi go nhanh (typo simulation nhe)
_TYPO_MAP = [
    ("ph", "f"),    # pho -> fo
    ("ng", "n"),    # ngang -> nan (cuoi tu)
    ("nh", "n"),    # nhanh -> nan
    ("ch", "c"),    # chon -> con
    ("kh", "k"),    # khong -> kong
    ("gi", "g"),    # giao -> gao
    ("qu", "q"),    # quan -> qan
    ("tr", "t"),    # truong -> tuong
    ("th", "t"),    # the -> te
]


def simulate_typo(text: str, p: float = 0.25) -> str:
    if random.random() > p:
        return text  # khong augment
    text_lower = text.lower()
    # Tim cac cap co the thay the
    candidates = [(old, new) for old, new in _TYPO_MAP if old in text_lower]
    if not candidates:
        return text
    old, new = random.choice(candidates)
    # Chi thay 1 lan dau tien
    return text_lower.replace(old, new, 1)


def augment_data(
    texts: list[str],
    labels: list[str],
    no_diacritic: bool = True,
    typo: bool = True,
    typo_p: float = 0.25,
    skip_tags: tuple = ("out_of_scope",),  # khong augment OOS
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Sinh them du lieu augmented:
    1. no_diacritic: moi example -> them ban khong dau (x2 du lieu)
    2. typo: moi example -> co the them ban typo nhe (x1.25 du lieu)

    Tra ve (texts_aug, labels_aug) chua ca ban goc lan augmented.
    """
    random.seed(seed)
    aug_texts, aug_labels = list(texts), list(labels)  # giu nguyen ban goc

    for text, label in zip(texts, labels):
        if label in skip_tags:
            continue  # khong augment out_of_scope (co the gay nhieu)

        # --- Augment 1: No-diacritic ---
        if no_diacritic:
            stripped = strip_diacritics(text)
            if stripped != text:  # chi them neu khac ban goc
                aug_texts.append(stripped)
                aug_labels.append(label)

        # --- Augment 2: Typo simulation ---
        if typo:
            typo_ver = simulate_typo(text, p=typo_p)
            if typo_ver != text.lower():  # chi them neu thuc su thay doi
                aug_texts.append(typo_ver)
                aug_labels.append(label)

    return aug_texts, aug_labels




# ------------------------------------------------------------------ #
# Pipeline definition
# ------------------------------------------------------------------ #
def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            max_features=8000,
            sublinear_tf=True,
        )),
        ("clf", MultinomialNB(alpha=0.1, fit_prior=False)),
    ])


# ------------------------------------------------------------------ #
# Đánh giá bằng cross-validation
# ------------------------------------------------------------------ #
def evaluate_cv(pipeline: Pipeline, X: list[str], y: list[str], k: int = 5):
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    X_arr = np.array(X, dtype=object)
    y_arr = np.array(y)
    scores = cross_val_score(pipeline, X_arr, y_arr, cv=cv, scoring="f1_macro")

    table = Table(title=f"{k}-Fold Cross-Validation (F1-Macro)", show_header=True)
    table.add_column("Fold", style="cyan", justify="center")
    table.add_column("F1-Macro", style="green", justify="center")
    for i, s in enumerate(scores, 1):
        table.add_row(str(i), f"{s:.4f}")
    table.add_row("[bold]Mean[/bold]", f"[bold]{scores.mean():.4f} ± {scores.std():.4f}[/bold]")
    console.print(table)
    return scores




# ------------------------------------------------------------------ #
# Vẽ Confusion Matrix
# ------------------------------------------------------------------ #
def plot_confusion_matrix(y_true, y_pred, labels, save_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    fig, ax = plt.subplots(figsize=(20, 20))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    # Plot with 'Blues' colormap just like the reference
    disp.plot(
        include_values=True,
        cmap='Blues',
        ax=ax,
        xticks_rotation='vertical',
        values_format='d'
    )
    
    # Optional styling to ensure it matches the reference perfectly
    ax.set_title("Confusion Matrix — Inference Engine", fontsize=18, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    console.print(f"[green]Confusion matrix -> {save_path}[/green]")
    plt.close()


# ------------------------------------------------------------------ #
# Main training
# ------------------------------------------------------------------ #
def train():
    console.print("\n[bold]Bat dau huan luyen Intent Classifier[/bold]\n")

    # 1. Load data
    texts_raw, labels = load_data(DATA_FILE)

    # 2. Data Augmentation (truoc preprocess)
    texts_aug, labels_aug = augment_data(
        texts_raw, labels,
        no_diacritic=True,
        typo=True,
        typo_p=0.25,
    )
    n_orig = len(texts_raw)
    n_aug  = len(texts_aug)
    console.print(
        f"[cyan]Du lieu sau augmentation: {n_aug} mau "
        f"(goc: {n_orig} | them: {n_aug - n_orig})[/cyan]"
    )

    # 3. Preprocess
    console.print("\n[cyan]Tien xu ly van ban...[/cyan]")
    texts = preprocess(texts_aug)
    labels = labels_aug

    # 4. Build & cross-validate
    console.print("\n[cyan]Cross-validation (k=5)...[/cyan]")
    pipeline = build_pipeline()
    scores = evaluate_cv(pipeline, texts, labels, k=5)
    console.print(f"[green]CV F1-Macro: {scores.mean():.4f} ± {scores.std():.4f}[/green]")

    # 5. CV per-class report (chinh xac hon in-sample)
    from sklearn.model_selection import cross_val_predict
    console.print("\n[cyan]CV Classification Report (per class)...[/cyan]")
    X_arr = np.array(texts, dtype=object)
    y_arr = np.array(labels)
    y_cv_pred = cross_val_predict(
        build_pipeline(), X_arr, y_arr,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    )
    report = classification_report(labels, y_cv_pred, zero_division=0)
    console.print("\n[bold]Classification Report (5-fold CV -- khong phai in-sample!):[/bold]")
    console.print(report)

    # 5.1 Tong hop metrics de frontend hien thi so lieu thuc
    metrics = {
        "model": "MultinomialNB",
        "cv_f1_macro_mean": float(scores.mean()),
        "cv_f1_macro_std": float(scores.std()),
        "accuracy": float(accuracy_score(labels, y_cv_pred)),
        "precision_macro": float(precision_score(labels, y_cv_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, y_cv_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, y_cv_pred, average="macro", zero_division=0)),
        "n_samples": len(labels),
        "n_classes": len(set(labels)),
        "cv_folds": 5,
    }

    # 6. Train tren toan bo du lieu (goc + augmented)
    console.print("\n[cyan]Huan luyen tren toan bo du lieu...[/cyan]")
    pipeline.fit(texts, labels)

    # 7. Confusion matrix (tren CV predictions)
    unique_labels = sorted(set(labels))
    plot_confusion_matrix(
        labels, y_cv_pred, unique_labels,
        save_path=MODEL_DIR / "confusion_matrix.png"
    )

    # 8. Luu mo hinh
    model_path = MODEL_DIR / "intent_classifier.pkl"
    joblib.dump(pipeline, model_path)
    console.print(f"\n[bold green]Mo hinh da luu -> {model_path}[/bold green]")

    # 9. Luu nhan
    label_path = MODEL_DIR / "labels.json"
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump(unique_labels, f, ensure_ascii=False, indent=2)

    # 10. Luu metrics danh gia
    metrics_path = MODEL_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    console.print(f"[green]Metrics da luu -> {metrics_path}[/green]")

    return pipeline


# ------------------------------------------------------------------ #
# Test nhanh sau khi train
# ------------------------------------------------------------------ #
def quick_test(pipeline: Pipeline):
    console.print("\n[bold]Quick Test:[/bold]")
    test_cases = [
        "Điểm chuẩn ngành công nghệ thông tin bao nhiêu?",
        "Hạn nộp hồ sơ xét học bạ là khi nào?",
        "Học từ xa có cần thi đầu vào không?",
        "Có IELTS 6.0 thì được cộng điểm không?",
        "Xin chào bạn ơi",
        "Học phí mỗi học kỳ hết bao nhiêu tiền?",
        "Ngành Trí tuệ nhân tạo học những gì?",
        "Mình thích lập trình nên chọn ngành nào?",
        "Tôi có bằng cao đẳng muốn liên thông lên đại học",
        "Trường có ký túc xá không?",
        "Nếu trượt đợt 1 còn cơ hội không?",
        "CLC khác chương trình đại trà thế nào?",
    ]

    table = Table(title="Kết quả dự đoán", show_header=True)
    table.add_column("Câu hỏi", style="white", max_width=50)
    table.add_column("Intent", style="cyan", justify="center")
    table.add_column("Confidence", style="green", justify="center")

    for text in test_cases:
        proba = pipeline.predict_proba([text])[0]
        label = pipeline.classes_[np.argmax(proba)]
        conf = np.max(proba)
        table.add_row(text[:50], label, f"{conf:.2%}")

    console.print(table)


if __name__ == "__main__":
    pipeline = train()
    quick_test(pipeline)
