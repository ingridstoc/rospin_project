import cv2
import matplotlib.pyplot as plt
import random
from pathlib import Path

def compare_images(original_path: Path, filtered_path: Path):
    orig = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
    filt = cv2.imread(str(filtered_path), cv2.IMREAD_GRAYSCALE)

    if orig is None or filt is None:
        print(f"⚠️ Nu pot citi {original_path} sau {filtered_path}")
        return

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(orig, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(filt, cmap="gray")
    plt.title("Mean filtered")
    plt.axis("off")

    plt.suptitle(original_path.name, fontsize=12)
    plt.tight_layout()
    plt.show()

def show_random_samples(root_dir, preproc_dir, n=5):
    root = Path(root_dir)
    preproc = Path(preproc_dir)

    # listăm toate imaginile din root (recursiv)
    all_imgs = [p for p in root.rglob("*") if p.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg")]
    if len(all_imgs) == 0:
        print("❌ Nu am găsit imagini în root_dir")
        return

    # alegem n aleatoare
    samples = random.sample(all_imgs, min(n, len(all_imgs)))

    for orig in samples:
        rel = orig.relative_to(root)   # ex: LawEnforce/Ship_C06S01N0004.tiff
        filt = preproc / rel
        compare_images(orig, filt)

if __name__ == "__main__":
    root_dir = r"C:\Users\Andreea\Desktop\rospin_project\data"
    preproc_dir = r"C:\Users\Andreea\Desktop\rospin_project\data\preprocessed"
    show_random_samples(root_dir, preproc_dir, n=5)
