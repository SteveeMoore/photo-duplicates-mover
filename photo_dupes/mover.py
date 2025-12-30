import os
import argparse
import hashlib
import shutil
import sys
from typing import Dict, List, Tuple, Optional

from PIL import Image, ImageOps

# --- HEIC/HEIF support (via pillow-heif) ---
try:
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
except Exception:
    # If pillow-heif isn't installed or platform lacks support, HEIC won't open.
    pass


DEFAULT_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif",
    ".heic", ".heif", ".avif"
)


# ----------------------------
# Progress bar (no deps)
# ----------------------------
def progress_bar(current: int, total: int, prefix: str = "", width: int = 28) -> None:
    if total <= 0:
        return
    current = min(max(current, 0), total)
    pct = (current / total) * 100.0
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    msg = f"\r{prefix} [{bar}] {pct:6.2f}% ({current}/{total})"
    sys.stdout.write(msg)
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def sha256_file(fp: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(fp, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def dhash(fp: str, hash_size: int = 8) -> int:
    """
    dHash: grayscale -> resize to (hash_size+1, hash_size)
    compare adjacent pixels horizontally. Returns an integer bitmask.
    """
    with Image.open(fp) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("L")
        im = im.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)

        pixels = list(im.getdata())
        bits = 0
        bit_index = 0
        for row in range(hash_size):
            row_start = row * (hash_size + 1)
            for col in range(hash_size):
                left = pixels[row_start + col]
                right = pixels[row_start + col + 1]
                if left > right:
                    bits |= (1 << bit_index)
                bit_index += 1
        return bits


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_move(src: str, dst_dir: str) -> str:
    """
    Moves src file into dst_dir. If name exists, appends _1, _2, ...
    Returns final destination path.
    """
    base = os.path.basename(src)
    name, ext = os.path.splitext(base)
    dst = os.path.join(dst_dir, base)

    i = 1
    while os.path.exists(dst):
        dst = os.path.join(dst_dir, f"{name}_{i}{ext}")
        i += 1

    shutil.move(src, dst)
    return dst


def is_image_file(fp: str, fast_ext_filter: Optional[Tuple[str, ...]] = None) -> bool:
    """
    If fast_ext_filter is provided, first check extension membership.
    Then tries to open and verify file as image (Pillow).
    """
    if not os.path.isfile(fp):
        return False

    if fast_ext_filter is not None:
        ext = os.path.splitext(fp)[1].lower()
        if ext not in fast_ext_filter:
            return False

    try:
        with Image.open(fp) as im:
            im.verify()
        return True
    except Exception:
        return False


def iter_image_files(root: str, recursive: bool, use_ext_filter: bool) -> List[str]:
    """
    Collect candidate files.
    If use_ext_filter=False: try every file (slower, but most universal).
    """
    ext_filter = DEFAULT_EXTENSIONS if use_ext_filter else None

    # Сначала собираем кандидатов (по расширению или все файлы)
    candidates: List[str] = []
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                candidates.append(os.path.join(dirpath, fn))
    else:
        for fn in os.listdir(root):
            candidates.append(os.path.join(root, fn))

    # Затем проверяем, что это реально изображения
    out: List[str] = []
    total = len(candidates)
    for i, fp in enumerate(candidates, 1):
        if is_image_file(fp, ext_filter):
            out.append(fp)
        progress_bar(i, total, prefix="Scanning files")

    return out


def group_exact_duplicates(files: List[str]) -> List[List[str]]:
    """
    Groups exact duplicates by sha256. Returns list of groups with len>=2.
    """
    sha_groups: Dict[str, List[str]] = {}
    total = len(files)
    for i, fp in enumerate(files, 1):
        try:
            s = sha256_file(fp)
            sha_groups.setdefault(s, []).append(fp)
        except Exception:
            pass
        progress_bar(i, total, prefix="Hashing (SHA-256)")

    groups = []
    for g in sha_groups.values():
        if len(g) >= 2:
            groups.append(sorted(g))
    return groups


def group_visual_duplicates(files: List[str], hash_size: int, threshold: int) -> List[List[str]]:
    """
    Groups by dHash.
    - threshold==0: group by equal dhash
    - threshold>0: build graph by pairs within threshold (O(n^2))
    """
    hashes: List[Tuple[str, int]] = []
    total = len(files)
    for i, fp in enumerate(files, 1):
        try:
            h = dhash(fp, hash_size)
            hashes.append((fp, h))
        except Exception:
            pass
        progress_bar(i, total, prefix="Hashing (dHash)")

    if threshold == 0:
        by_h: Dict[int, List[str]] = {}
        for fp, h in hashes:
            by_h.setdefault(h, []).append(fp)
        return [sorted(g) for g in by_h.values() if len(g) >= 2]

    # threshold > 0 : build edges and components
    n = len(hashes)
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}

    # прогресс по внешнему циклу
    for i in range(n):
        _, h1 = hashes[i]
        for j in range(i + 1, n):
            _, h2 = hashes[j]
            if hamming_distance(h1, h2) <= threshold:
                adj[i].append(j)
                adj[j].append(i)
        progress_bar(i + 1, n, prefix="Comparing (Hamming)")

    seen = set()
    groups: List[List[str]] = []
    for i in range(n):
        if i in seen:
            continue
        stack = [i]
        comp: List[str] = []
        seen.add(i)
        while stack:
            v = stack.pop()
            comp.append(hashes[v][0])
            for nxt in adj[v]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        if len(comp) >= 2:
            groups.append(sorted(comp))
    return groups


def exclude_duplicates_dir(files: List[str], duplicates_dir: str) -> List[str]:
    duplicates_dir = os.path.abspath(duplicates_dir)
    out = []
    for fp in files:
        ap = os.path.abspath(fp)
        try:
            common = os.path.commonpath([ap, duplicates_dir])
        except ValueError:
            common = ""
        if common == duplicates_dir:
            continue
        out.append(fp)
    return out


def move_group_all(
    group: List[str],
    duplicates_dir: str,
    dry_run: bool,
    moved_set: Optional[set] = None,
) -> int:
    """
    Moves ALL files in the group into duplicates_dir.
    moved_set is used to avoid moving the same file twice.
    Returns count moved.
    """
    moved = 0
    for fp in group:
        if moved_set is not None:
            if fp in moved_set:
                continue
        if not os.path.exists(fp):
            continue

        if dry_run:
            print(f"[DRY-RUN] MOVE: {fp} -> {duplicates_dir}")
        else:
            safe_move(fp, duplicates_dir)

        if moved_set is not None:
            moved_set.add(fp)
        moved += 1
    return moved


def move_duplicates(
    root: str,
    recursive: bool,
    duplicates_dir_name: str,
    mode: str,
    hash_size: int,
    threshold: int,
    dry_run: bool,
    try_all_files: bool,
) -> int:
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise ValueError(f"Not a directory: {root}")

    duplicates_dir = os.path.join(root, duplicates_dir_name)
    ensure_dir(duplicates_dir)

    files = iter_image_files(root, recursive=recursive, use_ext_filter=(not try_all_files))
    files = exclude_duplicates_dir(files, duplicates_dir)

    moved = 0

    if mode in ("exact", "both"):
        groups = group_exact_duplicates(files)
        total_groups = len(groups)
        for gi, g in enumerate(groups, 1):
            # Переносим ВСЕ файлы группы
            moved += move_group_all(g, duplicates_dir, dry_run=dry_run)
            progress_bar(gi, total_groups, prefix="Moving (exact groups)")

        # обновим список файлов после переноса
        files = [fp for fp in files if os.path.exists(fp)]

    if mode in ("visual", "both"):
        groups = group_visual_duplicates(files, hash_size=hash_size, threshold=threshold)
        total_groups = len(groups)
        already_moved = set()
        for gi, g in enumerate(groups, 1):
            moved += move_group_all(g, duplicates_dir, dry_run=dry_run, moved_set=already_moved)
            progress_bar(gi, total_groups, prefix="Moving (visual groups)")

    return moved


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="photo-dupes-mover",
        description="Find image duplicates by content and move duplicates into Dublicates folder.",
    )
    p.add_argument("path", help="Path to folder with images.")

    p.add_argument("--recursive", action="store_true", help="Scan folders recursively.")

    p.add_argument(
        "--duplicates-dir",
        default="Dublicates",
        help='Duplicates folder name (created inside target folder). Default: "Dublicates".',
    )

    p.add_argument(
        "--mode",
        choices=["exact", "visual", "both"],
        default="exact",
        help="Duplicate detection mode: exact (SHA-256), visual (dHash), both. Default: exact.",
    )

    p.add_argument(
        "--hash-size",
        type=int,
        default=8,
        help="dHash size (8 => 64-bit). Used for visual/both. Default: 8.",
    )

    p.add_argument(
        "--threshold",
        type=int,
        default=0,
        help="Hamming distance threshold for visual duplicates. 0=strict. Used for visual/both. Default: 0.",
    )

    p.add_argument("--dry-run", action="store_true", help="Do not move files, only print what would be moved.")

    p.add_argument(
        "--try-all-files",
        action="store_true",
        help="Try to treat ANY file as an image (slower). If not set, uses extension filter + verify.",
    )

    return p


def main():
    args = build_parser().parse_args()

    moved = move_duplicates(
        root=args.path,
        recursive=args.recursive,
        duplicates_dir_name=args.duplicates_dir,
        mode=args.mode,
        hash_size=args.hash_size,
        threshold=args.threshold,
        dry_run=args.dry_run,
        try_all_files=args.try_all_files,
    )

    print(f"Done. Moved files: {moved}")
    print(f'Duplicates folder: {os.path.join(os.path.abspath(args.path), args.duplicates_dir)}')


if __name__ == "__main__":
    main()
