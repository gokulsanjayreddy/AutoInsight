"""Windows-safe path sanitization and ZIP archive validation utilities for AutoInsight.

Ensures all generated ZIP archives and filenames extract cleanly and reliably
in Windows File Explorer (Compressed Folders shell extension) across Windows 10/11.
"""

from __future__ import annotations

import hashlib
import io
import re
import unicodedata
import zipfile

# Windows reserved device names (DOS device names).
# Cannot be used as stem (before extension) or directory name in Win32.
WINDOWS_RESERVED_NAMES: frozenset[str] = frozenset({
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "CLOCK$",
    "CONIN$",
    "CONOUT$",
    "COM0",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT0",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
})

# Windows forbidden characters: < > : " / \ | ? * and ASCII control characters (0-31, 127)
WINDOWS_INVALID_CHARS_RE = re.compile(r'[\x00-\x1f\x7f<>:"/\\|?*]')

# Redundant underscores pattern
MULTI_UNDERSCORE_RE = re.compile(r"_+")


def sanitize_filename_stem(
    name: str,
    max_len: int = 55,
    fallback: str = "item",
) -> str:
    """Sanitize a filename stem to be Windows-safe, readable, and bounded in length.

    Transformations:
    1. Normalize Unicode to NFC form.
    2. Replace Windows-invalid characters, spaces, and non-alphanumeric separators with '_'.
    3. Collapse consecutive underscores and strip leading/trailing dots, spaces, and underscores.
    4. Guard against empty names by falling back to `fallback`.
    5. Disambiguate Windows reserved device names (e.g. 'CON' -> 'CON_').
    6. Truncate long names sensibly to `max_len`, preserving a readable prefix and
       appending a deterministic short hash suffix (e.g., '..._A1B2').
    """
    if not name:
        name = fallback

    # 1. Unicode NFC normalization
    normalized = unicodedata.normalize("NFC", name)

    # 2. Replace invalid / separator characters with underscore
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in normalized)

    # 3. Collapse multiple underscores
    cleaned = MULTI_UNDERSCORE_RE.sub("_", cleaned)

    # Strip leading and trailing dots, spaces, underscores
    cleaned = cleaned.strip(". _")

    # 4. Fallback if empty or all characters were stripped
    if not cleaned:
        cleaned = fallback

    # 5. Check Windows reserved device names (case-insensitive check)
    if cleaned.upper() in WINDOWS_RESERVED_NAMES:
        cleaned = f"{cleaned}_"

    # 6. Length truncation
    if len(cleaned) > max_len:
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:4].upper()
        hash_suffix = f"_..._{digest}"
        keep_len = max(1, max_len - len(hash_suffix))
        prefix = cleaned[:keep_len].rstrip("._")
        if not prefix:
            prefix = cleaned[:keep_len]
        cleaned = f"{prefix}{hash_suffix}"
        if cleaned.upper() in WINDOWS_RESERVED_NAMES:
            cleaned = f"{cleaned}_"

    return cleaned


def sanitize_folder_component(
    folder: str,
    max_len: int = 40,
    fallback: str = "plots",
) -> str:
    """Sanitize a directory path to be Windows-safe, preserving subfolder structure if any."""
    folder = folder.replace("\\", "/").strip("/ ")
    parts = [p for p in folder.split("/") if p and p != "." and p != ".."]
    if not parts:
        return fallback

    sanitized_parts = []
    for part in parts:
        clean_part = sanitize_filename_stem(part, max_len=max_len, fallback=fallback)
        sanitized_parts.append(clean_part)

    return "/".join(sanitized_parts)


def build_safe_zip_path(
    folder: str,
    name_base: str,
    ext: str = ".png",
    used_paths_casefolded: set[str] | None = None,
    max_stem_len: int = 55,
) -> str:
    """Build a Windows-safe relative ZIP path with case-insensitive collision handling."""
    clean_folder = sanitize_folder_component(folder)

    # Strip extension if already present in name_base
    if ext and name_base.lower().endswith(ext.lower()):
        name_base = name_base[:-len(ext)]

    clean_stem = sanitize_filename_stem(name_base, max_len=max_stem_len)
    candidate = f"{clean_folder}/{clean_stem}{ext}"

    if used_paths_casefolded is None:
        return candidate

    cf = candidate.casefold()
    if cf not in used_paths_casefolded:
        used_paths_casefolded.add(cf)
        return candidate

    # Collision detected: generate deterministic suffix _2, _3, ...
    counter = 2
    suffix_sep = "" if clean_stem.endswith("_") else "_"
    while True:
        suffixed = f"{clean_folder}/{clean_stem}{suffix_sep}{counter}{ext}"
        suffixed_cf = suffixed.casefold()
        if suffixed_cf not in used_paths_casefolded:
            used_paths_casefolded.add(suffixed_cf)
            return suffixed
        counter += 1


def _validate_path_component(path: str, part: str) -> None:
    """Validate a single path component for Windows filesystem safety."""
    if not part or part == ".":
        raise ValueError(f"ZIP entry '{path}' contains empty or '.' component.")
    if part == "..":
        raise ValueError(
            f"ZIP entry '{path}' contains directory traversal '..' component."
        )

    # Trailing dot or space on full component
    if part.endswith(" "):
        raise ValueError(
            f"ZIP entry '{path}' component '{part}' has trailing space (invalid on Windows)."
        )
    if part.endswith("."):
        raise ValueError(
            f"ZIP entry '{path}' component '{part}' has trailing period (invalid on Windows)."
        )

    # For files with extension, also check if stem ends with dot or space
    if "." in part:
        part_stem = part.rsplit(".", 1)[0]
        if part_stem.endswith(" "):
            raise ValueError(
                f"ZIP entry '{path}' component stem '{part_stem}' has trailing space before "
                "extension (invalid on Windows)."
            )
        if part_stem.endswith("."):
            raise ValueError(
                f"ZIP entry '{path}' component stem '{part_stem}' has trailing period before "
                "extension (invalid on Windows)."
            )

    # Leading space
    if part.startswith(" "):
        raise ValueError(
            f"ZIP entry '{path}' component '{part}' has leading space (invalid on Windows)."
        )

    # Invalid characters
    invalid_match = WINDOWS_INVALID_CHARS_RE.search(part)
    if invalid_match:
        raise ValueError(
            f"ZIP entry '{path}' component '{part}' contains Windows-invalid character: "
            f"{invalid_match.group()!r}"
        )

    # Reserved device names
    stem = part.split(".")[0].upper()
    if stem in WINDOWS_RESERVED_NAMES or part.upper() in WINDOWS_RESERVED_NAMES:
        raise ValueError(
            f"ZIP entry '{path}' component '{part}' uses Windows reserved device name '{stem}'."
        )


def validate_zip_archive(
    zip_bytes: bytes,
    max_path_len: int = 180,
    allow_empty: bool = False,
) -> None:
    """Validate in-memory ZIP archive for integrity and complete Windows compatibility.

    Raises
    ------
    ValueError
        If any validation rule is violated, with a clear explanation of the offending entry.
    """
    if not zipfile.is_zipfile(io.BytesIO(zip_bytes)):
        raise ValueError("Invalid ZIP archive: data is not a valid zip archive.")

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        corrupted_entry = zf.testzip()
        if corrupted_entry is not None:
            raise ValueError(f"Corrupted ZIP entry detected: '{corrupted_entry}' failed CRC check.")

        infolist = zf.infolist()
        if not infolist and not allow_empty:
            raise ValueError("Invalid ZIP archive: archive contains 0 entries.")

        seen_exact: set[str] = set()
        seen_casefolded: set[str] = set()

        for info in infolist:
            path = info.filename

            # 1. Non-empty check
            if not path or not path.strip():
                raise ValueError(f"Invalid ZIP entry: empty or whitespace-only filename '{path}'.")

            # 2. Exact duplicate check
            if path in seen_exact:
                raise ValueError(f"Duplicate ZIP entry detected: '{path}' appears multiple times.")
            seen_exact.add(path)

            # 3. Case-insensitive collision check
            cf = path.casefold()
            if cf in seen_casefolded:
                raise ValueError(
                    f"Windows case-insensitive collision detected: '{path}' collides with an "
                    "existing entry."
                )
            seen_casefolded.add(cf)

            # 4. Path separator check: must use '/', never '\'
            if "\\" in path:
                raise ValueError(f"ZIP entry contains backslash path separator: '{path}'")

            # Must not have leading slash
            if path.startswith("/"):
                raise ValueError(f"ZIP entry contains forbidden leading slash: '{path}'")

            # 5. Path length check
            if len(path) > max_path_len:
                raise ValueError(
                    f"ZIP entry path length ({len(path)}) exceeds Windows limit "
                    f"({max_path_len}): '{path}'"
                )

            # 6. Check each component
            normalized_path = path[:-1] if path.endswith("/") else path
            parts = normalized_path.split("/")
            for part in parts:
                _validate_path_component(path, part)
