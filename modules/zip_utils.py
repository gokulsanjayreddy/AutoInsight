"""Windows-safe path sanitization and ZIP archive validation utilities for AutoInsight.

Ensures all generated ZIP archives and filenames extract cleanly and reliably
in Windows File Explorer (Compressed Folders shell extension) across Windows 10/11.
Re-exports the core implementations defined in modules.eda.
"""

from __future__ import annotations

from modules.eda import (
    MULTI_UNDERSCORE_RE,
    WINDOWS_INVALID_CHARS_RE,
    WINDOWS_RESERVED_NAMES,
    build_safe_zip_path,
    generate_safe_zip_entry_path,
    normalize_for_collision,
    sanitize_filename_stem,
    sanitize_folder_component,
    validate_zip_archive,
)

__all__ = [
    "MULTI_UNDERSCORE_RE",
    "WINDOWS_INVALID_CHARS_RE",
    "WINDOWS_RESERVED_NAMES",
    "build_safe_zip_path",
    "generate_safe_zip_entry_path",
    "normalize_for_collision",
    "sanitize_filename_stem",
    "sanitize_folder_component",
    "validate_zip_archive",
]
