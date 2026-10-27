"""Packages the official submission zip according to VoiceOps 2026 rules:
Rules:
- Include README.md, .env.example, requirements.txt, run.py, run.sh, src/, report.pdf, sample_outputs/
- Exclude .env, virtual environments, API keys, cache files, and the dataset copy.
"""
from pathlib import Path
import os
import zipfile

def package_submission(zip_name="team_voiceops_submission.zip"):
    root = Path(__file__).resolve().parent
    out_zip = root / zip_name

    included_top_files = [
        "README.md",
        ".env.example",
        "requirements.txt",
        "run.py",
        "run.sh",
        "report.pdf"
    ]

    included_dirs = [
        "src",
        "sample_outputs"
    ]

    print(f"[*] Packaging submission archive: {zip_name}...")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add top-level files
        for fname in included_top_files:
            fpath = root / fname
            if fpath.exists():
                zf.write(fpath, arcname=fname)
                print(f"  + Added: {fname}")
            else:
                print(f"  ! Warning: {fname} not found")

        # Add directories recursively
        for dname in included_dirs:
            dpath = root / dname
            for item in dpath.rglob("*"):
                if item.is_file():
                    # Filter out python bytecode or caches
                    if "__pycache__" in item.parts or item.suffix in (".pyc", ".pyo"):
                        continue
                    arcname = str(item.relative_to(root))
                    zf.write(item, arcname=arcname)
                    print(f"  + Added: {arcname}")

    print("\n" + "=" * 60)
    print(f"[SUCCESS] Package created successfully: {zip_name}")
    print(f"    File size: {round(os.path.getsize(out_zip) / (1024 * 1024), 2)} MB")
    print("=" * 60)

if __name__ == "__main__":
    package_submission()
