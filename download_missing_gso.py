"""
Download missing GSO models from Gazebo Fuel.

Reads the full label set from id_to_label_ext.json, checks which folders
already exist in object_database/ycbv_gso/, and downloads + unzips only
the missing ones.

Usage (from OSCAR root):
    python3 download_missing_gso.py

Optional flags:
    --dry-run    List missing models without downloading
    --output     Target directory (default: object_database/ycbv_gso)
"""

import os
import sys
import json
import zipfile
import requests
import argparse
import io

# --- Config ---
LABELS_FILE = "helper_files/id_to_label_ext.json"
DEFAULT_OUTPUT = "object_database/ycbv_gso"
FUEL_BASE = "https://fuel.gazebosim.org/1.0"
OWNER = "GoogleResearch"

# YCBV objects start with digits like 002_, 003_ etc. — skip these (not on Fuel)
def is_ycbv(name):
    parts = name.split("_")
    return len(parts) >= 2 and parts[0].isdigit() and len(parts[0]) == 3


def main():
    parser = argparse.ArgumentParser(description="Download missing GSO models")
    parser.add_argument("--dry-run", action="store_true", help="Only list missing models")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output directory")
    args = parser.parse_args()

    # Load expected labels
    with open(LABELS_FILE) as f:
        id_to_label = json.load(f)

    all_labels = set(id_to_label.values())
    gso_labels = {name for name in all_labels if not is_ycbv(name)}

    # Check existing folders
    existing = set()
    if os.path.isdir(args.output):
        existing = set(os.listdir(args.output))

    missing = sorted(gso_labels - existing)

    print(f"Total GSO labels: {len(gso_labels)}")
    print(f"Already downloaded: {len(gso_labels & existing)}")
    print(f"Missing: {len(missing)}")

    if args.dry_run:
        for name in missing:
            print(f"  {name}")
        return

    if not missing:
        print("Nothing to download!")
        return

    os.makedirs(args.output, exist_ok=True)

    success = 0
    failed = []
    for i, model_name in enumerate(missing, 1):
        url = f"{FUEL_BASE}/{OWNER}/models/{model_name}.zip"
        print(f"[{i}/{len(missing)}] Downloading {model_name}...", end=" ", flush=True)

        try:
            r = requests.get(url, stream=True, timeout=60)
            if r.status_code != 200:
                print(f"FAILED (HTTP {r.status_code})")
                failed.append((model_name, f"HTTP {r.status_code}"))
                continue

            # Read into memory and extract
            zip_bytes = io.BytesIO()
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                zip_bytes.write(chunk)
            zip_bytes.seek(0)

            with zipfile.ZipFile(zip_bytes) as zf:
                # Fuel ZIPs have NO top-level folder — extract into a named subfolder
                model_dir = os.path.join(args.output, model_name)
                os.makedirs(model_dir, exist_ok=True)
                zf.extractall(model_dir)

            success += 1
            print("OK")

        except requests.exceptions.RequestException as e:
            print(f"FAILED ({e})")
            failed.append((model_name, str(e)))
        except zipfile.BadZipFile:
            print("FAILED (bad zip)")
            failed.append((model_name, "bad zip"))

    print(f"\nDone: {success} downloaded, {len(failed)} failed")
    if failed:
        print("\nFailed models:")
        for name, reason in failed:
            print(f"  {name}: {reason}")
        # Save failed list for retry
        with open("failed_downloads.txt", "w") as f:
            for name, reason in failed:
                f.write(f"{name}\t{reason}\n")
        print("Failed list saved to failed_downloads.txt")


if __name__ == "__main__":
    main()
