"""Restore the exact public source file without extracting untrusted archive paths."""
import io
import json
import urllib.request
import zipfile

from heart.config import RAW_PATH, ROOT
from heart.data import sha256

URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"


def main():
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((ROOT / "data/provenance.json").read_text(encoding="utf-8"))
    if RAW_PATH.exists():
        if sha256(RAW_PATH) != metadata["raw_sha256"]:
            raise ValueError("Existing file differs from the source checksum; inspect it before replacing it.")
        print("Verified local Cleveland dataset; no download needed.")
        return
    with urllib.request.urlopen(URL, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    matches = [n for n in archive.namelist() if n.split("/")[-1] == "processed.cleveland.data"]
    if len(matches) != 1:
        raise ValueError("Could not identify a unique processed Cleveland file.")
    payload = archive.read(matches[0])
    import hashlib
    if hashlib.sha256(payload).hexdigest() != metadata["raw_sha256"]:
        raise ValueError("Downloaded source differs from the recorded checksum; no file written.")
    RAW_PATH.write_bytes(payload)
    print(f"Saved verified source to {RAW_PATH}")


if __name__ == "__main__":
    main()
