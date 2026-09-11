"""Baixa o dataset PubMedQA (PQA-L) para data/raw/."""

import sys
import urllib.request

from src.config import PUBMEDQA_RAW_FILE, PUBMEDQA_RAW_URL


def download(force: bool = False) -> None:
    """Baixa o arquivo bruto."""
    if PUBMEDQA_RAW_FILE.exists() and not force:
        tamanho = PUBMEDQA_RAW_FILE.stat().st_size / 1024 / 1024
        print(f"[skip] {PUBMEDQA_RAW_FILE.name} ja existe ({tamanho:.1f} MB)")
        print("       use --force para baixar novamente")
        return

    print(f"[info] baixando de {PUBMEDQA_RAW_URL}")
    PUBMEDQA_RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(PUBMEDQA_RAW_URL, PUBMEDQA_RAW_FILE)

    tamanho = PUBMEDQA_RAW_FILE.stat().st_size / 1024 / 1024
    print(f"[ok] {PUBMEDQA_RAW_FILE} ({tamanho:.1f} MB)")


if __name__ == "__main__":
    download(force="--force" in sys.argv)
