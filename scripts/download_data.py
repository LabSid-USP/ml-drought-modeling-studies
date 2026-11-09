#!/usr/bin/env python
"""Baixa os dados do artigo de previsão (HydroShare) para os caminhos do repositório.

Uso (na raiz do repositório)::

    python scripts/download_data.py                       # R1 (séries + saídas) e R3 (camadas)
    python scripts/download_data.py --only R1
    python scripts/download_data.py --from-dir build/hydroshare   # pacotes locais (teste)

Cada resource traz um ``MANIFEST-sha256.txt`` com o caminho (relativo à raiz
do repositório) e o SHA-256 de cada arquivo. O script lê o manifesto, baixa
cada arquivo para o mesmo caminho, confere o hash e pula os que já estão
corretos. Só usa a biblioteca padrão.

Resources (ver README, seção *Data*):

* R1 — séries mensais de entrada e saídas dos modelos → ``Previsao/Dados/``,
  ``Previsao/Outputs/``;
* R3 — camadas espaciais → ``Previsao/Mapa/dados/``.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESOURCES = {
    "R1": "4d705def5c5f4fd6b46c29dde302e846",
    "R3": "4141d4f21beb415b8099e967f743f633",
}
URL = "https://www.hydroshare.org/resource/{rid}/data/contents/{path}"


def sha256(path: Path) -> str:
    """SHA-256 de um arquivo local."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, rel: str, dst: Path, from_dir: Path | None) -> None:
    """Copia (``--from-dir``) ou baixa ``rel`` do resource ``name`` para ``dst``."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if from_dir is not None:
        shutil.copyfile(from_dir / name / rel, dst)
        return
    rid = RESOURCES[name]
    if rid.startswith("{{"):
        raise SystemExit(f"{name}: identificador do resource ainda não definido neste release")
    with urllib.request.urlopen(URL.format(rid=rid, path=rel), timeout=120) as r, \
            dst.open("wb") as fh:
        shutil.copyfileobj(r, fh)


def manifest(name: str, from_dir: Path | None) -> list[tuple[str, str]]:
    """Pares (sha256, caminho) do manifesto do resource."""
    tmp = REPO / "build" / "download" / name / "MANIFEST-sha256.txt"
    fetch(name, "MANIFEST-sha256.txt", tmp, from_dir)
    pairs = []
    for line in tmp.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split(maxsplit=1)
        if rel.startswith("/") or ".." in Path(rel).parts:
            raise SystemExit(f"{name}: caminho inválido no manifesto: {rel}")
        pairs.append((digest, rel))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=sorted(RESOURCES), action="append",
                    help="baixa só este resource (repetível)")
    ap.add_argument("--from-dir", type=Path,
                    help="pasta com os pacotes R1/ e R3/ já montados (sem rede)")
    args = ap.parse_args()

    bad = 0
    for name in args.only or sorted(RESOURCES):
        pairs = manifest(name, args.from_dir)
        got = 0
        for digest, rel in pairs:
            dst = REPO / rel
            if dst.exists() and sha256(dst) == digest:
                continue
            fetch(name, rel, dst, args.from_dir)
            if sha256(dst) != digest:
                print(f"ERRO: hash não confere: {rel}", file=sys.stderr)
                bad += 1
            got += 1
        print(f"{name}: {len(pairs)} arquivos ({got} baixados, {len(pairs) - got} já corretos)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
