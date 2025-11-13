#pathからcsvを読み込んで、encodeを変更して上書きする
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import os
import pathlib
import shutil
import sys
import tempfile
from typing import Optional

def detect_encoding(data: bytes) -> Optional[str]:
    """
    chardet/charset-normalizer が入っていれば使って推定。
    未導入でも動くようにフォールバックは None を返すだけ。
    """
    try:
        import chardet  # pip install chardet
        res = chardet.detect(data)
        return (res.get("encoding") or "").lower() or None
    except Exception:
        pass
    try:
        # pip install charset-normalizer
        from charset_normalizer import from_bytes
        best = from_bytes(data).best()
        if best and best.encoding:
            return best.encoding.lower()
    except Exception:
        pass
    return None

def transcode_csv(
    path: str | os.PathLike,
    src_encoding: Optional[str] = None,
    dst_encoding: str = "utf-8",
    add_bom: bool = False,
    make_backup: bool = False,
    newline: str = "\n",
) -> str:
    """
    CSVファイルのエンコードを変換して上書きする（原子的置換）。
    - src_encoding を省略すると自動推定を試みる（失敗時はよくある候補でトライ）
    - add_bom=True で UTF-8 BOM 付き（utf-8-sig）出力
    - newline: 出力時の改行（例: '\\n', '\\r\\n'）
    戻り値: 実際にデコードに使ったソースエンコード名
    """
    p = pathlib.Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")

    raw = p.read_bytes()

    # ソースエンコード決定
    used_src = (src_encoding or "").lower() or detect_encoding(raw)
    tried = []
    def try_decode(enc: str) -> Optional[str]:
        nonlocal text
        try:
            text = raw.decode(enc)
            return enc
        except Exception:
            tried.append(enc)
            return None

    text = None  # type: ignore
    if used_src:
        used_src = try_decode(used_src)

    if not used_src:
        # よくある日本語系や西欧系を順に試す
        for enc in ("cp932", "shift_jis", "euc_jp", "iso2022_jp", "utf-8", "utf-16", "latin-1"):
            if try_decode(enc):
                used_src = enc
                break

    if not used_src or text is None:
        raise UnicodeDecodeError("unknown", raw, 0, 1, f"decode failed; tried={tried}")

    # 出力エンコード
    out_enc = "utf-8-sig" if (dst_encoding.lower() in ("utf-8-sig", "utf_8_sig") or add_bom) else dst_encoding

    # 改行正規化
    if newline not in ("\n", "\r\n"):
        raise ValueError("newline must be '\\n' or '\\r\\n'")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if newline == "\r\n":
        text = text.replace("\n", "\r\n")

    # 一時ファイルへ書き出し（原子的置換）
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=p.name + ".", dir=str(p.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding=out_enc, newline="") as f:
            f.write(text)

        if make_backup:
            bak = p.with_suffix(p.suffix + ".bak")
            shutil.copy2(p, bak)

        os.replace(tmp_path, p)  # atomic
    except Exception:
        # 失敗時は一時ファイルを消す
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise

    return used_src

def main():
    ap = argparse.ArgumentParser(description="CSV のエンコードを変換して上書きする")
    ap.add_argument("path", help="変換対象のCSVファイルのパス")
    ap.add_argument("--src", dest="src", default=None, help="ソースエンコード（省略で自動推定）")
    ap.add_argument("--dst", dest="dst", default="utf-8", help="出力エンコード（既定: utf-8）")
    ap.add_argument("--bom", action="store_true", help="UTF-8 BOM を付けて出力する（utf-8-sig）")
    ap.add_argument("--backup", action="store_true", help="上書き前に .bak を作る")
    ap.add_argument("--crlf", action="store_true", help="改行を CRLF(Windows) にする（既定は LF）")
    args = ap.parse_args()

    try:
        used = transcode_csv(
            path=args.path,
            src_encoding=args.src,
            dst_encoding=args.dst,
            add_bom=args.bom,
            make_backup=args.backup,
            newline="\r\n" if args.crlf else "\n",
        )
        print(f"OK: decoded as {used}, wrote as {'utf-8-sig' if args.bom else args.dst}")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main(path='/Users/araya/Library/Mobile Documents/com~apple~CloudDocs/Downloads/Araya/persona_intelligence/materials/datasets/dataverse_files/JHPSDGs20192020_06012022/householdsurvey06012022withJapaneselabel.csv')

