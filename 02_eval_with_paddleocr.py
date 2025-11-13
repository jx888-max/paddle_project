# coding: utf-8
import os

# Set MKL environment variables before any imports to avoid runtime errors
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from paddleocr import PaddleOCR

from config import CLEAN_DIR

def load_split(split: str) -> pd.DataFrame:
    csv_path = CLEAN_DIR / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run 01_preprocess_and_split.py first.")
    return pd.read_csv(csv_path)

def extract_text_conf(result):
    """
    兼容不同 PaddleOCR 版本的 ocr() 返回结构，返回 (text:str, conf:float)。
    常见结构：
      - det=False: [[('文本', 置信度)]]
      - det=True:  [ [ [box], ('文本', 置信度) ], ... ]
    """
    if not isinstance(result, list) or not result:
        return "", 0.0

    first = result[0]
    # 形如 [[('text', conf), ...]]
    if isinstance(first, list) and first:
        # 直接取第一个候选
        if isinstance(first[0], (tuple, list)) and len(first[0]) >= 2 and isinstance(first[0][0], str):
            text = str(first[0][0])
            try:
                conf = float(first[0][1])
            except Exception:
                conf = 0.0
            return text, conf
        # 形如 [[ [box], ('text', conf) ], ...]
        if len(first) >= 2 and isinstance(first[1], (tuple, list)) and len(first[1]) >= 2 and isinstance(first[1][0], str):
            text = str(first[1][0])
            try:
                conf = float(first[1][1])
            except Exception:
                conf = 0.0
            return text, conf

    return "", 0.0

def build_ocr(lang: str, use_gpu: bool):
    """
    兼容不同 paddleocr 版本的构造参数形式：
    1) PaddleOCR(lang=..., use_gpu=...)
    2) PaddleOCR(lang=...)
    3) PaddleOCR()
    """
    # 1) lang + use_gpu
    try:
        print(f"[build_ocr] try PaddleOCR(lang='{lang}', use_gpu={use_gpu})", flush=True)
        return PaddleOCR(lang=lang, use_gpu=use_gpu), f"lang={lang}, use_gpu={use_gpu}"
    except Exception as e1:
        print(f"[build_ocr] fallback 1 due to: {e1}", flush=True)
    # 2) only lang
    try:
        print(f"[build_ocr] try PaddleOCR(lang='{lang}')", flush=True)
        return PaddleOCR(lang=lang), f"lang={lang}"
    except Exception as e2:
        print(f"[build_ocr] fallback 2 due to: {e2}", flush=True)
    # 3) no args
    print(f"[build_ocr] try PaddleOCR()  (using package defaults)", flush=True)
    return PaddleOCR(), "default"

def safe_ocr_call(ocr, img_path: str):
    """
    兼容不同 paddleocr 版本的 ocr() 和 predict() 调用：
    1) ocr.predict(img=...)
    2) ocr.ocr(img=..., det=False, cls=False)
    3) ocr.ocr(img=...)
    """
    # Try predict() first (newer API)
    if hasattr(ocr, 'predict'):
        try:
            return ocr.predict(img_path)
        except Exception as e:
            print(f"[safe_ocr_call] predict() failed: {e}, falling back to ocr()", flush=True)
    
    # Fall back to ocr() with parameters
    try:
        return ocr.ocr(img=img_path, det=False, cls=False)
    except TypeError:
        # 旧版/异构版本不接受 det/cls 关键字
        return ocr.ocr(img=img_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="val", choices=["train", "val"], help="which split to evaluate")
    ap.add_argument("--lang", type=str, default="japan", help="PaddleOCR language (japan, korean, thai, arabic, etc.)")
    ap.add_argument("--use_gpu", action="store_true", help="enable GPU if available")
    ap.add_argument("--rec_thresh", type=float, default=0.0, help="min confidence to accept prediction (0~1)")
    args = ap.parse_args()

    df = load_split(args.split)
    out_csv = CLEAN_DIR / f"ocr_eval_{args.split}.csv"

    print(f"Loading PaddleOCR (requested lang={args.lang}, use_gpu={args.use_gpu}) ...", flush=True)
    ocr, mode_desc = build_ocr(args.lang, args.use_gpu)
    print(f"[build_ocr] success with mode: {mode_desc}", flush=True)

    records = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="PaddleOCR eval"):
        p = str(row.path)
        gt = str(row.label)

        try:
            result = safe_ocr_call(ocr, p)
        except Exception as e:
            records.append({
                "path": p, "label": gt,
                "pred": "", "conf": 0.0,
                "is_correct": 0, "err": f"ocr_error:{e}"
            })
            continue

        pred_text, conf = extract_text_conf(result)
        pred_char = pred_text[0] if len(pred_text) > 0 else ""
        ok = (pred_char == gt) and (conf >= args.rec_thresh)

        records.append({
            "path": p, "label": gt,
            "pred": pred_char, "conf": conf,
            "is_correct": int(ok), "err": ""
        })

    out_df = pd.DataFrame(records)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"✅ Eval saved: {out_csv} (rows={len(out_df)})", flush=True)

if __name__ == "__main__":
    main()
