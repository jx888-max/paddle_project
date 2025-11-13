# PaddleOCR Evaluation Project

## Overview
This project evaluates PaddleOCR performance on various language datasets.

## Files
- `02_eval_with_paddleocr.py`: Main evaluation script with PaddleOCR
- `config.py`: Configuration for data paths

## Recent Fixes

### 1. Intel MKL Load Error Fix (Python 3.9)
**Issue**: `Intel MKL function load error: cpu specific dynamic library is not loaded.`

**Solution**: Added environment variable at the top of the script:
```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
```

This must be set **before** importing numpy, pandas, or paddleocr.

### 2. DeprecationWarning Fix
**Issue**: `DeprecationWarning: Please use 'predict' instead.`

**Solution**: Updated `safe_ocr_call()` function to:
1. Try `ocr.predict()` first (newer API)
2. Fall back to `ocr.ocr(det=False, cls=False)` if predict doesn't exist
3. Fall back to `ocr.ocr()` without parameters for older versions

## Usage

```bash
# Evaluate on validation set with Japanese language
python3 -u 02_eval_with_paddleocr.py --split val --lang japan 2>&1 | tee outputs/log_eval_val.txt

# Evaluate on training set with Korean language
python3 -u 02_eval_with_paddleocr.py --split train --lang korean 2>&1 | tee outputs/log_eval_train.txt

# Enable GPU
python3 -u 02_eval_with_paddleocr.py --split val --lang japan --use_gpu
```

## Arguments
- `--split`: Dataset split to evaluate (`train` or `val`, default: `val`)
- `--lang`: PaddleOCR language (e.g., `japan`, `korean`, `thai`, `arabic`, default: `japan`)
- `--use_gpu`: Enable GPU acceleration (flag)
- `--rec_thresh`: Minimum confidence threshold (0~1, default: 0.0)

## Requirements
- Python 3.9+
- paddleocr
- pandas
- tqdm

## Python 3.9 Compatibility
The code is fully compatible with Python 3.9 and includes workarounds for:
- Intel MKL threading issues
- PaddleOCR API deprecation warnings
- Multiple PaddleOCR version compatibility
