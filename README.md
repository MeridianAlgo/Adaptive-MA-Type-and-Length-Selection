# Adaptive MA Type & Length Selection

This repository contains a single main script `main.py` that scans moving-average types and lengths, simulates simple TP/SL trades, and selects the best MA by final cash outcome.

Quick start

1. Install dependencies (if needed):

   ```powershell
   python -m pip install -r requirements.txt  # or install pandas, numpy, yfinance, matplotlib
   ```

2. Run interactively:

   ```powershell
   python main.py
   ```

3. Run headless with defaults:

   ```powershell
   python main.py --auto
   ```

Outputs

- All chart PNGs and a small JSON summary are saved into the `outputs/` folder next to the script. Example files:
  - `outputs/single_ma_scores.png` — percent results
  - `outputs/single_ma_dollars.png` — dollars gained/lost by length
  - `outputs/selected_ma.json` — selected MA summary and run settings

Backups

- Original auxiliary files (if any) were backed up in `backups/` before removal.

If you want any changes to defaults, output locations, or to restore archived files, tell me and I will update the project.

Pinned dependencies

This project includes a `requirements.txt` with pinned versions known to work together (as of Oct 2025):

```
pandas==2.2.2
numpy==1.25.0
matplotlib==3.8.0
yfinance==0.2.29
```

Main script

Use `main.py` as the main entry point for scanning MAs and generating outputs. The older `strategy.py` file was archived and should not be used.
