# Adaptive MA Type & Length Selection

This repository contains a single main script `s.py` that scans moving-average types and lengths, simulates simple TP/SL trades, and selects the best MA by final cash outcome.

Quick start

1. Install dependencies (if needed):

   ```powershell
   python -m pip install -r requirements.txt  # or install pandas, numpy, yfinance, matplotlib
   ```

2. Run interactively:

   ```powershell
   python s.py
   ```

3. Run headless with defaults:

   ```powershell
   python s.py --auto
   ```

Outputs

- All chart PNGs and a small JSON summary are saved into the `outputs/` folder next to the script. Example files:
  - `outputs/single_ma_scores.png` — percent results
  - `outputs/single_ma_dollars.png` — dollars gained/lost by length
  - `outputs/selected_ma.json` — selected MA summary and run settings

Backups

- Original auxiliary files (if any) were backed up in `backups/` before removal.

If you want any changes to defaults, output locations, or to restore archived files, tell me and I will update the project.
