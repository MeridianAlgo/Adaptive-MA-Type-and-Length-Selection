import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import sys
import math
from datetime import datetime, timedelta

# small helper to coerce pandas/numpy objects to plain Python scalars
def scalarize(x, fallback=0.0):
    try:
        if isinstance(x, pd.Series):
            if x.empty:
                return fallback
            if x.size == 1:
                return x.iloc[0]
            return float(x.mean())
        arr = np.asarray(x)
        if arr.size == 0:
            return fallback
        if arr.size == 1:
            # return python scalar
            return arr.item()
        return float(np.nanmean(arr))
    except Exception:
        return fallback

# --- simple interactive interface (defaults used if input left blank) ---
def get_settings():
    def ask(prompt, cast, default):
        val = input(f"{prompt} [{default}]: ")
        if val.strip() == "":
            return default
        try:
            return cast(val)
        except Exception:
            print("Invalid input, using default.")
            return default

    print("--- MA scan settings (press Enter to accept default) ---")
    ticker = input("Ticker (e.g. BTC-USD) ") or 'BTC-USD'
    period = input("Period (e.g. 5d) ") or '5d'
    interval = input("Interval (e.g. 1m) ") or '1m'
    max_len = ask("Max MA length to scan (int)", int, 30)
    tp_dollars = ask("TP in dollars (float)", float, 1.5)
    sl_offset = ask("SL offset below MA in dollars (float)", float, 1.0)
    initial_cash = ask("Initial cash per MA (float)", float, 10000.0)
    max_holding = ask("Max holding in bars (int)", int, 60)
    ranking = input("Ranking method ('points' or 'final_cash') [points]: ") or 'points'
    return {
        'ticker': ticker,
        'period': period,
        'interval': interval,
        'max_len': max_len,
        'tp_dollars': tp_dollars,
        'sl_offset': sl_offset,
        'initial_cash': initial_cash,
        'max_holding': max_holding,
        'ranking': ranking
    }

def clamp_settings(settings):
    # enforce sane limits
    settings['max_len'] = max(1, min(200, int(settings.get('max_len', 30))))
    settings['max_holding'] = max(1, min(1440, int(settings.get('max_holding', 60))))
    settings['tp_dollars'] = float(max(0.0, settings.get('tp_dollars', 1.5)))
    settings['sl_offset'] = float(max(0.0, settings.get('sl_offset', 1.0)))
    settings['initial_cash'] = float(max(1.0, settings.get('initial_cash', 10000.0)))
    if settings.get('interval') not in ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d']:
        settings['interval'] = '1m'
    if settings.get('ranking') not in ['points', 'final_cash']:
        settings['ranking'] = 'points'
    return settings

def parse_period_to_days(period_str):
    # simple parser: '7d'->7, '1mo'->30, '90m'->0 (minutes not supported for long periods)
    try:
        if period_str.endswith('d'):
            return int(period_str[:-1])
        if period_str.endswith('mo'):
            return int(period_str[:-2]) * 30
        if period_str.endswith('y'):
            return int(period_str[:-1]) * 365
    except Exception:
        pass
    # fallback default
    return 7

def download_with_granularity(ticker, period='7d', interval='1m'):
    """Download data, auto-chunking 1m requests when period > 8 days.
    Returns a DataFrame or raises the yfinance exception.
    """
    days = parse_period_to_days(period)
    # yahoo allows about 7-8 days of 1m data per request; chunk into 7-day windows
    if interval == '1m' and days > 7:
        end = datetime.utcnow()
        start = end - timedelta(days=days)
        frames = []
        cur = start
        # chunk size 7 days
        chunk = timedelta(days=7)
        while cur < end:
            s = cur
            e = min(end, cur + chunk)
            try:
                df = yf.download(ticker, start=s.strftime('%Y-%m-%d'), end=(e + timedelta(days=1)).strftime('%Y-%m-%d'), interval=interval, progress=False)
            except Exception as ex:
                # propagate if any chunk fails
                raise
            if not df.empty:
                frames.append(df)
            cur = e
        if not frames:
            return pd.DataFrame()
        df_all = pd.concat(frames).sort_index().drop_duplicates()
        return df_all
    else:
        return yf.download(ticker, period=period, interval=interval, progress=False)

# Basic moving average helpers
def sma(src, length):
    if length <= 0:
        return src
    return src.rolling(window=length).mean()

def ema(src, length):
    if length <= 0:
        return src
    return src.ewm(span=length, adjust=False).mean()

def wma(src, length):
    if length <= 0:
        return src
    weights = np.arange(1, length + 1)
    def _wma(x):
        return np.dot(x, weights) / weights.sum()
    return src.rolling(window=length).apply(_wma, raw=True)

def rma(src, length):
    # Wilder's moving average (RMA)
    if length <= 0:
        return src
    alpha = 1.0 / length
    return src.ewm(alpha=alpha, adjust=False).mean()

def linreg(src, length):
    if length <= 0:
        return src
    def _lr(x):
        n = len(x)
        if n == 0:
            return np.nan
        y = np.asarray(x)
        xs = np.arange(n)
        # linear fit
        A = np.vstack([xs, np.ones(n)]).T
        try:
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            return m * (n - 1) + c
        except Exception:
            return np.nan
    return src.rolling(window=length).apply(_lr, raw=True)
def hma(src, length):
    if length < 2:
        return src
    half_len = length // 2
    wma_half = wma(src, half_len)
    wma_full = wma(src, length)
    raw_hma = 2 * wma_half - wma_full
    sqrt_len = int(np.sqrt(length))
    return wma(raw_hma, sqrt_len)
def alma(src, length, offset=0.85, sigma=6.0):
    def alma_x(x):
        n = len(x)
        m = np.arange(n)
        s = n / sigma
        w = np.exp(-((m - (n - 1) * offset)**2) / (2 * s**2))
        norm_w = w / w.sum()
        return np.dot(norm_w, x)
    return src.rolling(window=length).apply(alma_x, raw=True)
def vwma(src, volume, length): return (src * volume).rolling(window=length).sum() / volume.rolling(window=length).sum()

def get_ma(ma_type, src, length, volume=None, alma_offset=0.85, alma_sigma=6.0):
    if ma_type == 'SMA': return sma(src, length)
    elif ma_type == 'EMA': return ema(src, length)
    elif ma_type == 'WMA': return wma(src, length)
    elif ma_type == 'Hull': return hma(src, length)
    elif ma_type == 'VWMA':
        if volume is None:
            raise ValueError("Volume required for VWMA")
        return vwma(src, volume, length)
    elif ma_type == 'ALMA': return alma(src, length, alma_offset, alma_sigma)
    elif ma_type == 'RMA': return rma(src, length)
    elif ma_type == 'LINREG': return linreg(src, length)

def crossover(a, b, i):
    return (a.values[-i] > b.values[-i]) and (a.values[-(i+1)] <= b.values[-(i+1)])
def crossunder(a, b, i):
    return (a.values[-i] < b.values[-i]) and (a.values[-(i+1)] >= b.values[-(i+1)])

def calculate_signals(src, ma, eval_period):
    n = len(src)
    max_i = min(eval_period, n - 1)
    if max_i <= 0:
        return 0, 0, 0
    signals_index = src.tail(max_i).index
    signals = pd.Series(0, index=signals_index)
    positions = pd.Series(0, index=signals_index)
    for i in range(1, max_i + 1):
        if crossover(src, ma, i):
            signals.iloc[-i] = 1
            positions.iloc[-i] = 1
        elif crossunder(src, ma, i):
            signals.iloc[-i] = -1
            positions.iloc[-i] = -1
    delta = src.shift(-1) - src
    delta_pos = delta.loc[positions.index]
    src_pos = src.loc[positions.index]
    total_profit = (positions * delta_pos).sum()
    # avoid division by zero and ensure we operate on aligned numeric Series
    # src_pos may contain zeros; replace zeros with NaN to avoid infs
    safe_src = src_pos.replace(0, np.nan)
    pct_delta = delta_pos / safe_src.abs()
    trade_returns = (positions * pct_delta).dropna()
    # trade_returns is now a 1-D numeric Series; compute scalar std
    if trade_returns.empty:
        sharpe = 0
    else:
        std_val = trade_returns.std()
        if pd.isna(std_val) or std_val <= 0:
            sharpe = 0
        else:
            sharpe = (trade_returns.mean() / std_val * np.sqrt(252 * 390))
    num_signals = (signals != 0).sum()
    # coerce return values to Python scalars to avoid Series/array surprises
    try:
        total_profit = float(total_profit)
    except Exception:
        # fallback: try to reduce to a scalar
        total_profit = float(np.nanmean(np.asarray(total_profit)))
    try:
        sharpe = float(sharpe)
    except Exception:
        sharpe = float(np.nanmean(np.asarray(sharpe)))
    try:
        num_signals = int(num_signals)
    except Exception:
        num_signals = int(np.nanmax(np.asarray(num_signals)))

    return total_profit, sharpe, num_signals

def score_ma(src, ma, length, eval_period):
    profit, sharpe, num_signals = calculate_signals(src, ma, eval_period)
    if num_signals == 0:
        return -np.inf
    # coerce profit and sharpe to scalars if they are Series/arrays
    def to_scalar(x):
        if isinstance(x, pd.Series):
            if x.size == 0:
                return 0.0
            if x.size == 1:
                return x.iloc[0]
            return x.mean()
        arr = np.asarray(x)
        if arr.size == 0:
            return 0.0
        if arr.size == 1:
            return arr.item()
        return float(np.nanmean(arr))

    p = to_scalar(profit)
    s = to_scalar(sharpe)
    score = (p * s) / length
    return score


def calculate_pair_signals(src, ma_fast, ma_slow, eval_period):
    # Signals based on fast MA crossing slow MA
    n = len(src)
    max_i = min(eval_period, n - 1)
    if max_i <= 0:
        return 0, 0, 0
    signals_index = src.tail(max_i).index
    signals = pd.Series(0, index=signals_index)
    positions = pd.Series(0, index=signals_index)
    for i in range(1, max_i + 1):
        if crossover(ma_fast, ma_slow, i):
            signals.iloc[-i] = 1
            positions.iloc[-i] = 1
        elif crossunder(ma_fast, ma_slow, i):
            signals.iloc[-i] = -1
            positions.iloc[-i] = -1
    delta = src.shift(-1) - src
    delta_pos = delta.loc[positions.index]
    src_pos = src.loc[positions.index]
    total_profit = (positions * delta_pos).sum()
    # safe pct returns
    safe_src = src_pos.replace(0, np.nan)
    pct_delta = delta_pos / safe_src.abs()
    trade_returns = (positions * pct_delta).dropna()
    if trade_returns.empty:
        sharpe = 0
    else:
        std_val = trade_returns.std()
        if pd.isna(std_val) or std_val <= 0:
            sharpe = 0
        else:
            sharpe = (trade_returns.mean() / std_val * np.sqrt(252 * 390))
    num_signals = (signals != 0).sum()
    try:
        total_profit = float(total_profit)
    except Exception:
        total_profit = float(np.nanmean(np.asarray(total_profit)))
    try:
        sharpe = float(sharpe)
    except Exception:
        sharpe = float(np.nanmean(np.asarray(sharpe)))
    try:
        num_signals = int(num_signals)
    except Exception:
        num_signals = int(np.nanmax(np.asarray(num_signals)))
    return total_profit, sharpe, num_signals


def score_pair(src, ma_fast, ma_slow, len_fast, len_slow, eval_period):
    profit, sharpe, num_signals = calculate_pair_signals(src, ma_fast, ma_slow, eval_period)
    if num_signals == 0:
        return -np.inf
    # coerce scalars (should already be scalars but be defensive)
    def s(x):
        try:
            return float(x)
        except Exception:
            return float(np.nanmean(np.asarray(x)))
    p = s(profit)
    sh = s(sharpe)
    # score penalizes longer combined lengths to prefer simpler setups
    score = (p * sh) / (len_fast + len_slow)
    return score


def find_best_ma_pairs(df, train_split=0.8, max_length=40, eval_period=50, alma_offset=0.85, alma_sigma=6.0):
    """Search across MA-type pairs and length combinations.
    Defaults use max_length=40 to keep runtime reasonable. Increase if needed.
    """
    split_idx = int(len(df) * train_split)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    src_train = train_df['Close']
    vol_train = train_df['Volume']

    types = ['EMA', 'SMA', 'WMA', 'Hull', 'VWMA', 'ALMA', 'RMA', 'LINREG']
    best_score = -np.inf
    best_combo = None

    for t_fast in types:
        for t_slow in types:
            # allow same types but require fast length < slow length when searching
            for len_fast in range(1, max_length):
                for len_slow in range(len_fast + 1, max_length + 1):
                    vol_fast = vol_train if t_fast == 'VWMA' else None
                    vol_slow = vol_train if t_slow == 'VWMA' else None
                    ma_fast = get_ma(t_fast, src_train, len_fast, vol_fast, alma_offset, alma_sigma)
                    ma_slow = get_ma(t_slow, src_train, len_slow, vol_slow, alma_offset, alma_sigma)
                    score = score_pair(src_train, ma_fast, ma_slow, len_fast, len_slow, eval_period)
                    if score > best_score:
                        best_score = score
                        best_combo = (t_fast, len_fast, t_slow, len_slow)

    # Backtest best pair on test set
    if best_combo is None:
        return None
    t_fast, len_fast, t_slow, len_slow = best_combo
    vol_fast_bt = test_df['Volume'] if t_fast == 'VWMA' else None
    vol_slow_bt = test_df['Volume'] if t_slow == 'VWMA' else None
    ma_fast_test = get_ma(t_fast, test_df['Close'], len_fast, vol_fast_bt, alma_offset, alma_sigma)
    ma_slow_test = get_ma(t_slow, test_df['Close'], len_slow, vol_slow_bt, alma_offset, alma_sigma)
    test_profit, test_sharpe, test_signals = calculate_pair_signals(test_df['Close'], ma_fast_test, ma_slow_test, min(len(test_df), eval_period))

    return best_combo, best_score, test_profit, test_sharpe, test_signals


def simulate_single_ma_trades(src, ma, max_holding=60, tp_sl_ratio=1.5, sl_offset_dollars=1.0, tp_dollars=1.5, initial_cash=10000.0):
    """Simulate long trades when price crosses above MA.
    - Entry is next bar's price (we'll use next bar's Close as proxy).
    - SL is MA value at signal bar.
    - TP = entry + tp_sl_ratio * (entry - SL)
    - Stop loss or take profit can occur within max_holding bars; if neither, close at last available price.
    Returns total_percent_profit, num_trades
    """
    # convert to numpy float arrays for safe positional operations
    try:
        src_vals = np.asarray(src, dtype=float)
    except Exception:
        src_vals = np.asarray(src).astype(float)
    # ensure ma is 1-D and aligned
    if isinstance(ma, pd.DataFrame):
        try:
            ma = ma.squeeze()
        except Exception:
            ma = ma.iloc[:, 0]
    ma = ma.reindex(src.index)
    try:
        ma_vals = np.asarray(ma, dtype=float)
    except Exception:
        ma_vals = np.asarray(ma).astype(float)

    n = len(src_vals)
    trades = []
    dollars_gained = 0.0
    dollars_lost = 0.0
    points = 0
    cash = float(initial_cash)
    # iterate with explicit index and allow only one open position at a time
    i = 1
    while i < n:
        ma_i = ma_vals[i]
        ma_im1 = ma_vals[i-1]
        src_i = src_vals[i]
        src_im1 = src_vals[i-1]
        # skip invalid values
        if np.isnan(ma_i) or np.isnan(ma_im1) or np.isnan(src_i) or np.isnan(src_im1):
            i += 1
            continue
        # detect cross above
        if (src_im1 <= ma_im1) and (src_i > ma_i):
            signal_idx = i
            # entry at next bar (i+1) if exists
            if signal_idx + 1 >= n:
                # no next bar to enter
                break
            # entry at next bar
            # extract Python scalars safely using .item() when present
            raw_entry = src_vals[signal_idx + 1]
            entry = raw_entry.item() if hasattr(raw_entry, 'item') else float(raw_entry)
            raw_ma = ma_vals[signal_idx]
            ma_at_signal = raw_ma.item() if hasattr(raw_ma, 'item') else float(raw_ma)
            # SL is MA minus sl_offset_dollars
            sl = ma_at_signal - sl_offset_dollars
            if np.isnan(entry) or np.isnan(sl) or np.isnan(ma_at_signal):
                i = signal_idx + 1
                continue
            # absolute TP defined in dollars above entry
            tp = entry + tp_dollars
            # simulate forward to find TP or SL; only one position open
            exit_price = None
            exit_reason = None
            exit_idx = None
            for j in range(signal_idx + 1, min(n, signal_idx + 1 + max_holding)):
                price = src_vals[j]
                if np.isnan(price):
                    continue
                # cast price to float in comparisons
                raw_p = price
                pval = raw_p.item() if hasattr(raw_p, 'item') else float(raw_p)
                if pval <= sl:
                    exit_price = sl if isinstance(sl, float) else float(sl)
                    exit_reason = 'SL'
                    exit_idx = j
                    break
                if pval >= tp:
                    exit_price = tp if isinstance(tp, float) else float(tp)
                    exit_reason = 'TP'
                    exit_idx = j
                    break
            if exit_reason is None:
                exit_idx = min(n - 1, signal_idx + max_holding)
                raw_exit = src_vals[exit_idx]
                exit_price = raw_exit.item() if hasattr(raw_exit, 'item') else float(raw_exit)
                exit_reason = 'TIMEOUT'
            pct = float((exit_price - entry) / entry)
            trades.append(pct)
            # dollar outcome using full cash allocation at entry
            units = float(cash) / float(entry) if entry > 0 else 0.0
            dollar_diff = float(units * (exit_price - entry))
            if dollar_diff > 0:
                dollars_gained += dollar_diff
            else:
                dollars_lost += -dollar_diff
            cash += dollar_diff
            if exit_reason == 'TP':
                points += 2
            elif exit_reason == 'SL':
                points -= 1
            # move index to after the exit to wait for next crossover
            i = exit_idx + 1 if exit_idx is not None else signal_idx + 2
        else:
            i += 1
    # coerce to plain python scalars to avoid ndarray-to-scalar deprecation
    if len(trades) == 0:
        total_percent = 0.0
        num_trades = 0
    else:
        total_percent = float(sum(float(x) for x in trades))
        num_trades = int(len(trades))
    cash = float(cash)
    dollars_gained = float(dollars_gained)
    dollars_lost = float(dollars_lost)
    points = int(points)
    return total_percent, num_trades, cash, dollars_gained, dollars_lost, points


def scan_all_mas_single(df, max_len=30, tp_sl_ratio=1.5, max_holding=60, sl_offset_dollars=1.0, tp_dollars=1.5, initial_cash=10000.0):
    """Scan each MA type and lengths 1..max_len on df['Close'] and return a summary.
    Scores are total percent profit across trades for long-only cross-above signals.
    """
    src = df['Close']
    vol = df['Volume']
    types = ['EMA', 'SMA', 'WMA', 'Hull', 'VWMA', 'ALMA', 'RMA', 'LINREG']
    results = {t: [] for t in types}
    # overall_best = (type, len, points, final_cash, total_pct, dollars_gained, dollars_lost, num_trades)
    overall_best = (None, None, -10**9, float(initial_cash), -np.inf, 0.0, 0.0, 0)
    for t in types:
        for length in range(1, max_len + 1):
            volume = vol if t == 'VWMA' else None
            ma = get_ma(t, src, length, volume)
            total_pct, num_trades, final_cash, dollars_gained, dollars_lost, points = simulate_single_ma_trades(
                src, ma, max_holding=max_holding, tp_sl_ratio=tp_sl_ratio,
                sl_offset_dollars=sl_offset_dollars, tp_dollars=tp_dollars, initial_cash=initial_cash)
            # coerce to plain scalars
            total_pct_s = float(np.nan_to_num(scalarize(total_pct, 0.0)))
            num_trades_s = int(np.nan_to_num(scalarize(num_trades, 0)))
            dollars_gained_s = float(np.nan_to_num(scalarize(dollars_gained, 0.0)))
            dollars_lost_s = float(np.nan_to_num(scalarize(dollars_lost, 0.0)))
            points_s = int(np.nan_to_num(scalarize(points, 0)))
            final_cash_s = float(np.nan_to_num(scalarize(final_cash, initial_cash)))
            results[t].append((length, total_pct_s, num_trades_s, dollars_gained_s, dollars_lost_s, points_s, final_cash_s))
            # rank by points primarily, then total_pct
            if num_trades_s > 0:
                if (points_s > overall_best[2]) or (points_s == overall_best[2] and total_pct_s > overall_best[4]):
                    overall_best = (t, length, points_s, final_cash_s, total_pct_s, dollars_gained_s, dollars_lost_s, num_trades_s)
    return results, overall_best

def find_best_ma_improved(df, train_split=0.8, max_length=100, eval_period=20, alma_offset=0.85, alma_sigma=6.0):
    split_idx = int(len(df) * train_split)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    src_train = train_df['Close']
    vol_train = train_df['Volume']
    src_test = test_df['Close']
    vol_test = test_df['Volume']
    
    types = ['EMA', 'SMA', 'WMA', 'Hull', 'VWMA', 'ALMA', 'RMA', 'LINREG']
    best_score = -np.inf
    best_type, best_len = None, None
    scores_dict = {t: [] for t in types}
    
    for t in types:
        for length in range(1, max_length + 1):
            volume_train = vol_train if t == 'VWMA' else None
            ma_train = get_ma(t, src_train, length, volume_train, alma_offset, alma_sigma)
            score = score_ma(src_train, ma_train, length, eval_period)
            scores_dict[t].append(score)
            if score > best_score:
                best_score = score
                best_type, best_len = t, length
    
    # Backtest on test set
    volume_test_bt = vol_test if best_type == 'VWMA' else None
    best_ma_test = get_ma(best_type, src_test, best_len, volume_test_bt, alma_offset, alma_sigma)
    test_profit, test_sharpe, test_signals = calculate_signals(src_test, best_ma_test, min(len(test_df), eval_period))
    
    # Plot scores vs lengths
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    for t in types:
        ax[0].plot(range(1, max_length + 1), scores_dict[t], label=t)
    ax[0].set_title('Train Scores by MA Type and Length')
    ax[0].legend()
    ax[0].set_xlabel('Length')
    ax[0].set_ylabel('Score')
    
    lengths = list(range(1, max_length + 1))
    test_scores = []
    for l in lengths:
        if l > len(src_test):
            break
        volume_l = vol_test if best_type == 'VWMA' else None
        ma_l = get_ma(best_type, src_test, l, volume_l, alma_offset, alma_sigma)
        score_l = score_ma(src_test, ma_l, l, min(len(test_df), eval_period))
        test_scores.append(score_l)
    ax[1].plot(lengths[:len(test_scores)], test_scores, label=f'{best_type} Test')
    ax[1].set_title('Test Scores for Best Type')
    ax[1].legend()
    ax[1].set_xlabel('Length')
    ax[1].set_ylabel('Score')
    plt.tight_layout()
    plt.savefig('ma_scores.png')
    plt.close()
    
    return best_type, best_len, best_score, test_profit, test_sharpe, test_signals

if __name__ == '__main__':
    # support a non-interactive --auto flag for headless runs with sane defaults
    auto = '--auto' in sys.argv
    if auto:
        # defaults
        settings = {
            'ticker': 'BTC-USD',
            'period': '7d',
            'interval': '1m',
            'max_len': 30,
            'tp_dollars': 1.5,
            'sl_offset': 1.0,
            'initial_cash': 10000.0,
            'max_holding': 60,
            'ranking': 'points'
        }
    else:
        settings = get_settings()
    settings = clamp_settings(settings)
    ticker = settings['ticker']
    # use the new downloader which chunks 1m requests when needed
    try:
        data = download_with_granularity(ticker, period=settings['period'], interval=settings['interval'])
    except Exception as ex:
        print('Failed to download data:', ex)
        raise
    best_type, best_len, train_score, test_profit, test_sharpe, test_signals = find_best_ma_improved(data, max_length=settings['max_len'])
    print(f"Best MA (Train): {best_type}({best_len}), Score: {train_score:.4f}")
    print(f"Test: Profit={test_profit:.4f}, Sharpe={test_sharpe:.4f}, Signals={test_signals}")

    # run single-MA scan
    results, overall_best = scan_all_mas_single(
        data,
        max_len=settings['max_len'],
        tp_sl_ratio=None,  # tp handled in dollars by tp_dollars
        max_holding=settings['max_holding'],
        sl_offset_dollars=settings['sl_offset'],
        tp_dollars=settings['tp_dollars'],
        initial_cash=settings['initial_cash'])
    print(f"\nSingle-MA TP/SL scan results (length 1..{settings['max_len']}):")
    for t, rows in results.items():
        def keyfn(r):
            # prefer points then number of trades
            return (scalarize(r[5], 0.0), float(scalarize(r[2], 0.0)))
        best_row = max(rows, key=keyfn)
        length = int(best_row[0])
        total_pct = float(scalarize(best_row[1], 0.0))
        num_trades = int(scalarize(best_row[2], 0))
        dollars_gained_b = float(scalarize(best_row[3], 0.0))
        dollars_lost_b = float(scalarize(best_row[4], 0.0))
        points_b = int(scalarize(best_row[5], 0))
        final_cash_b = float(scalarize(best_row[6], settings['initial_cash']))
        print(f"{t}: best length={length}, total_pct={total_pct:.4f}, trades={num_trades}, final_cash=${final_cash_b:.2f}, gained=${dollars_gained_b:.2f}, lost=${dollars_lost_b:.2f}, points={points_b}")

    # if user prefers final_cash ranking, recompute overall_best by final_cash
    if settings.get('ranking', 'points') == 'final_cash':
        best_fc = None
        for t, rows in results.items():
            for r in rows:
                if best_fc is None or r[6] > best_fc[6]:
                    best_fc = (t, ) + r
        if best_fc is not None:
            overall_best = (best_fc[0], best_fc[1], int(best_fc[5]), float(best_fc[6]), float(best_fc[2]), float(best_fc[3]), float(best_fc[4]), int(best_fc[2]))

    if overall_best[0] is not None:
        ob_type = overall_best[0]
        ob_len = int(overall_best[1])
        ob_points = int(overall_best[2])
        ob_final_cash = float(overall_best[3])
        ob_pct = float(overall_best[4])
        ob_gained = float(overall_best[5])
        ob_lost = float(overall_best[6])
        ob_tr = int(overall_best[7])
        print(f"\nOverall best single MA (by chosen ranking): {ob_type} length={ob_len}, points={ob_points}, final_cash=${ob_final_cash:.2f}, total_pct={ob_pct:.4f}, gained=${ob_gained:.2f}, lost=${ob_lost:.2f}, trades={ob_tr}")
    else:
        print('\nNo profitable single-MA trades found in scan.')

    # Plotting: lengths vs total percent for each MA type
    plt.figure(figsize=(12, 6))
    for t, rows in results.items():
        lengths = [int(r[0]) for r in rows]
        total_pcts = [float(scalarize(r[1], 0.0)) for r in rows]
        plt.plot(lengths, total_pcts, label=t)
    plt.xlabel('Length')
    plt.ylabel('Total percent profit')
    plt.title(f'Single-MA scan results (max_len={settings["max_len"]})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('single_ma_scores.png')
    print('\nSaved single-MA percent chart to single_ma_scores.png')

    # Dollars chart (gained and lost)
    plt.figure(figsize=(12, 6))
    for t, rows in results.items():
        lengths = [int(r[0]) for r in rows]
        gained = [float(scalarize(r[3], 0.0)) for r in rows]
        lost = [float(scalarize(r[4], 0.0)) for r in rows]
        plt.plot(lengths, gained, label=f'{t} gained')
        plt.plot(lengths, lost, linestyle='--', label=f'{t} lost')
    plt.xlabel('Length')
    plt.ylabel('Dollars')
    plt.title('Single-MA dollars gained (solid) and lost (dashed) by length')
    plt.legend(fontsize='small', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('single_ma_dollars.png')
    print('Saved dollars chart to single_ma_dollars.png')

    # Points chart
    plt.figure(figsize=(12, 6))
    for t, rows in results.items():
        lengths = [int(r[0]) for r in rows]
        pts = [int(scalarize(r[5], 0)) for r in rows]
        plt.plot(lengths, pts, label=t)
    plt.xlabel('Length')
    plt.ylabel('Points')
    plt.title('Single-MA points by length')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('single_ma_points.png')
    print('Saved points chart to single_ma_points.png')
