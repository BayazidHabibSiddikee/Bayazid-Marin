# Fix: Stock tool not triggering for bare tickers (e.g. "TSLA")

## Root Cause

When user says `"TSLA"` to Marin without explicit keywords like "stock" or "ticker":

1. **`_regex_stage()`** (`marin_fier.py:306`) requires `\b(stock|share\s*price|ticker|market\s*price)\b` — bare "TSLA" doesn't match.
2. Falls through to **qwen2.5:0.5b** (`_llm_stage`) — too small to reliably classify a 4-letter input as a stock request.
3. Intent becomes `"chat"` → no tool launches → the main LLM (**gemma4:31b-cloud**) just roleplays fetching data.

**Secondary issue**: `_extract_company_from()` applies `.title()` which converts `"TSLA"` → `"Tsla"`, `"AAPL"` → `"Aapl"`. Yahoo search is usually robust enough to handle this, but it's still wrong.

---

## Fix 1 — Detect bare tickers in `_regex_stage()` (`marin_fier.py`)

**File**: `marin_fier.py`  
**Location**: After the `_STOCK_PAT` block (~line 334), before the alarm check (~line 336)

Add a new check for standalone 2–5 letter uppercase words that aren't crypto coins or common uppercase abbreviations.

### Add new constant after `_SHOT_PAT` (~line 315):
```python
_TICKER_PAT = re.compile(r'\b[A-Z]{2,5}\b')
_COMMON_UPPER = {"I","OK","AI","ID","TV","PC","DJ","MC","GO","NO","HI",
                 "MY","BY","TO","IN","ON","AT","IS","IT","AM","AN","AS",
                 "BE","DO","HE","IF","ME","OF","OR","SO","UP","US","WE"}
```

### Add helper function after `_extract_duration_from` (~after line 303):
```python
def _extract_ticker_from(text: str) -> str | None:
    words = re.findall(r'\b[A-Z]{2,5}\b', text)
    for w in words:
        if w not in _COMMON_UPPER and w.lower() not in _ALIAS_TO_CANON:
            return w
    return None
```

### Add check in `_regex_stage()` after stock block (~line 334):
```python
    # Check for bare ticker symbols (e.g. "TSLA", "AAPL")
    ticker = _extract_ticker_from(text)
    if ticker:
        return {"intent": "get_stock_info",
                "params": {"company": ticker},
                "confidence": 0.95}
```

---

## Fix 2 — Preserve ticker case in `_extract_company_from()` (`marin_fier.py`)

**File**: `marin_fier.py:274-287`  
**Problem**: `.title()` converts `"TSLA"` → `"Tsla"`

**Change**: Accept original `text` instead of `lower`, map lowered words back to original text to preserve case:

```python
def _extract_company_from(text: str) -> str:
    lower = text.lower()
    cleaned = re.sub(
        r'\b(show me|get me|check|open|what is the|what\'s the|tell me|'
        r'look up|pull up|find|give me|display|fetch|please|'
        r'stock price of|stock of|price of|share price of|'
        r'stock price|share price|stock info|market price|ticker for|'
        r'stock|price|market|shares?|the|a|an)\b',
        ' ', lower, flags=re.IGNORECASE
    )
    words = [w for w in cleaned.split() if len(w) > 1][:3]
    if not words:
        return ""
    # Preserve original case from user input (tickers like TSLA stay TSLA)
    result = []
    pos = 0
    for w in words:
        idx = lower.find(w, pos)
        if idx != -1:
            result.append(text[idx:idx + len(w)])
            pos = idx + len(w)
        else:
            result.append(w.title())
    company = " ".join(result).strip()
    if not company or len(company) > 30:
        return ""
    return company
```

Also update the call site in `_regex_stage()` (line 329):
- Old: `company = _extract_company_from(lower)`
- New: `company = _extract_company_from(text)`

---

## Fix 3 — Use `--ticker` flag for ticker-like inputs (`marin_fier.py`)

**File**: `marin_fier.py:100-106`  
**Problem**: Always passes company name to stock.py without `--ticker`, forcing a Yahoo search even when we already have the ticker.

```python
def tool_get_stock_info(company: str) -> str:
    company = " ".join(company.split()[:3])
    is_ticker = bool(re.match(r'^[A-Z]{1,5}$', company))
    args = ["--ticker", company] if is_ticker else [company]
    err = _popen("tools/stock.py", args)
    if err: return err
    return (f"Stock info window opened for {company}. "
            f"Fetching current market price and 30-day chart via Yahoo Finance.")
```

---

## Fix 4 — Fallback when ticker resolves to empty data (`stock.py`)

**File**: `tools/stock.py:29-56`  
**Problem**: If `--ticker` is given with an invalid symbol, `obj.info` is empty and `price` is None → just says "could not retrieve price".

```python
def show_stock(ticker_symbol: str):
    try:
        obj = yf.Ticker(ticker_symbol)
        info = obj.info or {}
        price = info.get("regularMarketPrice")
        name = info.get("longName", ticker_symbol)
        if price is None and not info:
            # Empty info — maybe it's a company name, not a ticker
            resolved = get_ticker(ticker_symbol)
            if resolved and resolved.upper() != ticker_symbol.upper():
                return show_stock(resolved)
        elif price is None:
            talk1(f"Could not retrieve price for {ticker_symbol}")
            return
        talk2(f"{name} stock price is ${price}")
        # ... rest unchanged
```
