# Plan: Unified CLI for All Tools

## What

Standardize every tool's CLI to `--flag` arguments with consistent output:
```
python3 tools/stock.py --ticker TSLA
→ Fetching data for [TSLA]
→ [price table]
→ [matplotlib chart]
```

## Files to Modify (7 tool scripts + 1 dispatcher)

---

### 1. `tools/stock.py` — Stock Price + Chart

**New CLI:**
```
# By ticker (direct, no Yahoo search):
python3 tools/stock.py --ticker TSLA

# By company name (auto-search):
python3 tools/stock.py --company "Tesla"
```

**Terminal output:**
```
→ Fetching data for [TSLA]
→ Tesla, Inc. — $248.50
→ 30d change: +$6.20 (+2.56%)

   Date        │ Close
───────────────┼────────
   Apr 21      │ $242.30
   Apr 22      │ $245.10
   ...         │ ...
   May 21      │ $248.50
```

**Changes:**
- Add `argparse` with `--ticker` and `--company` flags
- Banner line `print(f"→ Fetching data for [{ticker_symbol}]")`
- After `talk2(f"{name} stock price is ${price}")`, print the price to terminal too
- After plotting, print a date/close table using `tabulate` or manual formatting
- Keep `plt.show()` for the chart window

---

### 2. `tools/crypto.py` — Live Crypto Price

**New CLI:**
```bash
python3 tools/crypto.py --coin ethereum
```

**Terminal output:**
```
→ Fetching market price for [Ethereum]
→ [tkinter live price window opens]
```

**Changes:**
- Add `--coin` flag (default: bitcoin)
- Banner `print(f"→ Fetching market price for [{currency.title()}]")`

---

### 3. `tools/news.py` — Open News

**New CLI:**
```bash
python3 tools/news.py --source BBC
```

**Terminal output:**
```
→ Fetching latest news from [BBC News]
→ [browser opens]
```

**Changes:**
- Add `--source` flag with choices: BBC, AlJazeera, NDTV (maps to URLs)
- Banner `print(f"→ Fetching latest news from [{source}]")`

---

### 4. `tools/alarm.py` — Set Alarm

**New CLI:**
```bash
python3 tools/alarm.py --time "05:00"
```

**Terminal output:**
```
→ Setting alarm for [05:00 AM]
→ [alarm loop runs in background]
```

**Changes:**
- Add `--time` flag
- Simplify parsing (accept `HH:MM` or `HH:MM AM/PM`)
- Banner `print(f"→ Setting alarm for [{alarm_time}]")`

---

### 5. `tools/timer.py` — Countdown Timer

**New CLI:**
```bash
python3 tools/timer.py --duration 300
```

**Terminal output:**
```
→ Starting timer for [5 minutes]
→ [countdown runs]
```

**Changes:**
- Add `--duration` flag accepting seconds
- Banner `print(f"→ Starting timer for [{duration_formatted}]")`

---

### 6. `tools/translate.py` — Dictionary Translator

**New CLI:**
```bash
python3 tools/translate.py --text "I love you" --to bn
```

**Terminal output:**
```
→ Translating to [Bengali]
→ [turtle GUI opens showing translation]
```

**Changes:**
- Add `--text` and `--to` flags
- Banner `print(f"→ Translating to [{language_name}]")`
- Auto-run translation (no interactive prompts when args given)

---

### 7. `tools/image.py` — Screenshot + Stencil

**New CLI:**
```bash
python3 tools/image.py --prompt "Starry Night"
```

**Terminal output:**
```
→ Generating image of [Starry Night]
→ [screenshot area selector + stencil window]
```

**Changes:**
- Add `--prompt` flag (cosmetic label for what's being drawn)
- Banner `print(f"→ Generating image of [{prompt}]")`
- Keep same screenshot + stencil logic

---

### 8. `marin_fier.py` — Update Tool Launchers

Each `tool_*` function needs to pass the new `--flag` arguments:

| Function | Current | New |
|----------|---------|-----|
| `tool_get_stock_info(company)` | `[company]` or `["--ticker", co]` | `["--ticker", co]` (always, stock.py handles both) |
| `tool_get_crypto_price(coin)` | `[coin]` | `["--coin", coin]` |
| `tool_open_news()` | `[]` | `["--source", "BBC"]` |
| `tool_set_alarm(time)` | `[time]` | `["--time", time]` |
| `tool_set_timer(duration)` | `[duration]` | `["--duration", duration]` |

Functions that stay unchanged (interactive or game wrappers):
- `tool_send_email()`, `tool_play_tictactoc()`, `tool_play_connect4()`, `tool_play_wordgame()`, `tool_draw_me()`, `tool_take_screenshot()`

---

## Implementation Order

1. `tools/stock.py` — most complex (table + chart)
2. `tools/crypto.py` — simple flag change
3. `tools/news.py` — simple flag change
4. `tools/alarm.py` — flag + simplify parsing
5. `tools/timer.py` — flag + convert to seconds
6. `tools/translate.py` — flag + non-interactive mode
7. `tools/image.py` — cosmetic prompt label
8. `marin_fier.py` — update all launcher args

---

## Questions

1. Do you want `--duration` for the timer in **seconds** (300 = 5 min) or a natural string like `"5 minutes"`?
2. `translate.py` currently opens a Turtle GUI — should `--text` / `--to` still open the GUI, or just print the result to terminal?
3. `alarm.py` — when run from CLI, should it block (keep running until alarm fires) or just print and exit?
