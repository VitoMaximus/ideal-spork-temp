# ideal-spork-temp
Trading.py 

# Ideal Spork — CSV Outputs & Priority Blending (V9)
# (paste the section above here if you want to manage it manually)

## What’s new in V9 — Signals + Guards

**Highlights**
- Expert CSV now includes: `ZoneK`, `LadderScale`, `TrailActive`, `TrailStop`, `TrailMult`, `TrailReason`.
- Priority blending exports: `Priority_Mult` and `Priority_Notes` (with human-readable reasons).
- Gating flags exported: `Squeeze`, `Earnings_Window_Flag`, `RSI_OK`, `GapChase_OK`, `Stoch_Long_OK`.
- Boolean normalization applied to gating flags for clean CSV output.
- Weekly regime block canonicalized; monthly regime helper integrated.
- Housekeeping: CSV outputs & backups ignored; unused helper removed.

**New columns (Expert.csv)**
| Column              | Meaning                                                                 |
|---                  |---                                                                       |
| ZoneK               | Zone scaling factor derived from ATR% (resistance/support sizing).      |
| LadderScale         | Per-asset ladder scaling (volatility-aware).                            |
| TrailActive         | ATR trailing stop currently engaged (bool).                             |
| TrailStop           | Current ATR trailing stop price.                                        |
| TrailMult           | ATR multiple used for trailing stop.                                    |
| TrailReason         | Why trailing stop engaged/updated.                                      |
| Priority_Mult       | Final blend multiplier applied to priority score.                       |
| Priority_Notes      | Text reasons composing the blend (e.g., “squeeze +10%; stoch +5%”).     |
| Squeeze             | Bollinger “squeeze” condition (bool).                                   |
| Earnings_Window_Flag| In earnings window (bool).                                              |
| RSI_OK              | RSI gate passed (bool).                                                 |
| GapChase_OK         | Gap + volume sanity gate passed (bool).                                 |
| Stoch_Long_OK       | Stoch(14,3,3) long bias gate passed (bool).                             |

**Smoke test**
```bash
python3 -m py_compile Trading.py && python3 Trading.py -A --no-xlsx
DEST="$HOME/Desktop/stock_screener"; mkdir -p "$DEST"
find . -maxdepth 1 -type f -name 'Ticker_Tech_Indicators_V9_*.csv' -exec mv -v {} "$DEST"/ \;
python3 - <<'PY'
import csv, os
p=os.path.expanduser('~/Desktop/stock_screener/Ticker_Tech_Indicators_V9_Expert.csv')
need={"ZoneK","LadderScale","TrailActive","TrailStop","TrailMult","TrailReason","Priority_Mult","Priority_Notes","Squeeze","Earnings_Window_Flag","RSI_OK","GapChase_OK","Stoch_Long_OK"}
with open(p, newline='') as fh:
    hdr=set(next(csv.reader(fh)))
print("Missing:", sorted(need-hdr))
PY







## What’s new in V9 — Signals + Guards

**Highlights**
- Expert CSV now includes: `ZoneK`, `LadderScale`, `TrailActive`, `TrailStop`, `TrailMult`, `TrailReason`.
- Priority blending exports: `Priority_Mult` and `Priority_Notes` (with human-readable reasons).
- Gating flags exported: `Squeeze`, `Earnings_Window_Flag`, `RSI_OK`, `GapChase_OK`, `Stoch_Long_OK`.
- Boolean normalization applied to gating flags for clean CSV output.
- Weekly regime block canonicalized; monthly regime helper integrated.
- Housekeeping: CSV outputs & backups ignored; unused helper removed.

**New columns (Expert.csv)**
| Column | Meaning |
|---|---|
| ZoneK | Zone scaling factor derived from ATR% (resistance/support sizing). |
| LadderScale | Per-asset ladder scaling (volatility-aware). |
| TrailActive | ATR trailing stop currently engaged (bool). |
| TrailStop | Current ATR trailing stop price. |
| TrailMult | ATR multiple used for trailing stop. |
| TrailReason | Why trailing stop engaged/updated. |
| Priority_Mult | Final blend multiplier applied to priority score. |
| Priority_Notes | Text reasons composing the blend (e.g., “squeeze +10%; stoch +5%”). |
| Squeeze | Bollinger “squeeze” condition (bool). |
| Earnings_Window_Flag | In earnings window (bool). |
| RSI_OK | RSI gate passed (bool). |
| GapChase_OK | Gap + volume sanity gate passed (bool). |
| Stoch_Long_OK | Stoch(14,3,3) long bias gate passed (bool). |

**Smoke test**
```bash
python3 -m py_compile Trading.py && python3 Trading.py -A --no-xlsx
DEST="$HOME/Desktop/stock_screener"; mkdir -p "$DEST"
find . -maxdepth 1 -type f -name 'Ticker_Tech_Indicators_V9_*.csv' -exec mv -v {} "$DEST"/ \;
python3 - <<'PY'
import csv, os
p=os.path.expanduser('~/Desktop/stock_screener/Ticker_Tech_Indicators_V9_Expert.csv')
need={"ZoneK","LadderScale","TrailActive","TrailStop","TrailMult","TrailReason","Priority_Mult","Priority_Notes","Squeeze","Earnings_Window_Flag","RSI_OK","GapChase_OK","Stoch_Long_OK"}
with open(p, newline='') as fh:
    hdr=set(next(csv.reader(fh)))
print("Missing:", sorted(need-hdr))
PY

## What’s new in V9 — Signals + Guards

**Highlights**
- Expert CSV now includes: `ZoneK`, `LadderScale`, `TrailActive`, `TrailStop`, `TrailMult`, `TrailReason`.
- Priority blending exports: `Priority_Mult` and `Priority_Notes` (with human-readable reasons).
- Gating flags exported: `Squeeze`, `Earnings_Window_Flag`, `RSI_OK`, `GapChase_OK`, `Stoch_Long_OK`.
- Boolean normalization applied to gating flags for clean CSV output.
- Weekly regime block canonicalized; monthly regime helper integrated.
- Housekeeping: CSV outputs & backups ignored; unused helper removed.

**New columns (Expert.csv)**
| Column              | Meaning                                                                 |
|---                  |---                                                                       |
| ZoneK               | Zone scaling factor derived from ATR% (resistance/support sizing).      |
| LadderScale         | Per-asset ladder scaling (volatility-aware).                            |
| TrailActive         | ATR trailing stop currently engaged (bool).                             |
| TrailStop           | Current ATR trailing stop price.                                        |
| TrailMult           | ATR multiple used for trailing stop.                                    |
| TrailReason         | Why trailing stop engaged/updated.                                      |
| Priority_Mult       | Final blend multiplier applied to priority score.                       |
| Priority_Notes      | Text reasons composing the blend (e.g., “squeeze +10%; stoch +5%”).     |
| Squeeze             | Bollinger “squeeze” condition (bool).                                   |
| Earnings_Window_Flag| In earnings window (bool).                                              |
| RSI_OK              | RSI gate passed (bool).                                                 |
| GapChase_OK         | Gap + volume sanity gate passed (bool).                                 |
| Stoch_Long_OK       | Stoch(14,3,3) long bias gate passed (bool).                             |

**Smoke test**
\`\`\`bash
python3 -m py_compile Trading.py && python3 Trading.py -A --no-xlsx
DEST="$HOME/Desktop/stock_screener"; mkdir -p "$DEST"
find . -maxdepth 1 -type f -name 'Ticker_Tech_Indicators_V9_*.csv' -exec mv -v {} "$DEST"/ \;
python3 - <<'PY'
import csv, os
p=os.path.expanduser('~/Desktop/stock_screener/Ticker_Tech_Indicators_V9_Expert.csv')
need={"ZoneK","LadderScale","TrailActive","TrailStop","TrailMult","TrailReason","Priority_Mult","Priority_Notes","Squeeze","Earnings_Window_Flag","RSI_OK","GapChase_OK","Stoch_Long_OK"}
with open(p, newline='') as fh:
    hdr=set(next(csv.reader(fh)))
print("Missing:", sorted(need-hdr))
PY
\`\`\`

