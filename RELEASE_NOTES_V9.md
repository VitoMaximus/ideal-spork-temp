Highlights
- Expert CSV: added fields ZoneK, LadderScale, TrailActive/Stop/Mult/Reason.
- Priority blending: exposed Priority_Mult and Priority_Notes with reasons.
- Gating flags exported: Squeeze, Earnings_Window_Flag, RSI_OK, GapChase_OK, Stoch_Long_OK.
- Boolean normalization for gating flags to ensure clean CSV output.
- Weekly regime block canonicalized; monthly regime helper integrated.
- Housekeeping: ignore CSV outputs and backups; removed unused helper.

Smoke Test
- Run: python3 -m py_compile Trading.py && python3 Trading.py -A --no-xlsx
- Verify Expert CSV contains: ZoneK, LadderScale, TrailActive, TrailStop, TrailMult,
  TrailReason, Priority_Mult, Priority_Notes, Squeeze, Earnings_Window_Flag, RSI_OK,
  GapChase_OK, Stoch_Long_OK.
