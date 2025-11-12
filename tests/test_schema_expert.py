
import csv, os, pytest

CSV_PATH = os.path.expanduser('~/Desktop/stock_screener/Ticker_Tech_Indicators_V9_Expert.csv')
EXPECTED = {
  "Ticker","Description","Priority","Priority_Notes","EMA_Regime","Weekly_Regime",
  "Signal_Age","Fresh_Cross","Open","High","Low","Close","SMA21","EMA_Short",
  "EMA_Long","EMA21","EMA21w","RSI(14)","ADX(14)","MACD_Hist_Pos","MACD_Hist","ATR",
  "ATR%","BreakoutStrength_ATR","VolRatio_20d","UDVR(20d)","VolZ_20","Breakout20",
  "Breakdown20","High_52W","Low_52W","%From_21SMA","HitToday_Up1ATR %","HitWeek_Up1ATR %",
  "Entry1","Entry2","Entry3","Exit1","Exit2","Stop","Dist_to_Entry1_pct",
  "Dist_to_Stop_pct","RR_Entry2_to_Exit1","Breakout_Base","Res1_Pivot","Res1_Zone_Lo",
  "Res1_Zone_Hi","BO_T1","BO_T2","At_Res1_Zone","D_Layer","D_Zone_Lo","D_Zone_Hi",
  "W_Layer","W_Zone_Lo","W_Zone_Hi","D_AVWAP_H","D_AVWAP_L","W_AVWAP_H","W_AVWAP_L",
  "PE","EPS_Growth_Pct","ZoneK","LadderScale","Priority_Mult","Squeeze",
  "Earnings_Window_Flag","RSI_OK","GapChase_OK","Stoch_Long_OK","TrailActive",
  "TrailStop","TrailMult","TrailReason","DataMode","AsOf","Session","DelayMin",
  "AdjustedHistory","Live_Delta%","At_Res1_Zone_LIVE","T1_Hit","T2_Hit","Stop_Hit",
  "E1_Hit","E2_Hit","E3_Hit","PremarketGapPct","EarningsInDays","Ladder_Sanity",
  "AvgVol20","Rank"
}

@pytest.mark.skipif(not os.path.exists(CSV_PATH), reason="Expert CSV not present")
def test_expert_schema_columns_present():
    with open(CSV_PATH, newline="") as fh:
        hdr = set(next(csv.reader(fh)))
    missing = sorted(EXPECTED - hdr)
    assert not missing, f"Missing: {missing}"
