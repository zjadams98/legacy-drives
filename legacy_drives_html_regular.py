import nfl_data_py as nfl
import pandas as pd
from collections import Counter
from datetime import datetime
import json
import os

import warnings
from pandas.errors import PerformanceWarning

warnings.simplefilter("ignore", PerformanceWarning)
warnings.simplefilter("ignore", category=pd.errors.SettingWithCopyWarning)

CURRENT_SEASON = 2025
CACHE_FILE = "legacy_drives_regularseason_cache_website.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            data = json.load(f)
            return (data.get('opportunities', []), 
                    set(data.get('processed_games', [])), 
                    data.get('last_season_processed', 2000),
                    data.get('debug_rows', []))
    return [], set(), 2000, []

def save_cache(opportunities, processed_games, last_season, debug_rows):
    with open(CACHE_FILE, 'w') as f:
        json.dump({
            'opportunities': opportunities,
            'processed_games': list(processed_games),
            'last_season_processed': last_season,
            'debug_rows': debug_rows,
            'last_updated': datetime.now().isoformat()
        }, f)

cached_opportunities, processed_games, last_season_processed, cached_debug_rows = load_cache()
print(f"Loaded {len(cached_opportunities)} cached opportunities from {len(processed_games)} games")
print(f"Last season fully processed: {last_season_processed}")

if last_season_processed < CURRENT_SEASON - 1:
    seasons_to_load = list(range(last_season_processed, CURRENT_SEASON + 1))
    print(f"Loading seasons {seasons_to_load[0]}-{seasons_to_load[-1]} to catch up...")
else:
    seasons_to_load = [CURRENT_SEASON]
    print(f"Only loading current season: {CURRENT_SEASON}")

pbp = nfl.import_pbp_data(seasons_to_load, downcast=True, cache=False)
pbp = pbp[pbp['season_type'] == 'REG']

all_games = set(pbp['game_id'].unique())
new_games = all_games - processed_games
print(f"Found {len(new_games)} new games to process")

if new_games:
    pbp = pbp[pbp['game_id'].isin(new_games)]
else:
    print("No new games to process. Using cached data only.")
    pbp = pd.DataFrame()

name_map = pbp.groupby('passer_id')['passer'].first().to_dict() if not pbp.empty else {}
qb_name_map = {}
if not pbp.empty and 'qb_id' in pbp.columns and 'qb' in pbp.columns:
    qb_name_map = pbp.groupby('qb_id')['qb'].first().to_dict()

NAME_MAP_FILE = "qb_name_map_regularseason_website.json"
if os.path.exists(NAME_MAP_FILE):
    with open(NAME_MAP_FILE, 'r') as f:
        persistent_name_map = json.load(f)
        name_map = {**persistent_name_map, **name_map}

opportunities = cached_opportunities.copy()
debug_rows = cached_debug_rows.copy()

if not pbp.empty:
    pbp_q4 = pbp[pbp['qtr'] == 4].copy()
    pbp_ot = pbp[pbp['qtr'] >= 5].copy()

    def build_drive_starts(pbp_period: pd.DataFrame) -> pd.DataFrame:
        return (
            pbp_period.sort_values(['game_id', 'drive', 'play_id'])
            .groupby(['game_id', 'drive'])
            .first()
            .reset_index()
        )

    drive_starts_q4 = build_drive_starts(pbp_q4)
    drive_starts_q4['score_diff'] = drive_starts_q4['posteam_score'] - drive_starts_q4['defteam_score']
    q4_opps = drive_starts_q4[
        (drive_starts_q4['quarter_seconds_remaining'] <= 180) &
        (drive_starts_q4['quarter_seconds_remaining'] >= 0) &
        (drive_starts_q4['score_diff'] <= -1) &
        (drive_starts_q4['score_diff'] >= -8)
    ].copy()
    q4_opps['period'] = 'Q4'

    drive_starts_ot = build_drive_starts(pbp_ot)
    drive_starts_ot['score_diff'] = drive_starts_ot['posteam_score'] - drive_starts_ot['defteam_score']
    ot_opps = drive_starts_ot.copy()
    ot_opps['period'] = 'OT'

    ot_opps = ot_opps.sort_values(['game_id', 'qtr', 'play_id'])
    ot_opps['ot_drive_rank'] = ot_opps.groupby('game_id').cumcount() + 1

    opps = pd.concat([q4_opps, ot_opps], ignore_index=True)

    for _, row in opps.iterrows():
        game_id = row['game_id']
        drive_num = row['drive']

        period = row.get('period', 'Q4')
        pbp_period = pbp_q4 if period == 'Q4' else pbp_ot
        drive_all = pbp_period[(pbp_period['game_id'] == game_id) & (pbp_period['drive'] == drive_num)].copy()
        if drive_all.empty:
            continue

        qb_id = None
        qb_name = None

        if 'qb_id' in drive_all.columns:
            qb_series = drive_all['qb_id'].dropna()
            if not qb_series.empty:
                qb_id = qb_series.mode().iloc[0]
                qb_name = qb_name_map.get(qb_id)

        if qb_id is None:
            drive_pass = drive_all[drive_all.get('pass_attempt', 0) == 1] if 'pass_attempt' in drive_all.columns else pd.DataFrame()
            if not drive_pass.empty and 'passer_id' in drive_pass.columns:
                qb_counts = Counter(drive_pass['passer_id'].dropna())
                if qb_counts:
                    qb_id = qb_counts.most_common(1)[0][0]
                    qb_name = name_map.get(qb_id)

        if qb_id is None:
            qb_id = f"TEAM_{row.get('posteam', 'UNK')}"
            qb_name = qb_id

        if 'game_seconds_remaining' in drive_all.columns:
            drive_all = drive_all.sort_values(['game_seconds_remaining', 'play_id'], ascending=[True, False])
        else:
            drive_all = drive_all.sort_values(['quarter_seconds_remaining', 'play_id'], ascending=[True, False])

        for c in ['posteam_score_post', 'defteam_score_post']:
            if c in drive_all.columns:
                drive_all[c] = drive_all[c].ffill().bfill()

        last_play = drive_all.iloc[0]

        end_team_score = last_play.get('posteam_score_post', pd.NA)
        end_opp_score = last_play.get('defteam_score_post', pd.NA)

        if pd.isna(end_team_score) or pd.isna(end_opp_score):
            candidates = drive_all.dropna(subset=['posteam_score_post', 'defteam_score_post'])
            if candidates.empty:
                continue
            last_play = candidates.iloc[0]
            end_team_score = last_play['posteam_score_post']
            end_opp_score = last_play['defteam_score_post']

        if period == 'Q4':
            if end_team_score >= end_opp_score:
                result = 'W'
                reason = 'Q4: drive ended tied or leading (Success)'
            else:
                result = 'L'
                reason = 'Q4: drive ended still trailing (Failure)'
        else:
            ot_rank = int(row.get('ot_drive_rank', 1))
            start_posteam = row.get('posteam')

            season_val = row.get('season')
            try:
                season_val = int(season_val) if pd.notna(season_val) else None
            except Exception:
                season_val = None

            td_scored = False
            if 'touchdown' in drive_all.columns:
                if 'td_team' in drive_all.columns and start_posteam is not None:
                    td_scored = ((drive_all['touchdown'] == 1) & (drive_all['td_team'] == start_posteam)).any()
                elif 'posteam' in drive_all.columns and start_posteam is not None:
                    td_scored = ((drive_all['touchdown'] == 1) & (drive_all['posteam'] == start_posteam)).any()
                else:
                    td_scored = (drive_all['touchdown'] == 1).any()

            fg_scored = False
            if 'field_goal_result' in drive_all.columns:
                if 'posteam' in drive_all.columns and start_posteam is not None:
                    fg_scored = ((drive_all['field_goal_result'] == 'made') & (drive_all['posteam'] == start_posteam)).any()
                else:
                    fg_scored = (drive_all['field_goal_result'] == 'made').any()

            if season_val is not None and season_val < 2012:
                if td_scored or fg_scored:
                    result = 'W'
                    reason = 'OT (pre-2012): FG/TD scored on drive (Success)'
                else:
                    result = 'L'
                    reason = 'OT (pre-2012): no FG/TD scored on drive (Failure)'
            else:
                if ot_rank == 1:
                    if td_scored:
                        result = 'W'
                        reason = 'OT (1st drive): TD scored (Success)'
                    else:
                        result = 'L'
                        reason = 'OT (1st drive): no TD (FG or no score) (Failure)'
                else:
                    if end_team_score > end_opp_score:
                        result = 'W'
                        reason = f'OT (drive {ot_rank}): ended leading (Success)'
                    else:
                        result = 'L'
                        reason = f'OT (drive {ot_rank}): ended not leading (Failure)'

        if period == 'Q4':
            start_qsr = row.get('quarter_seconds_remaining')
            if pd.notna(start_qsr) and start_qsr <= 30 and result == 'L':
                continue

        opportunities.append({'qb_id': qb_id, 'result': result})

        final_row = last_play
        idx = 1
        while True:
            desc_txt = str(final_row.get('desc', '') or '')
            ptype = final_row.get('play_type')
            if (ptype not in ['extra_point', 'two_point_attempt']) and ('END GAME' not in desc_txt) and (not desc_txt.strip().startswith('Timeout')) and (not desc_txt.strip().startswith('TWO-POINT ATTEMPT')):
                break
            if idx >= len(drive_all):
                break
            final_row = drive_all.iloc[idx]
            idx += 1

        final_desc = final_row.get('desc')

        final_down = final_row.get('down')
        final_yds = final_row.get('ydstogo')

        final_down = f"{int(final_down)}down" if pd.notna(final_down) and int(final_down) > 0 else None
        final_yds = f"{int(final_yds)}yrdstogo" if pd.notna(final_yds) and int(final_yds) > 0 else None

        debug_rows.append({
            'qb_name': qb_name,
            'game_id': game_id,
            'period': period,
            'start_score_diff': f"down {abs(int(row['score_diff']))}",
            'start_time': row.get('time'),
            'end_time': last_play.get('time'),
            'final_down': final_down,
            'final_ydstogo': final_yds,
            'end_team_score': int(end_team_score),
            'end_opp_score': int(end_opp_score),
            'final_play': final_desc,
            'result': result,
            'reason': reason
        })

    processed_games.update(new_games)
    
    if seasons_to_load[-1] < CURRENT_SEASON:
        last_season_processed = seasons_to_load[-1]
    elif len(seasons_to_load) > 1:
        last_season_processed = CURRENT_SEASON - 1
    
    save_cache(opportunities, processed_games, last_season_processed, debug_rows)
    
    with open(NAME_MAP_FILE, 'w') as f:
        json.dump(name_map, f)
    
    print(f"Processed {len(new_games)} new games. Total opportunities: {len(opportunities)}")

# Generate HTML output
if opportunities:
    df = pd.DataFrame(opportunities)
    records = (
        df.pivot_table(index='qb_id', columns='result', aggfunc='size', fill_value=0)
        .rename(columns={'W': 'wins', 'L': 'losses'})
    )
    records = records[~records.index.astype(str).str.startswith("TEAM_")]
    for c in ['wins', 'losses']:
        if c not in records.columns:
            records[c] = 0

    records['qb_name'] = (
        records.index.to_series().map(qb_name_map)
        .fillna(records.index.to_series().map(name_map))
        .fillna(records.index.to_series().astype(str))
    )
    records['decisions'] = records['wins'] + records['losses']
    records['win_pct'] = (records['wins'] / records['decisions'].where(records['decisions'] > 0) * 100).fillna(0).round(1)
    records['total_opps'] = records['wins'] + records['losses']
    records = records[records['total_opps'] > 0]
    records = records.drop(columns=['total_opps'])

    records = records[['qb_name', 'wins', 'losses', 'win_pct']].sort_values(['wins','losses','win_pct'], ascending=[False, True, False])

    # Generate HTML
    generated_ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Legacy Drives Leaderboard - Regular Season</title>
    <style>
        body {{
            font-family: 'Courier New', monospace;
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
            line-height: 1.6;
        }}
        h1 {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .subtitle {{
            font-size: 12px;
            margin-bottom: 15px;
        }}
        .timestamp {{
            font-size: 10px;
            margin-bottom: 20px;
        }}
        .criteria {{
            font-size: 11px;
            margin-bottom: 20px;
            line-height: 1.5;
        }}
        .criteria-title {{
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .section-header {{
            font-weight: bold;
            margin-top: 10px;
        }}
        .leaderboard {{
            font-size: 12px;
            columns: 2;
            column-gap: 40px;
        }}
        .qb-entry {{
            break-inside: avoid;
            margin-bottom: 3px;
        }}
    </style>
</head>
<body>
    <h1>Legacy Drives Leaderboard (Regular Season)</h1>
    <div class="subtitle">Seasons: 2000-{CURRENT_SEASON} | Sorted by Legacy Drive Successes</div>
    <div class="timestamp">Generated: {generated_ts}</div>
    
    <div class="criteria">
        <div class="criteria-title">Legacy Drive Opportunity Criteria:</div>
        Q4: Drive starts at 3:00 or less but > 0:30 left (Unless a Success), down 1–8 points. OT: All Drives
        
        <div class="section-header">Q4 Result:</div>
        Success = Lead or Tied at Drive End<br>
        Fail = Trailing at Drive End
        
        <div class="section-header">OT Result:</div>
        Success = TD on 1st Drive, Lead at Drive End on All Other Drives (pre-2012 FG on 1st Drive is Success)<br>
        Fail = FG or No Score on 1st OT Drive, Tied or Trailing on All Other Drives
        
        <div class="section-header" style="margin-top: 15px;">Name: LD Successes - LD Failures (Success %)</div>
    </div>
    
    <div class="leaderboard">
"""
    
    # Add all QB entries
    for _, row in records.iterrows():
        html_content += f'        <div class="qb-entry">{row["qb_name"]}: {int(row["wins"])} - {int(row["losses"])} ({row["win_pct"]}%)</div>\n'
    
    html_content += """    </div>
</body>
</html>"""
    
    # Save HTML file
    with open("regular.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"\nGenerated regular.html")
    print(f"Total QBs: {len(records)}")

else:
    print("No legacy drive opportunities found.")