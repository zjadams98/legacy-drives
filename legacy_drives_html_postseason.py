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
CACHE_FILE = "legacy_drives_postseason_cache_website.json"

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
pbp = pbp[pbp['season_type'] == 'POST']

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

NAME_MAP_FILE = "qb_name_map_postseason_website.json"
if os.path.exists(NAME_MAP_FILE):
    with open(NAME_MAP_FILE, 'r') as f:
        persistent_name_map = json.load(f)
        name_map = {**persistent_name_map, **name_map}

opportunities = cached_opportunities.copy()
debug_rows = cached_debug_rows.copy()

if not pbp.empty:
    pbp_q4 = pbp[pbp['qtr'] == 4]
    pbp_ot = pbp[pbp['qtr'] >= 5]

    def build_drive_starts(pbp_period: pd.DataFrame) -> pd.DataFrame:
        return (
            pbp_period.sort_values(['game_id', 'drive', 'play_id'])
            .groupby(['game_id', 'drive'], as_index=False)
            .first()
        )

    drive_starts_q4 = build_drive_starts(pbp_q4)
    drive_starts_q4['score_diff'] = drive_starts_q4['posteam_score'] - drive_starts_q4['defteam_score']
    q4_opps = drive_starts_q4[
        (drive_starts_q4['quarter_seconds_remaining'] <= 180) &
        (drive_starts_q4['quarter_seconds_remaining'] >= 0) &
        (drive_starts_q4['score_diff'].between(-8, -1))
    ]
    q4_opps['period'] = 'Q4'

    drive_starts_ot = build_drive_starts(pbp_ot)
    drive_starts_ot['score_diff'] = drive_starts_ot['posteam_score'] - drive_starts_ot['defteam_score']
    ot_opps = drive_starts_ot.copy()
    ot_opps['period'] = 'OT'
    ot_opps = ot_opps.sort_values(['game_id', 'qtr', 'play_id'])
    ot_opps['ot_drive_rank'] = ot_opps.groupby('game_id').cumcount() + 1

    opps = pd.concat([q4_opps, ot_opps], ignore_index=True)

    pbp_q4_grouped = pbp_q4.groupby(['game_id', 'drive'])
    pbp_ot_grouped = pbp_ot.groupby(['game_id', 'drive'])

    def get_qb_for_drive(drive_all, row):
        qb_id = None
        qb_name = None
        
        if 'qb_id' in drive_all.columns:
            qb_series = drive_all['qb_id'].dropna()
            if not qb_series.empty:
                qb_id = qb_series.mode().iloc[0]
                qb_name = qb_name_map.get(qb_id)
        
        if qb_id is None and 'pass_attempt' in drive_all.columns:
            drive_pass = drive_all[drive_all['pass_attempt'] == 1]
            if not drive_pass.empty and 'passer_id' in drive_pass.columns:
                qb_counts = Counter(drive_pass['passer_id'].dropna())
                if qb_counts:
                    qb_id = qb_counts.most_common(1)[0][0]
                    qb_name = name_map.get(qb_id)
        
        if qb_id is None:
            qb_id = f"TEAM_{row.get('posteam', 'UNK')}"
            qb_name = qb_id
        
        return qb_id, qb_name

    def get_meaningful_final_play(drive_all):
        for idx, row in drive_all.iterrows():
            desc_txt = str(row.get('desc', '') or '')
            ptype = row.get('play_type')
            
            if (ptype not in ['extra_point', 'two_point_attempt'] and 
                'END GAME' not in desc_txt and 
                not desc_txt.strip().startswith('Timeout') and 
                not desc_txt.strip().startswith('TWO-POINT ATTEMPT')):
                return row
        
        return drive_all.iloc[0]

    for _, row in opps.iterrows():
        game_id = row['game_id']
        drive_num = row['drive']
        period = row.get('period', 'Q4')
        
        try:
            if period == 'Q4':
                drive_all = pbp_q4_grouped.get_group((game_id, drive_num)).copy()
            else:
                drive_all = pbp_ot_grouped.get_group((game_id, drive_num)).copy()
        except KeyError:
            continue
        
        qb_id, qb_name = get_qb_for_drive(drive_all, row)
        
        sort_col = 'game_seconds_remaining' if 'game_seconds_remaining' in drive_all.columns else 'quarter_seconds_remaining'
        drive_all = drive_all.sort_values([sort_col, 'play_id'], ascending=[True, False])
        
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
            except (ValueError, TypeError):
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
            
            if season_val is not None and season_val < 2010:
                if td_scored or fg_scored:
                    result = 'W'
                    reason = 'OT (pre-2010): FG/TD scored on drive (Success)'
                else:
                    result = 'L'
                    reason = 'OT (pre-2010): no FG/TD scored on drive (Failure)'
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
        
        final_row = get_meaningful_final_play(drive_all)
        final_desc = final_row.get('desc')
        final_down = final_row.get('down')
        final_yds = final_row.get('ydstogo')
        
        final_down_str = f"{int(final_down)}down" if pd.notna(final_down) and int(final_down) > 0 else None
        final_yds_str = f"{int(final_yds)}yrdstogo" if pd.notna(final_yds) and int(final_yds) > 0 else None
        
        debug_rows.append({
            'qb_name': qb_name,
            'game_id': game_id,
            'period': period,
            'start_score_diff': f"down {abs(int(row['score_diff']))}",
            'start_time': row.get('time'),
            'end_time': last_play.get('time'),
            'final_down': final_down_str,
            'final_ydstogo': final_yds_str,
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
    
    records['qb_name'] = records.index.map(name_map)
    records['decisions'] = records['wins'] + records['losses']
    records['win_pct'] = records.apply(
        lambda x: round(x['wins'] / x['decisions'] * 100, 1) if x['decisions'] > 0 else 0.0,
        axis=1
    )
    
    records = records[records['decisions'] > 0]
    records = records[['qb_name', 'wins', 'losses', 'win_pct']].sort_values(
        ['wins', 'losses', 'win_pct'], 
        ascending=[False, True, False]
    )
    
    # Generate HTML
    generated_ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Legacy Drives Leaderboard - Post Season</title>
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
            columns: 1;
        }}
        .qb-entry {{
            break-inside: avoid;
            margin-bottom: 3px;
            cursor: pointer;
        }}
        .qb-entry:hover {{
            text-decoration: underline;
        }}
        .search-wrap {{
            margin: 10px 0 18px;
            position: relative;
            max-width: 520px;
        }}
        .search-row {{
            display: flex;
            gap: 8px;
            align-items: center;
        }}
        #playerSearch {{
            width: 100%;
            padding: 8px 10px;
            font-family: inherit;
            font-size: 12px;
            border: 1px solid #999;
            border-radius: 6px;
            outline: none;
        }}
        #clearSearch {{
            padding: 8px 10px;
            font-family: inherit;
            font-size: 12px;
            border: 1px solid #999;
            border-radius: 6px;
            background: transparent;
            cursor: pointer;
            white-space: nowrap;
        }}
        #clearSearch:disabled {{
            opacity: 0.5;
            cursor: default;
        }}
        .search-hint {{
            font-size: 10px;
            margin-top: 6px;
        }}
        .dropdown {{
            position: absolute;
            top: 38px;
            left: 0;
            right: 0;
            border: 1px solid #999;
            border-radius: 6px;
            background: #fff;
            z-index: 50;
            max-height: 220px;
            overflow: auto;
            display: none;
        }}
        .dropdown button {{
            width: 100%;
            text-align: left;
            padding: 8px 10px;
            font-family: inherit;
            font-size: 12px;
            border: 0;
            background: transparent;
            cursor: pointer;
        }}
        .dropdown button:hover {{
            background: #eee;
        }}
        .qb-details {{
            display: none;
            margin: 6px 0 10px;
            padding: 10px;
            border: 1px solid #999;
            border-radius: 6px;
            background: #fafafa;
            break-inside: avoid;
            width: 100%;
            box-sizing: border-box;
            overflow-x: auto;
        }}
        .qb-details table {{
            width: 100%;
            min-width: 800px;
            font-size: 10px;
            border-collapse: collapse;
        }}
        .qb-details th {{
            text-align: left;
            padding: 4px;
            border-bottom: 1px solid #999;
            font-weight: bold;
        }}
        .qb-details td {{
            padding: 4px;
            border-bottom: 1px solid #ddd;
        }}
        .qb-details tr:last-child td {{
            border-bottom: none;
        }}
        .result-w {{
            color: #0a7c0a;
            font-weight: bold;
        }}
        .result-l {{
            color: #c41e3a;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <h1>Legacy Drives Leaderboard (Post Season)</h1>
    <div class="subtitle">Seasons: 2000-{CURRENT_SEASON} | Sorted by Legacy Drive Successes</div>
    <div class="timestamp">Generated: {generated_ts}</div>
    
    <div class="criteria">
        <div class="criteria-title">Legacy Drive Opportunity Criteria:</div>
        Q4: Drive starts at 3:00 or less but > 0:30 left (Unless a Success), down 1–8 points. OT: All Drives
        
        <div class="section-header">Q4 Result:</div>
        Success = Lead or Tied at Drive End<br>
        Fail = Trailing at Drive End
        
        <div class="section-header">OT Result:</div>
        Success = TD on 1st Drive, Lead at Drive End on All Other Drives (pre-2010 FG on 1st Drive is Success)<br>
        Fail = FG or No Score on 1st OT Drive, Tied or Trailing on All Other Drives
        
        <div class="section-header" style="margin-top: 15px;">Name: LD Successes - LD Failures (Success %)</div>
    </div>
    <div class="search-wrap">
        <div class="search-row">
            <input
                id="playerSearch"
                type="text"
                autocomplete="off"
                placeholder="Search a QB (e.g., homes)"
            />
            <button id="clearSearch" type="button" disabled>Clear</button>
        </div>
        <div class="search-hint">
            Start typing to see matching players. Click a name to show only that player.
        </div>
        <div id="searchDropdown" class="dropdown"></div>
    </div>

    <div class="leaderboard">
"""
    
    # Add all QB entries with their detail divs
    for _, row in records.iterrows():
        qb_name = row["qb_name"]
        html_content += f'        <div class="qb-entry" data-qb="{qb_name}">{qb_name}: {int(row["wins"])} - {int(row["losses"])} ({row["win_pct"]}%)</div>\n'
        html_content += f'        <div class="qb-details" id="details-{qb_name.replace(" ", "-")}"></div>\n'
    
    html_content += """    </div>

<script>
(function () {
  const input = document.getElementById("playerSearch");
  const dropdown = document.getElementById("searchDropdown");
  const clearBtn = document.getElementById("clearSearch");
  const entries = Array.from(document.querySelectorAll(".qb-entry"));

  const players = entries.map(el => {
    const text = el.textContent.trim();
    return { name: (text.split(":")[0] || "").trim(), el };
  });

  function normalize(s) {
    return (s || "").toString().trim().toLowerCase();
  }

  function showAll() {
    players.forEach(p => {
      p.el.style.display = "";
      const detailsEl = p.el.nextElementSibling;
      if (detailsEl && detailsEl.classList.contains("qb-details")) {
        detailsEl.style.display = "none";
      }
    });
    clearBtn.disabled = true;
  }

  function showOnly(name) {
    const target = normalize(name);
    players.forEach(p => {
      const isMatch = normalize(p.name) === target;
      p.el.style.display = isMatch ? "" : "none";
      const detailsEl = p.el.nextElementSibling;
      if (detailsEl && detailsEl.classList.contains("qb-details")) {
        detailsEl.style.display = "none";
      }
    });
    clearBtn.disabled = false;
  }

  input.addEventListener("input", () => {
    const value = normalize(input.value);
    dropdown.innerHTML = "";

    if (!value) {
      showAll();
      dropdown.style.display = "none";
      return;
    }

    const matches = players.filter(p => normalize(p.name).includes(value));

    matches.forEach(p => {
      const btn = document.createElement("button");
      btn.textContent = p.name;
      btn.type = "button";
      btn.onclick = () => {
        input.value = p.name;
        showOnly(p.name);
        dropdown.style.display = "none";
      };
      dropdown.appendChild(btn);
    });

    dropdown.style.display = matches.length ? "block" : "none";
  });

  clearBtn.addEventListener("click", () => {
    input.value = "";
    showAll();
    dropdown.style.display = "none";
  });

  // Embedded debug data
  const EMBEDDED_DEBUG_DATA = """ + json.dumps(debug_rows) + """;
  
  function loadDebugData() {
    return EMBEDDED_DEBUG_DATA;
  }

  entries.forEach(entry => {
    entry.addEventListener("click", () => {
      const qbName = entry.getAttribute("data-qb");
      const detailsId = `details-${qbName.replace(/ /g, "-")}`;
      const detailsEl = document.getElementById(detailsId);
      
      if (!detailsEl) return;
      
      // Toggle visibility
      if (detailsEl.style.display === "block") {
        detailsEl.style.display = "none";
        return;
      }
      
      // Close all other details
      document.querySelectorAll(".qb-details").forEach(el => {
        el.style.display = "none";
      });
      
      detailsEl.style.display = "block";
      
      // Load data
      const rows = loadDebugData();
      let qbRows = rows.filter(r => r.qb_name === qbName);
      if (qbRows.length === 0) {
        const qbLower = qbName.toLowerCase();
        qbRows = rows.filter(r => r.qb_name && r.qb_name.toLowerCase() === qbLower);
      }
      
      if (qbRows.length === 0) {
        detailsEl.innerHTML = "<p>No drives found for this QB.</p>";
        return;
      }
      
      // Build table
      let tableHtml = `
        <table>
          <thead>
            <tr>
              <th>Result</th>
              <th>Game ID</th>
              <th>Score Diff</th>
              <th>Period</th>
              <th>Time Range</th>
              <th>Down</th>
              <th>Yds</th>
              <th>Final</th>
              <th>Reason</th>
            </tr>
          </thead>
          <tbody>
      `;
      
      qbRows.forEach(drive => {
        const resultClass = drive.result === 'W' ? 'result-w' : 'result-l';
        const timeRange = (drive.start_time && drive.end_time) 
          ? `${drive.start_time}-${drive.end_time}` 
          : (drive.start_time || drive.end_time || '');
        const finalScore = `${drive.end_team_score}-${drive.end_opp_score}`;
        
        tableHtml += `
          <tr>
            <td class="${resultClass}">${drive.result}</td>
            <td>${drive.game_id || ''}</td>
            <td>${drive.start_score_diff || ''}</td>
            <td>${drive.period || ''}</td>
            <td>${timeRange}</td>
            <td>${drive.final_down || ''}</td>
            <td>${drive.final_ydstogo || ''}</td>
            <td>${finalScore}</td>
            <td>${drive.reason || ''}</td>
          </tr>
        `;
      });
      
      tableHtml += `
          </tbody>
        </table>
      `;
      
      detailsEl.innerHTML = tableHtml;
    });
  });
})();
</script>
</body>
</html>"""
    
    # Save HTML file
    with open("postseason.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"\nGenerated postseason.html")
    print(f"Total QBs: {len(records)}")

else:
    print("No legacy drive opportunities found.")
