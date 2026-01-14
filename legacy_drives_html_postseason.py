from __future__ import annotations
import json
import os
import warnings
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Sequence, Set, Tuple
import nfl_data_py as nfl
import pandas as pd
from pandas.errors import PerformanceWarning
warnings.simplefilter("ignore", PerformanceWarning)
warnings.simplefilter("ignore", category=pd.errors.SettingWithCopyWarning)

CURRENT_SEASON = 2025
CACHE_FILE = "legacy_drives_postseason_cache_website.json"
NAME_MAP_FILE = "qb_name_map_postseason_website.json"
OUTPUT_HTML = "postseason.html"

Opportunity = Dict[str, Any]
LegacyDriveRow = Dict[str, Any]

#Load saved JSON files if available.
def load_cache(cache_file: str = CACHE_FILE) -> Tuple[List[Opportunity], Set[str], int, List[LegacyDriveRow]]:
    if not os.path.exists(cache_file):
        return [], set(), 2000, []

    with open(cache_file, "r", encoding="utf-8") as f:
        data = json.load(f) or {}

    opportunities = data.get("opportunities", []) or []
    processed_games = set(data.get("processed_games", []) or [])
    last_season_processed = int(data.get("last_season_processed", 2000) or 2000)

    # "legacydrive_rows" is the canonical key; still read "debug_rows" for backward compatibility.
    legacydrive_rows = data.get("legacydrive_rows", data.get("debug_rows", [])) or []

    return opportunities, processed_games, last_season_processed, legacydrive_rows

#Save new JSON file with updated games.
def save_cache(
    opportunities: Sequence[Opportunity],
    processed_games: Set[str],
    last_season_processed: int,
    legacydrive_rows: Sequence[LegacyDriveRow],
    cache_file: str = CACHE_FILE,
) -> None:
    payload = {
        "opportunities": list(opportunities),
        "processed_games": list(processed_games),
        "last_season_processed": int(last_season_processed),
        "legacydrive_rows": list(legacydrive_rows),
        "last_updated": datetime.now().isoformat(),
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(payload, f)

def load_persistent_name_map(name_map_file: str = NAME_MAP_FILE) -> Dict[str, str]:
    if not os.path.exists(name_map_file):
        return {}
    with open(name_map_file, "r", encoding="utf-8") as f:
        return json.load(f) or {}

def save_persistent_name_map(name_map: Dict[str, str], name_map_file: str = NAME_MAP_FILE) -> None:
    with open(name_map_file, "w", encoding="utf-8") as f:
        json.dump(name_map, f)

#Only load seasons not saved in JSON file.
def seasons_to_load(last_season_processed: int, current_season: int) -> List[int]:
    if last_season_processed < current_season - 1:
        return list(range(last_season_processed, current_season + 1))
    return [current_season]

#Import PBP data
def import_postseason_pbp(seasons: Sequence[int]) -> pd.DataFrame:
    pbp = nfl.import_pbp_data(list(seasons), downcast=True, cache=False)
    return pbp[pbp["season_type"] == "POST"]

#Determine when drive starts
def build_drive_starts(pbp_period: pd.DataFrame) -> pd.DataFrame:
    return (
        pbp_period.sort_values(["game_id", "drive", "play_id"])
        .groupby(["game_id", "drive"], as_index=False)
        .first()
    )

#Determine the QB on the drive.
def get_qb_for_drive(
    drive_all: pd.DataFrame,
    drive_start_row: pd.Series,
    qb_name_map: Dict[Any, str],
    passer_name_map: Dict[Any, str],
) -> Tuple[str, str]:
    qb_id = None
    qb_name = None

    # Prefer qb_id/qb columns if present (more direct).
    if "qb_id" in drive_all.columns:
        qb_series = drive_all["qb_id"].dropna()
        if not qb_series.empty:
            qb_id = qb_series.mode().iloc[0]
            qb_name = qb_name_map.get(qb_id)

    # Fallback: infer from most common passer_id on pass attempts.
    if qb_id is None and "pass_attempt" in drive_all.columns:
        drive_pass = drive_all[drive_all["pass_attempt"] == 1]
        if not drive_pass.empty and "passer_id" in drive_pass.columns:
            qb_counts = Counter(drive_pass["passer_id"].dropna())
            if qb_counts:
                qb_id = qb_counts.most_common(1)[0][0]
                qb_name = passer_name_map.get(qb_id)

    # Final fallback: team placeholder.
    if qb_id is None:
        qb_id = f"TEAM_{drive_start_row.get('posteam', 'UNK')}"
        qb_name = qb_id

    return str(qb_id), str(qb_name) if qb_name is not None else str(qb_id)

#Find the important final play. Either TD or FG attempt, skips extra points and timeouts.
def get_meaningful_final_play(drive_all: pd.DataFrame) -> pd.Series:
    for _, row in drive_all.iterrows():
        desc_txt = str(row.get("desc", "") or "")
        ptype = row.get("play_type")

        if (
            ptype not in ["extra_point", "two_point_attempt"]
            and "END GAME" not in desc_txt
            and not desc_txt.strip().startswith("Timeout")
            and not desc_txt.strip().startswith("TWO-POINT ATTEMPT")
        ):
            return row
    # Fallback: first row (preserves legacy behavior).
    return drive_all.iloc[0]

def period_order(p: Any) -> int:
    if p == "Q4":
        return 4
    if p == "OT":
        return 5
    return 99

def time_to_seconds(t: Any) -> int:
    if not t or not isinstance(t, str) or ":" not in t:
        return -1
    try:
        m, s = t.split(":", 1)
        return int(m) * 60 + int(s)
    except Exception:
        return -1

def sort_legacydrive_rows(rows: List[LegacyDriveRow]) -> List[LegacyDriveRow]:
    return sorted(
        rows,
        key=lambda r: (
            int(r.get("season") or 0),
            int(r.get("week") or 0),
            str(r.get("game_id") or ""),
            period_order(r.get("period")),
            -time_to_seconds(r.get("start_time")),
        ),
    )

#Build the legacy drive leaderboard
def build_leaderboard_records(opportunities: Sequence[Opportunity], name_map: Dict[str, str]) -> pd.DataFrame:
    df = pd.DataFrame(opportunities)
    records = (
        df.pivot_table(index="qb_id", columns="result", aggfunc="size", fill_value=0)
        .rename(columns={"W": "wins", "L": "losses"})
    )
    # Exclude team placeholders.
    records = records[~records.index.astype(str).str.startswith("TEAM_")]
    # Ensure required columns exist.
    for c in ("wins", "losses"):
        if c not in records.columns:
            records[c] = 0
    records["qb_name"] = records.index.map(name_map)
    records["decisions"] = records["wins"] + records["losses"]
    records["win_pct"] = records.apply(
        lambda x: round(x["wins"] / x["decisions"] * 100, 1) if x["decisions"] > 0 else 0.0,
        axis=1,
    )
    records = records[records["decisions"] > 0]
    records = records[["qb_name", "wins", "losses", "win_pct"]].sort_values(
        ["wins", "losses", "win_pct"], ascending=[False, True, False]
    )
    return records

#Generate the HTML file
def generate_html(records: pd.DataFrame, legacydrive_rows: List[LegacyDriveRow]) -> str:
    generated_ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
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

    for _, row in records.iterrows():
        qb_name = row["qb_name"]
        html += f'        <div class="qb-entry" data-qb="{qb_name}">{qb_name}: {int(row["wins"])} - {int(row["losses"])} ({row["win_pct"]}%)</div>\n'
        html += f'        <div class="qb-details" id="details-{qb_name.replace(" ", "-")}"></div>\n'

    embedded_data = json.dumps(legacydrive_rows)

    html += f"""    </div>

<script>
(function () {{
  const input = document.getElementById("playerSearch");
  const dropdown = document.getElementById("searchDropdown");
  const clearBtn = document.getElementById("clearSearch");
  const entries = Array.from(document.querySelectorAll(".qb-entry"));

  const players = entries.map(el => {{
    const text = el.textContent.trim();
    return {{ name: (text.split(":")[0] || "").trim(), el }};
  }});

  function normalize(s) {{
    return (s || "").toString().trim().toLowerCase();
  }}

  function showAll() {{
    players.forEach(p => {{
      p.el.style.display = "";
      const detailsEl = p.el.nextElementSibling;
      if (detailsEl && detailsEl.classList.contains("qb-details")) {{
        detailsEl.style.display = "none";
      }}
    }});
    clearBtn.disabled = true;
  }}

  function showOnly(name) {{
    const target = normalize(name);
    players.forEach(p => {{
      const isMatch = normalize(p.name) === target;
      p.el.style.display = isMatch ? "" : "none";
      const detailsEl = p.el.nextElementSibling;
      if (detailsEl && detailsEl.classList.contains("qb-details")) {{
        detailsEl.style.display = "none";
      }}
    }});
    clearBtn.disabled = false;
  }}

  input.addEventListener("input", () => {{
    const value = normalize(input.value);
    dropdown.innerHTML = "";

    if (!value) {{
      showAll();
      dropdown.style.display = "none";
      return;
    }}

    const matches = players.filter(p => normalize(p.name).includes(value));

    matches.forEach(p => {{
      const btn = document.createElement("button");
      btn.textContent = p.name;
      btn.type = "button";
      btn.onclick = () => {{
        input.value = p.name;
        showOnly(p.name);
        dropdown.style.display = "none";
      }};
      dropdown.appendChild(btn);
    }});

    dropdown.style.display = matches.length ? "block" : "none";
  }});

  clearBtn.addEventListener("click", () => {{
    input.value = "";
    showAll();
    dropdown.style.display = "none";
  }});

  // Embedded legacy-drive rows (formerly called "debug" rows)
  const EMBEDDED_LEGACYDRIVE_DATA = {embedded_data};

  function loadLegacyDriveData() {{
    return EMBEDDED_LEGACYDRIVE_DATA;
  }}

  entries.forEach(entry => {{
    entry.addEventListener("click", () => {{
      const qbName = entry.getAttribute("data-qb");
      const detailsId = `details-${{qbName.replace(/ /g, "-")}}`;
      const detailsEl = document.getElementById(detailsId);

      if (!detailsEl) return;

      // Toggle visibility
      if (detailsEl.style.display === "block") {{
        detailsEl.style.display = "none";
        return;
      }}

      // Close all other details
      document.querySelectorAll(".qb-details").forEach(el => {{
        el.style.display = "none";
      }});

      detailsEl.style.display = "block";

      // Load data
      const rows = loadLegacyDriveData();
      let qbRows = rows.filter(r => r.qb_name === qbName);
      if (qbRows.length === 0) {{
        const qbLower = qbName.toLowerCase();
        qbRows = rows.filter(r => r.qb_name && r.qb_name.toLowerCase() === qbLower);
      }}

      // Sort: earliest season/week first, then game/period (Q4 before OT), then time.
      qbRows.sort((a, b) => {{
        const aSeason = Number(a.season || 0);
        const bSeason = Number(b.season || 0);
        if (aSeason !== bSeason) return aSeason - bSeason;

        const aWeek = Number(a.week || 0);
        const bWeek = Number(b.week || 0);
        if (aWeek !== bWeek) return aWeek - bWeek;

        const aGame = String(a.game_id || "");
        const bGame = String(b.game_id || "");
        if (aGame !== bGame) return aGame.localeCompare(bGame);

        const periodOrder = (p) => (p === "Q4" ? 4 : (p === "OT" ? 5 : 99));
        const aP = periodOrder(a.period);
        const bP = periodOrder(b.period);
        if (aP !== bP) return aP - bP;

        // "time" in nfl_data_py is mm:ss remaining in the quarter. Sort higher first (earlier in period).
        const toSec = (t) => {{
          if (!t) return -1;
          const parts = String(t).split(":");
          if (parts.length !== 2) return -1;
          const m = Number(parts[0]);
          const s = Number(parts[1]);
          if (Number.isNaN(m) || Number.isNaN(s)) return -1;
          return m * 60 + s;
        }};
        return toSec(b.start_time) - toSec(a.start_time);
      }});

      if (qbRows.length === 0) {{
        detailsEl.innerHTML = "<p>No drives found for this QB.</p>";
        return;
      }}

      // Build table
      let tableHtml = `
        <table>
          <thead>
            <tr>
              <th>Result</th>
              <th>Year</th>
              <th>Week</th>
              <th>Away Team</th>
              <th>Home Team</th>
              <th>Period</th>
              <th>Score Diff</th>
              <th>Time Range</th>
              <th>Down</th>
              <th>Yards To Go</th>
              <th>Final Play of Drive</th>
              <th>New Score</th>
              <th>Result Explanation</th>
            </tr>
          </thead>
          <tbody>
      `;

      qbRows.forEach(drive => {{
        const resultClass = drive.result === 'W' ? 'result-w' : 'result-l';
        const timeRange = (drive.start_time && drive.end_time) 
          ? `${{drive.start_time}}-${{drive.end_time}}` 
          : (drive.start_time || drive.end_time || '');
        const finalScore = `${{drive.end_team_score}}-${{drive.end_opp_score}}`;

        tableHtml += `
          <tr>
            <td class="${{resultClass}}">${{drive.result}}</td>
            <td>${{drive.season || ''}}</td>
            <td>${{drive.week || ''}}</td>
            <td>${{drive.away_team || ''}}</td>
            <td>${{drive.home_team || ''}}</td>
            <td>${{drive.period || ''}}</td>
            <td>${{drive.start_score_diff || ''}}</td>
            <td>${{timeRange}}</td>
            <td>${{drive.final_down || ''}}</td>
            <td>${{drive.final_yds || drive.final_ydstogo || ''}}</td>
            <td>${{drive.final_play || ''}}</td>
            <td>${{finalScore}}</td>
            <td>${{drive.reason || ''}}</td>
          </tr>
        `;
      }});

      tableHtml += `
          </tbody>
        </table>
      `;

      detailsEl.innerHTML = tableHtml;
    }});
  }});
}})();
</script>
</body>
</html>"""
    return html


def process_new_games(
    pbp: pd.DataFrame,
    new_games: Set[str],
    opportunities: List[Opportunity],
    legacydrive_rows: List[LegacyDriveRow],
    passer_name_map: Dict[Any, str],
    qb_name_map: Dict[Any, str],
) -> None:
    if not new_games:
        return

    pbp = pbp[pbp["game_id"].isin(new_games)]
    if pbp.empty:
        return

    pbp_q4 = pbp[pbp["qtr"] == 4]
    pbp_ot = pbp[pbp["qtr"] >= 5]

    drive_starts_q4 = build_drive_starts(pbp_q4)
    drive_starts_q4["score_diff"] = drive_starts_q4["posteam_score"] - drive_starts_q4["defteam_score"]
    q4_opps = drive_starts_q4[
        (drive_starts_q4["quarter_seconds_remaining"] <= 180)
        & (drive_starts_q4["quarter_seconds_remaining"] >= 0)
        & (drive_starts_q4["score_diff"].between(-8, -1))
    ].copy()
    q4_opps["period"] = "Q4"

    drive_starts_ot = build_drive_starts(pbp_ot)
    drive_starts_ot["score_diff"] = drive_starts_ot["posteam_score"] - drive_starts_ot["defteam_score"]
    ot_opps = drive_starts_ot.copy()
    ot_opps["period"] = "OT"
    ot_opps = ot_opps.sort_values(["game_id", "qtr", "play_id"])
    ot_opps["ot_drive_rank"] = ot_opps.groupby("game_id").cumcount() + 1

    opps = pd.concat([q4_opps, ot_opps], ignore_index=True)

    pbp_q4_grouped = pbp_q4.groupby(["game_id", "drive"])
    pbp_ot_grouped = pbp_ot.groupby(["game_id", "drive"])

    for _, row in opps.iterrows():
        game_id = row["game_id"]
        drive_num = row["drive"]
        period = row.get("period", "Q4")

        try:
            drive_all = (
                pbp_q4_grouped.get_group((game_id, drive_num)).copy()
                if period == "Q4"
                else pbp_ot_grouped.get_group((game_id, drive_num)).copy()
            )
        except KeyError:
            continue

        qb_id, qb_name = get_qb_for_drive(drive_all, row, qb_name_map, passer_name_map)

        sort_col = "game_seconds_remaining" if "game_seconds_remaining" in drive_all.columns else "quarter_seconds_remaining"
        drive_all = drive_all.sort_values([sort_col, "play_id"], ascending=[True, False])

        # Fill post scores for robust end-of-drive score inference.
        for c in ("posteam_score_post", "defteam_score_post"):
            if c in drive_all.columns:
                drive_all[c] = drive_all[c].ffill().bfill()

        last_play = drive_all.iloc[0]
        end_team_score = last_play.get("posteam_score_post", pd.NA)
        end_opp_score = last_play.get("defteam_score_post", pd.NA)

        if pd.isna(end_team_score) or pd.isna(end_opp_score):
            candidates = drive_all.dropna(subset=["posteam_score_post", "defteam_score_post"])
            if candidates.empty:
                continue
            last_play = candidates.iloc[0]
            end_team_score = last_play["posteam_score_post"]
            end_opp_score = last_play["defteam_score_post"]

        if period == "Q4":
            if end_team_score >= end_opp_score:
                result = "W"
                reason = "Q4: drive ended tied or leading (Success)"
            else:
                result = "L"
                reason = "Q4: drive ended still trailing (Failure)"
        else:
            ot_rank = int(row.get("ot_drive_rank", 1))
            start_posteam = row.get("posteam")

            season_val = row.get("season")
            try:
                season_val = int(season_val) if pd.notna(season_val) else None
            except (ValueError, TypeError):
                season_val = None

            td_scored = False
            if "touchdown" in drive_all.columns:
                if "td_team" in drive_all.columns and start_posteam is not None:
                    td_scored = ((drive_all["touchdown"] == 1) & (drive_all["td_team"] == start_posteam)).any()
                elif "posteam" in drive_all.columns and start_posteam is not None:
                    td_scored = ((drive_all["touchdown"] == 1) & (drive_all["posteam"] == start_posteam)).any()
                else:
                    td_scored = (drive_all["touchdown"] == 1).any()

            fg_scored = False
            if "field_goal_result" in drive_all.columns:
                if "posteam" in drive_all.columns and start_posteam is not None:
                    fg_scored = ((drive_all["field_goal_result"] == "made") & (drive_all["posteam"] == start_posteam)).any()
                else:
                    fg_scored = (drive_all["field_goal_result"] == "made").any()

            if season_val is not None and season_val < 2010:
                if td_scored or fg_scored:
                    result = "W"
                    reason = "OT (pre-2010): FG/TD scored on drive (Success)"
                else:
                    result = "L"
                    reason = "OT (pre-2010): no FG/TD scored on drive (Failure)"
            else:
                if ot_rank == 1:
                    if td_scored:
                        result = "W"
                        reason = "OT (1st drive): TD scored (Success)"
                    else:
                        result = "L"
                        reason = "OT (1st drive): no TD (FG or no score) (Failure)"
                else:
                    if end_team_score > end_opp_score:
                        result = "W"
                        reason = f"OT (drive {ot_rank}): ended leading (Success)"
                    else:
                        result = "L"
                        reason = f"OT (drive {ot_rank}): ended not leading (Failure)"

        # Legacy rule: skip Q4 opportunities that start with <= 0:30 left and end in failure.
        if period == "Q4":
            start_qsr = row.get("quarter_seconds_remaining")
            if pd.notna(start_qsr) and start_qsr <= 30 and result == "L":
                continue

        opportunities.append({"qb_id": qb_id, "result": result})

        final_row = get_meaningful_final_play(drive_all)
        final_desc = final_row.get("desc")
        final_down = final_row.get("down")
        final_yds = final_row.get("ydstogo")

        final_down_str = f"{int(final_down)}down" if pd.notna(final_down) and int(final_down) > 0 else None
        final_yds_str = f"{int(final_yds)}yrdstogo" if pd.notna(final_yds) and int(final_yds) > 0 else None

        legacydrive_rows.append(
            {
                "qb_name": qb_name,
                "season": int(row.get("season")) if pd.notna(row.get("season")) else None,
                "week": int(row.get("week")) if pd.notna(row.get("week")) else None,
                "away_team": row.get("away_team"),
                "home_team": row.get("home_team"),
                "game_id": game_id,
                "period": period,
                "start_score_diff": f"down {abs(int(row['score_diff']))}",
                "start_time": row.get("time"),
                "end_time": last_play.get("time"),
                "final_down": final_down_str,
                "final_ydstogo": final_yds_str,
                "final_play": final_desc,
                "end_team_score": int(end_team_score),
                "end_opp_score": int(end_opp_score),
                "result": result,
                "reason": reason,
            }
        )

def main() -> None:
    cached_opportunities, processed_games, last_season_processed, cached_legacydrive_rows = load_cache()
    print(f"Loaded {len(cached_opportunities)} cached opportunities from {len(processed_games)} games")
    print(f"Last season fully processed: {last_season_processed}")
    seasons = seasons_to_load(last_season_processed, CURRENT_SEASON)
    if len(seasons) > 1:
        print(f"Loading seasons {seasons[0]}-{seasons[-1]} to catch up...")
    else:
        print(f"Only loading current season: {seasons[0]}")
    pbp_all = import_postseason_pbp(seasons)
    all_games = set(pbp_all["game_id"].unique())
    new_games = all_games - processed_games
    print(f"Found {len(new_games)} new games to process")
    if new_games:
        pbp = pbp_all[pbp_all["game_id"].isin(new_games)]
    else:
        print("No new games to process. Using cached data only.")
        pbp = pd.DataFrame()

        # Build name maps for display.
        passer_name_map: Dict[Any, str] = (
            pbp.groupby("passer_id")["passer"].first().to_dict() if not pbp.empty else {}
        )
        qb_name_map: Dict[Any, str] = {}
        if not pbp.empty and "qb_id" in pbp.columns and "qb" in pbp.columns:
            qb_name_map = pbp.groupby("qb_id")["qb"].first().to_dict()

        # Merge in persistent qb name map (and keep it updated).
        persistent_name_map = load_persistent_name_map()
        passer_name_map = {**persistent_name_map, **passer_name_map}
        opportunities = list(cached_opportunities)
        legacydrive_rows = list(cached_legacydrive_rows)
        if new_games:
            process_new_games(pbp, new_games, opportunities, legacydrive_rows, passer_name_map, qb_name_map)
            processed_games.update(new_games)
            # Preserve legacy logic for advancing last_season_processed.
            if seasons[-1] < CURRENT_SEASON:
                last_season_processed = seasons[-1]
            elif len(seasons) > 1:
                last_season_processed = CURRENT_SEASON - 1
            save_cache(opportunities, processed_games, last_season_processed, legacydrive_rows)
            save_persistent_name_map(passer_name_map)
            print(f"Processed {len(new_games)} new games. Total opportunities: {len(opportunities)}")

        # Generate HTML output
        if not opportunities:
            print("No legacy drive opportunities found.")
            return
        records = build_leaderboard_records(opportunities, passer_name_map)

        # Ensure the per-drive rows are in a stable chronological order for embedding/caching.
        legacydrive_rows_sorted = sort_legacydrive_rows(legacydrive_rows)
        html_content = generate_html(records, legacydrive_rows_sorted)
        with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"\nGenerated {OUTPUT_HTML}")
        print(f"Total QBs: {len(records)}")

if __name__ == "__main__":
    main()
