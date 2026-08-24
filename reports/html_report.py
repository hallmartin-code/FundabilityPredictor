"""Optional HTML rendering, for layout review and debugging.

Deliberately plain: it exists so the *content* of a report can be checked
without opening a PDF viewer, and so a layout problem can be attributed to
either the analysis or the PDF renderer.
"""

from __future__ import annotations

import html
import logging
from pathlib import Path

from extractors.normalization import format_money
from models.prediction import AnalysisResult
from reports.json_report import DISCLAIMER

LOGGER = logging.getLogger(__name__)

_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; padding: 28px; background: #EEF1F5; color: #101820;
       font: 13px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
.sheet { max-width: 900px; margin: 0 auto; background: #fff; padding: 0 0 22px;
         box-shadow: 0 2px 14px rgba(16,24,32,.12); border-radius: 4px; overflow: hidden; }
header { background: #14213D; color: #fff; padding: 18px 26px; }
header h1 { margin: 0; font-size: 26px; letter-spacing: -.4px; }
header .meta { color: #C6D3E6; font-size: 12px; margin-top: 4px; }
header .fin { font-weight: 700; font-size: 12px; margin-top: 6px; }
main { padding: 20px 26px; }
h2 { font-size: 11px; letter-spacing: .06em; text-transform: uppercase;
     color: #14213D; margin: 22px 0 8px; }
.tiles { display: grid; grid-template-columns: repeat(3, 1fr) 1.1fr; gap: 10px; }
.tile { border-radius: 5px; padding: 14px 12px; color: #fff; text-align: center; }
.tile .v { font-size: 30px; font-weight: 700; line-height: 1; }
.tile .l { font-size: 10px; letter-spacing: .06em; margin-top: 7px; }
.badge { border: 1px solid #D5DAE2; background: #F4F6F9; border-radius: 5px; padding: 11px 13px; }
.badge .s { font-size: 24px; font-weight: 700; color: #14213D; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 26px; }
ul { margin: 0; padding-left: 17px; }
li { margin-bottom: 6px; }
.sev { font-weight: 700; font-size: 11px; }
.CRITICAL { color: #9B1C1C; } .HIGH { color: #B03A2E; } .MODERATE { color: #B7791F; }
.why { color: #4A5568; font-size: 12px; }
table { border-collapse: collapse; width: 100%; font-size: 12px; }
td { padding: 3px 0; border-bottom: 1px solid #EDF0F4; }
td.v { text-align: right; font-weight: 700; }
.YES { color: #1B7F4B; } .NO { color: #B03A2E; }
.REVIEW { color: #B7791F; } .UNKNOWN { color: #7A8699; }
.bridge div { font-size: 12px; padding: 1px 0; }
.pos { color: #1B7F4B; font-weight: 700; } .neg { color: #B03A2E; font-weight: 700; }
footer { border-top: 1px solid #D5DAE2; margin: 22px 26px 0; padding-top: 10px;
         color: #7A8699; font-size: 10px; }
.conf { color: #9B1C1C; font-weight: 700; }
"""


def _e(value: object) -> str:
    return html.escape(str(value if value is not None else "Not Provided"))


def _tile_colour(value: float) -> str:
    if value >= 60:
        return "#1B7F4B"
    if value >= 30:
        return "#B7791F"
    return "#B03A2E"


def render_html(result: AnalysisResult, path: Path) -> Path:
    profile = result.profile
    final = result.prediction.final
    stage = result.stage

    stage_text = stage.inferred.value if stage.inferred.is_known else "Not Provided"
    if stage.declared.is_known and stage.declared is not stage.inferred:
        stage_text += f" (declared {stage.declared.value})"

    tiles = "".join(
        f'<div class="tile" style="background:{_tile_colour(v)}">'
        f'<div class="v">{v:.0f}%</div><div class="l">{_e(label)}</div></div>'
        for label, v in (("3 Months", final.p3), ("6 Months", final.p6), ("12 Months", final.p12))
    )

    drivers = "".join(f"<li>{_e(d)}</li>" for d in result.prediction.positive_signals) or (
        "<li>No positive financing signals were evidenced.</li>"
    )
    blockers = "".join(
        f'<li><span class="sev {b.severity}">{b.severity}</span> <b>{_e(b.title)}</b><br>'
        f'<span class="why">{_e(b.detail)}</span></li>'
        for b in result.blockers
    ) or "<li>No structural blockers identified.</li>"
    actions = "".join(
        f"<li><b>{_e(a.action)}</b> "
        f'<span class="why">[{a.impact} · {a.timeframe}]</span><br>'
        f'<span class="why">{_e(a.why)}</span></li>'
        for a in result.actions
    )
    flags = "".join(
        f'<tr><td>{_e(f.label)}</td><td class="v {f.value}">{f.value}</td></tr>'
        for f in result.ten_capital.flags
    )

    bridge_rows = []
    for step in result.prediction.bridge:
        if step.kind == "baseline":
            bridge_rows.append(f"<div><b>Baseline: {step.running.get('p6', 0):.0f}%</b></div>")
        else:
            delta = step.delta.get("p6", 0.0)
            css = "pos" if delta >= 0 else "neg"
            bridge_rows.append(
                f'<div><span class="{css}">{delta:+.1f}%</span> {_e(step.label)}</div>'
            )
    bridge_rows.append(f"<div><b>Final 6-month probability: {final.p6:.0f}%</b></div>")

    notes = "".join(f"<li>{_e(n)}</li>" for n in result.confidence.notes) or (
        "<li>The deck disclosed every probability-critical field.</li>"
    )

    valuation = (
        profile.number("pre_money_valuation")
        or profile.number("note_cap")
        or profile.number("safe_cap")
    )
    geography = (
        profile.text("headquarters")
        or profile.text("primary_geography")
        or profile.text("geographic_market")
        or "Not Provided"
    )

    document = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(result.company_name)} - Financing Probability Report</title>
<style>{_CSS}</style></head>
<body><div class="sheet">
<header>
  <h1>{_e(result.company_name)}</h1>
  <div class="meta">{_e(result.sector.label)} &nbsp;|&nbsp; {_e(stage_text)} &nbsp;|&nbsp; {_e(geography)}</div>
  <div class="fin">Target Raise {_e(format_money(profile.number('target_raise')))}
    &nbsp;·&nbsp; Committed {_e(format_money(profile.number('amount_committed')))}
    &nbsp;·&nbsp; Valuation {_e(format_money(valuation))}</div>
</header>
<main>
  <h2>Probability of Completing Round</h2>
  <div class="tiles">{tiles}
    <div class="badge">
      <div style="font-size:10px;color:#7A8699">FUNDABILITY</div>
      <div class="s">{result.readiness.total:.0f}<span style="font-size:13px;color:#4A5568">/100</span></div>
      <div style="font-size:11px;font-weight:700;color:#1F4E79">{_e(result.readiness.label)}</div>
      <div style="font-size:11px;margin-top:6px;color:#4A5568">Confidence
        <b>{_e(result.confidence.level)}</b> ({result.confidence.score:.0f})</div>
    </div>
  </div>

  <div class="cols">
    <div>
      <h2>Probability Drivers</h2><ul>{drivers}</ul>
      <h2>Critical Blockers</h2><ul>{blockers}</ul>
      <h2>Comparable Financing Profile</h2>
      <p><b>{_e(result.comparable.label)}</b><br>
         <span class="why">{_e(result.comparable.rationale)}</span></p>
      <p class="why"><i>{_e(result.prediction.market_condition.disclosure)}</i></p>
    </div>
    <div>
      <h2>Five Actions to Increase Funding Probability</h2><ol>{actions}</ol>
      <h2>TEN Capital Assessment</h2>
      <p><b>Fit Score {result.ten_capital.fit_score:.0f}/100</b> ·
         <b>{_e(result.ten_capital.recommended_next_step)}</b><br>
         <span class="why">{_e(result.ten_capital.explanation)}</span></p>
      <table>{flags}</table>
    </div>
  </div>

  <div class="cols">
    <div><h2>How the 6-Month Probability Was Reached</h2>
      <div class="bridge">{''.join(bridge_rows)}</div></div>
    <div><h2>What Limits Confidence</h2><ul>{notes}</ul></div>
  </div>
</main>
<footer>
  <p>Financing probability estimates are model-based assessments, not guarantees of financing outcomes.</p>
  <p>{_e(DISCLAIMER)}</p>
  <p class="conf">CONFIDENTIAL - FOR INTENDED RECIPIENT ONLY</p>
  <p>Compiled on {result.generated_at.strftime('%d %B %Y')} by TEN Capital Network</p>
</footer>
</div></body></html>"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    LOGGER.info("Wrote %s", path)
    return path
