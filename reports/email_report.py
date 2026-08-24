"""Compose the analysis notification email.

Written for mail clients, not browsers: table layout and inline styles only,
no flexbox, grid or external stylesheets. Outlook in particular ignores most
modern CSS, and a report that arrives unreadable is worse than no email.

A plain-text alternative is always included - it is what text-only clients
render, and its absence is a well-known spam signal.
"""

from __future__ import annotations

import html
from pathlib import Path

from extractors.normalization import format_money
from models.prediction import AnalysisResult
from reports.json_report import DISCLAIMER
from services.email import Attachment, EmailMessage

NOT_PROVIDED = "Not Provided"

NAVY = "#14213D"
INK = "#101820"
SLATE = "#4A5568"
MUTED = "#7A8699"
RULE = "#D5DAE2"
PANEL = "#F4F6F9"
GREEN = "#1B7F4B"
AMBER = "#B7791F"
RED = "#B03A2E"
CRIT = "#9B1C1C"

SEVERITY_COLOUR = {"CRITICAL": CRIT, "HIGH": RED, "MODERATE": AMBER}


def _e(value: object) -> str:
    return html.escape(str(value))


def _probability_colour(value: float) -> str:
    if value >= 60:
        return GREEN
    if value >= 30:
        return AMBER
    return RED


def _stage_text(result: AnalysisResult) -> str:
    stage = result.stage
    if not stage.inferred.is_known:
        return NOT_PROVIDED
    if stage.declared.is_known and stage.declared is not stage.inferred:
        return f"{stage.inferred.value} (declared {stage.declared.value})"
    return stage.inferred.value


def subject_for(result: AnalysisResult) -> str:
    """Subject lines are scanned in a list, so lead with the decision."""
    company = result.company_name or "Unknown company"
    return (
        f"[{result.ten_capital.recommended_next_step.split(' - ')[0]}] "
        f"{company} - {result.prediction.final.p6:.0f}% 6-month financing probability"
    )


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
def _tile(label: str, value: float) -> str:
    return (
        f'<td width="33%" align="center" style="padding:0 4px;">'
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">'
        f'<tr><td align="center" bgcolor="{_probability_colour(value)}" '
        f'style="border-radius:5px;padding:14px 6px;color:#ffffff;">'
        f'<div style="font:700 28px/1 Helvetica,Arial,sans-serif;">{value:.0f}%</div>'
        f'<div style="font:700 10px/1.4 Helvetica,Arial,sans-serif;letter-spacing:.06em;'
        f'padding-top:7px;">{_e(label.upper())}</div>'
        f"</td></tr></table></td>"
    )


def _facts_rows(rows: list[tuple[str, str]]) -> str:
    out = []
    for key, value in rows:
        out.append(
            f'<tr>'
            f'<td style="padding:4px 14px 4px 0;color:{MUTED};'
            f'font:400 13px/1.5 Helvetica,Arial,sans-serif;white-space:nowrap;">{_e(key)}</td>'
            f'<td style="padding:4px 0;color:{INK};'
            f'font:600 13px/1.5 Helvetica,Arial,sans-serif;">{_e(value)}</td></tr>'
        )
    return "".join(out)


def _section(title: str, body: str) -> str:
    return (
        f'<tr><td style="padding:22px 0 0;">'
        f'<div style="font:700 11px/1.4 Helvetica,Arial,sans-serif;letter-spacing:.07em;'
        f'text-transform:uppercase;color:{NAVY};padding-bottom:9px;'
        f'border-bottom:1px solid {RULE};margin-bottom:10px;">{_e(title)}</div>'
        f'{body}</td></tr>'
    )


def build_html(result: AnalysisResult, attachment_names: list[str]) -> str:
    profile = result.profile
    final = result.prediction.final
    valuation = (
        profile.number("pre_money_valuation")
        or profile.number("note_cap")
        or profile.number("safe_cap")
    )

    blockers = "".join(
        f'<tr><td style="padding:3px 0;font:400 13px/1.5 Helvetica,Arial,sans-serif;color:{INK};">'
        f'<span style="font:700 11px/1.5 Helvetica,Arial,sans-serif;'
        f'color:{SEVERITY_COLOUR.get(b.severity, SLATE)};">{_e(b.severity)}</span>&nbsp;&nbsp;'
        f"<strong>{_e(b.title)}</strong><br>"
        f'<span style="color:{SLATE};">{_e(b.detail)}</span></td></tr>'
        for b in result.blockers
    ) or (
        f'<tr><td style="font:400 13px/1.5 Helvetica,Arial,sans-serif;color:{SLATE};">'
        f"No structural blockers identified.</td></tr>"
    )

    actions = "".join(
        f'<tr><td style="padding:3px 0;font:400 13px/1.5 Helvetica,Arial,sans-serif;color:{INK};">'
        f"<strong>{index}.</strong> {_e(a.action)} "
        f'<span style="color:{MUTED};">[{_e(a.impact)} &middot; {_e(a.timeframe)}]</span></td></tr>'
        for index, a in enumerate(result.actions, start=1)
    )

    drivers = "".join(
        f'<tr><td style="padding:2px 0;font:400 13px/1.5 Helvetica,Arial,sans-serif;color:{INK};">'
        f"&bull; {_e(d)}</td></tr>"
        for d in result.prediction.positive_signals[:5]
    ) or (
        f'<tr><td style="font:400 13px/1.5 Helvetica,Arial,sans-serif;color:{SLATE};">'
        f"No positive financing signals were evidenced in the deck.</td></tr>"
    )

    flags = "".join(
        f'<tr><td style="padding:2px 12px 2px 0;color:{SLATE};'
        f'font:400 12.5px/1.5 Helvetica,Arial,sans-serif;">{_e(f.label)}</td>'
        f'<td align="right" style="padding:2px 0;'
        f'font:700 12.5px/1.5 Helvetica,Arial,sans-serif;color:'
        f'{ {"YES": GREEN, "NO": RED, "REVIEW": AMBER}.get(f.value, MUTED) };">'
        f"{_e(f.value)}</td></tr>"
        for f in result.ten_capital.flags
    )

    llm = profile.llm_run
    extraction_rows = [
        ("Fields found", f"{len(profile.known_fields())} of 81"),
        ("Extraction engine", f"Rules + {llm['model']}" if llm else "Rules only (deterministic)"),
    ]
    if llm:
        extraction_rows += [
            ("Accepted from Claude", f"{llm['fields_accepted']} of {llm['fields_proposed']} proposed"),
            ("Estimated API cost", f"${llm['estimated_cost_usd']:.3f}"),
        ]
        if llm["rejected_unverified"]:
            extraction_rows.append(
                ("Discarded as unverified", ", ".join(llm["rejected_unverified"]))
            )

    attachment_list = "".join(
        f'<tr><td style="padding:2px 0;font:400 13px/1.5 Helvetica,Arial,sans-serif;color:{INK};">'
        f"&bull; {_e(name)}</td></tr>"
        for name in attachment_names
    ) or (
        f'<tr><td style="font:400 13px/1.5 Helvetica,Arial,sans-serif;color:{SLATE};">'
        f"No files attached.</td></tr>"
    )

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(result.company_name)} - Financing Probability</title></head>
<body style="margin:0;padding:0;background:#EEF1F5;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" bgcolor="#EEF1F5">
<tr><td align="center" style="padding:24px 12px;">
<table role="presentation" width="640" cellpadding="0" cellspacing="0" border="0"
       style="max-width:640px;width:100%;background:#ffffff;border:1px solid {RULE};border-radius:6px;">

  <tr><td bgcolor="{NAVY}" style="padding:20px 24px;border-radius:6px 6px 0 0;">
    <div style="font:700 21px/1.25 Helvetica,Arial,sans-serif;color:#ffffff;">
      {_e(result.company_name)}</div>
    <div style="font:400 12.5px/1.6 Helvetica,Arial,sans-serif;color:#C6D3E6;padding-top:3px;">
      {_e(result.sector.label)} &nbsp;|&nbsp; {_e(_stage_text(result))}</div>
    <div style="font:700 12.5px/1.6 Helvetica,Arial,sans-serif;color:#ffffff;padding-top:5px;">
      Target Raise {_e(format_money(profile.number('target_raise')))} &nbsp;&middot;&nbsp;
      Committed {_e(format_money(profile.number('amount_committed')))} &nbsp;&middot;&nbsp;
      Valuation {_e(format_money(valuation))}</div>
  </td></tr>

  <tr><td style="padding:20px 24px 0;">
    <div style="font:700 11px/1.4 Helvetica,Arial,sans-serif;letter-spacing:.07em;
                text-transform:uppercase;color:{NAVY};padding-bottom:10px;">
      Probability of completing the round</div>
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"><tr>
      {_tile("3 Months", final.p3)}{_tile("6 Months", final.p6)}{_tile("12 Months", final.p12)}
    </tr></table>
  </td></tr>

  <tr><td style="padding:0 24px;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
      {_section("Assessment", f'<table role="presentation" cellpadding="0" cellspacing="0" border="0">' + _facts_rows([
          ("Fundability score", f"{result.readiness.total:.0f}/100 - {result.readiness.label}"),
          ("Prediction confidence", f"{result.confidence.level} ({result.confidence.score:.0f}/100)"),
          ("Comparable profile", result.comparable.label),
          ("TEN Capital fit", f"{result.ten_capital.fit_score:.0f}/100"),
          ("Recommended next step", result.ten_capital.recommended_next_step),
          ("Market condition", result.prediction.market_condition.disclosure),
      ]) + '</table>')}

      {_section("Probability drivers",
                '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">'
                + drivers + '</table>')}

      {_section("Critical blockers",
                '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">'
                + blockers + '</table>')}

      {_section("Five actions to increase funding probability",
                '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">'
                + actions + '</table>')}

      {_section("TEN Capital readiness flags",
                '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">'
                + flags + '</table>')}

      {_section("Extraction",
                '<table role="presentation" cellpadding="0" cellspacing="0" border="0">'
                + _facts_rows(extraction_rows) + '</table>')}

      {_section("Attached",
                '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">'
                + attachment_list + '</table>')}
    </table>
  </td></tr>

  <tr><td style="padding:22px 24px 24px;">
    <div style="border-top:1px solid {RULE};padding-top:12px;
                font:400 10.5px/1.5 Helvetica,Arial,sans-serif;color:{MUTED};">
      <p style="margin:0 0 5px;">Financing probability estimates are model-based assessments,
         not guarantees of financing outcomes.</p>
      <p style="margin:0 0 5px;">{_e(DISCLAIMER)}</p>
      <p style="margin:0 0 5px;color:{CRIT};font-weight:700;">
         CONFIDENTIAL &mdash; FOR INTENDED RECIPIENT ONLY</p>
      <p style="margin:0;">Compiled on {result.generated_at.strftime('%d %B %Y')}
         by TEN Capital Network.</p>
    </div>
  </td></tr>

</table>
</td></tr></table>
</body></html>"""


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------
def build_text(result: AnalysisResult, attachment_names: list[str]) -> str:
    profile = result.profile
    final = result.prediction.final
    valuation = (
        profile.number("pre_money_valuation")
        or profile.number("note_cap")
        or profile.number("safe_cap")
    )
    rule = "=" * 66

    lines = [
        result.company_name,
        rule,
        f"{result.sector.label} | {_stage_text(result)}",
        f"Target Raise {format_money(profile.number('target_raise'))}  "
        f"Committed {format_money(profile.number('amount_committed'))}  "
        f"Valuation {format_money(valuation)}",
        "",
        "PROBABILITY OF COMPLETING THE ROUND",
        f"   3 months   {final.p3:5.0f}%",
        f"   6 months   {final.p6:5.0f}%",
        f"  12 months   {final.p12:5.0f}%",
        "",
        f"Fundability score      {result.readiness.total:.0f}/100 ({result.readiness.label})",
        f"Prediction confidence  {result.confidence.score:.0f}/100 ({result.confidence.level})",
        f"Comparable profile     {result.comparable.label}",
        f"TEN Capital fit        {result.ten_capital.fit_score:.0f}/100",
        f"Recommended next step  {result.ten_capital.recommended_next_step}",
        f"{result.prediction.market_condition.disclosure}",
        "",
        "PROBABILITY DRIVERS",
    ]
    drivers = result.prediction.positive_signals[:5]
    lines += [f"  - {d}" for d in drivers] or [
        "  No positive financing signals were evidenced in the deck."
    ]

    lines += ["", "CRITICAL BLOCKERS"]
    lines += [f"  [{b.severity}] {b.title}\n      {b.detail}" for b in result.blockers] or [
        "  No structural blockers identified."
    ]

    lines += ["", "FIVE ACTIONS TO INCREASE FUNDING PROBABILITY"]
    lines += [
        f"  {i}. {a.action} [{a.impact} | {a.timeframe}]"
        for i, a in enumerate(result.actions, start=1)
    ]

    lines += ["", "TEN CAPITAL READINESS FLAGS"]
    lines += [f"  {f.label:32s} {f.value}" for f in result.ten_capital.flags]

    llm = profile.llm_run
    lines += [
        "",
        "EXTRACTION",
        f"  Fields found       {len(profile.known_fields())} of 81",
        f"  Engine             {'Rules + ' + llm['model'] if llm else 'Rules only (deterministic)'}",
    ]
    if llm:
        lines.append(
            f"  From Claude        {llm['fields_accepted']} accepted of "
            f"{llm['fields_proposed']} proposed (${llm['estimated_cost_usd']:.3f})"
        )
        if llm["rejected_unverified"]:
            lines.append(
                f"  Discarded          {', '.join(llm['rejected_unverified'])} (unverified)"
            )

    lines += ["", "ATTACHED"]
    lines += [f"  - {name}" for name in attachment_names] or ["  No files attached."]

    lines += [
        "",
        rule,
        # Kept on one line rather than wrapped: a required disclaimer split
        # across lines is harder to verify and easy to break silently.
        "Financing probability estimates are model-based assessments, "
        "not guarantees of financing outcomes.",
        "",
        DISCLAIMER,
        "",
        "CONFIDENTIAL - FOR INTENDED RECIPIENT ONLY",
        f"Compiled on {result.generated_at.strftime('%d %B %Y')} by TEN Capital Network.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build_analysis_email(
    result: AnalysisResult,
    recipients: list[str],
    attachment_paths: list[Path] | None = None,
) -> EmailMessage:
    """Compose the notification for a completed analysis."""
    attachments: list[Attachment] = []
    for path in attachment_paths or []:
        item = Attachment.from_path(path)
        if item:
            attachments.append(item)

    names = [a.filename for a in attachments]
    return EmailMessage(
        subject=subject_for(result),
        html=build_html(result, names),
        text=build_text(result, names),
        to=list(recipients),
        attachments=attachments,
        tags=[
            ("type", "analysis"),
            # Tag values must be ASCII letters, digits, underscores or dashes.
            ("next_step", _tag_safe(result.ten_capital.recommended_next_step)),
            ("sector", _tag_safe(result.sector.key)),
        ],
    )


def build_failure_email(filename: str, error: str, recipients: list[str]) -> EmailMessage:
    """Notify that an analysis failed, so a silent drop is not the outcome."""
    subject = f"[FAILED] Financing analysis could not be completed - {filename}"
    html_body = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"></head>
<body style="margin:0;padding:24px;background:#EEF1F5;
             font:400 14px/1.55 Helvetica,Arial,sans-serif;color:{INK};">
<table role="presentation" width="600" cellpadding="0" cellspacing="0" border="0"
       style="max-width:600px;background:#fff;border:1px solid {RULE};border-radius:6px;">
<tr><td bgcolor="{CRIT}" style="padding:16px 22px;border-radius:6px 6px 0 0;
        font:700 16px/1.3 Helvetica,Arial,sans-serif;color:#fff;">
  Analysis failed</td></tr>
<tr><td style="padding:20px 22px;">
  <p style="margin:0 0 10px;"><strong>File:</strong> {_e(filename)}</p>
  <p style="margin:0 0 14px;"><strong>Error:</strong></p>
  <pre style="margin:0;padding:12px;background:{PANEL};border:1px solid {RULE};
              border-radius:4px;font:400 12px/1.5 Consolas,Menlo,monospace;
              white-space:pre-wrap;color:{SLATE};">{_e(error)}</pre>
  <p style="margin:14px 0 0;color:{SLATE};font-size:13px;">
    No report was produced for this upload.</p>
</td></tr></table></body></html>"""

    text_body = (
        f"Analysis failed\n{'=' * 40}\n\n"
        f"File:  {filename}\n\nError: {error}\n\n"
        "No report was produced for this upload.\n"
    )
    return EmailMessage(
        subject=subject,
        html=html_body,
        text=text_body,
        to=list(recipients),
        tags=[("type", "failure")],
    )


def _tag_safe(value: str) -> str:
    """Resend tag values allow only ASCII letters, digits, underscore and dash."""
    cleaned = "".join(ch if (ch.isascii() and (ch.isalnum() or ch in "_-")) else "_" for ch in value)
    return cleaned[:60] or "unknown"
