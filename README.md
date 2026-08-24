# Startup Financing Probability Predictor

Reads an investor pitch deck (PDF, PPTX or PPT), extracts the company and
financing facts it can evidence, and estimates the probability that the company
completes its stated round within **3, 6 and 12 months**.

Runs as a **web app** (upload a deck in the browser) or a **CLI**. Two outputs,
one run:

* a founder-facing **one-page PDF Financing Probability Report**
* a machine-readable **TEN Capital lead record** plus a full extraction
  evidence map

Claude reads the deck; the scoring model stays deterministic.

---

## Quick start — web app

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                                   # add your ANTHROPIC_API_KEY
uvicorn web.main:app --reload
```

Open <http://127.0.0.1:8000>, drop a deck on the page, and the report plus both
JSON files come back as downloads. Analysis runs as a background job with live
progress, so a long deck never hits a request timeout.

The interface uses the TEN Capital brand system — dark navy surface, the
tri-colour coral/amber/teal accent from the logo mark, Sora for display and
JetBrains Mono for metadata. Those three brand colours also carry meaning:
**teal reads as strong, amber as moderate, coral as weak**, so the probability
tiles, blocker severities and readiness flags all speak one visual language
instead of introducing a second palette. Styles live in
[`web/static/app.css`](web/static/app.css).

---

## Quick start — CLI

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python sample_data/generate_samples.py                 # builds the sample decks

python app.py analyse sample_data/northwind_saas_series_a.pptx
```

Three files land in `output/`:

```
Northwind_Analytics_Financing_Probability_Report.pdf
Northwind_Analytics_Financing_Probability_Data.json
Northwind_Analytics_Extraction_Evidence.json
```

Add `--html` for an HTML rendering of the same report, useful for reviewing
content without opening a PDF viewer.

---

## Commands

| Command | What it does |
| --- | --- |
| `analyse <deck>` | Full run: extract, score, predict, write the PDF and JSON files. `analyze` also works. |
| `review <deck>` | Print every extracted value with its source page, then stop. Optionally emit an editable overrides file. |
| `explain <deck>` | Print the probability bridge for all three horizons and the full readiness-score breakdown. |
| `test-email` | Send a test email to verify the Resend configuration, before a real report depends on it. |

Common options:

```
--set FIELD=VALUE     Override an extracted field (repeatable)
--overrides FILE      Apply a JSON file of overrides
--market ENVIRONMENT  Capital market assumption (default: neutral)
--page LETTER|A4      PDF page size
--out DIR             Output directory (default: ./output)
--html                Also write an HTML report
--email               Email the results (requires RESEND_API_KEY)
--llm                 Read the deck with Claude (verified against source quotes)
--consent-analysis    Record that the company consented to the analysis
--consent-contact     Record that the company consented to TEN Capital contact
```

### Review and correct before scoring

Extraction is never perfect, so corrections are a first-class step rather than
an afterthought:

```bash
python app.py review sample_data/northwind_saas_series_a.pptx --save-overrides fix.json
# edit the "value" entries in fix.json
python app.py analyse sample_data/northwind_saas_series_a.pptx --overrides fix.json
```

Overrides are stored **separately** from what was extracted. The evidence map
keeps both, so an audit can always see what the deck said and what a human
changed:

```json
{
  "field": "target_raise",
  "value": 4000000,
  "status": "EXPLICIT",
  "overridden": true,
  "extracted_value": 9000000,
  "extracted_status": "EXPLICIT",
  "source_page": 9
}
```

---

## How the number is produced

```
deck -> extraction -> derived values -> sector + stage
     -> Financing Readiness Score (0-100)
     -> baseline probability band
     -> signal adjustments -> sector modifier -> market environment
     -> clamp (1-98%) -> monotonicity (3m <= 6m <= 12m)
```

### 1. Financing Readiness Score

Nine weighted categories, all configurable in
[`config/scoring_weights.yaml`](config/scoring_weights.yaml):

| Category | Weight |
| --- | ---: |
| Traction & Revenue | 20 |
| Round Progress | 15 |
| Market Opportunity | 10 |
| Team | 10 |
| Financial Quality | 10 |
| Financing Terms | 10 |
| Product & Defensibility | 10 |
| Investor Fit | 10 |
| Fundraising Readiness | 5 |

Score bands map to labels: **HIGHLY FINANCEABLE** (85+), **FINANCING READY**
(70-84), **NEARLY READY** (55-69), **MATERIAL GAPS** (40-54),
**EARLY / HIGH RISK** (0-39). These describe *fundraising readiness*, not
investment quality.

### 2. Probability

The score selects a band in
[`config/probability_rules.yaml`](config/probability_rules.yaml) and is
interpolated inside it, then adjusted by detected signals. Every horizon is
clamped to 1-98% — the model never reports a financing outcome as impossible or
certain — and `3m <= 6m <= 12m` is enforced.

`explain` prints the whole derivation:

```
Baseline probability: 41%
+3.5% Major commercial partnership
+2.6% Round size appropriate for stage
-5.1% No customers identified
-5.0% No lead investor identified
-4.0% Long regulatory timeline ahead
-1.9% MedTech sector financing dynamics

Final 6 months probability: 26%
```

### 3. Confidence is scored separately

A high readiness score does **not** produce a high confidence score. Confidence
is driven by how much the deck actually disclosed, weighted by evidence
quality, extraction quality and internal consistency. A thin deck describing a
strong company yields a confident-looking readiness score with LOW confidence,
and the report says so.

---

## Design decisions worth knowing

**Nothing is invented.** Every value carries a status: `EXPLICIT` (the deck said
it), `CALCULATED` (arithmetic from explicit values), `ESTIMATED` (a model
inference) or `MISSING`. Missing fields render as *Not Provided* and are never
back-filled. The confidence notes explain what each gap cost.

**Traction is judged by sector norms.** `config/sector_rules.yaml` assigns each
sector a `traction_mode`. A `milestone` sector (Biotech, Life Sciences) is
scored on clinical, regulatory, IP and partnership evidence, so a pre-revenue
biotech is not penalised for lacking SaaS-style revenue. `hybrid` sectors
(MedTech, DeepTech, Hardware, Energy) blend both.

**Targets are not achievements.** Forecast tables and forward-looking language
are excluded from current-state fields, so "$6M ARR within 18 months" or a 2029
projection can never be reported as today's revenue.

**Third-party claims are not the company's.** An advisor's prior exit is not the
founders' prior exit, and "once developed and FDA approved" is not an approval.
Both are checked against surrounding lines rather than a whole-document regex.

**Letter-spaced headings are recovered.** Design decks set headings with wide
tracking, which PDF extraction turns into `RAIS ING $2M`. Cues are matched
against a whitespace-free rendering and mapped back, so the raise amount is
still read from the untouched text.

**Silence is not consent.** Consent fields default to `null`, never `false` and
never `true`. Only the explicit `--consent-*` flags record a decision.

**The default run is deterministic.** Claude extraction is off unless you pass
`--llm` or set `FP_LLM_ENABLED`, even when `ANTHROPIC_API_KEY` is in your
environment — the presence of a key must not silently change a score. When
enabled, every value it returns is verified against a quote in the source
before being accepted; see *How Claude is used*.

**Contradictions are flagged, not corrected.** Committed exceeding target, a
post-money below pre-money, a runway inconsistent with cash and burn, a Series A
with no prior financing — each is surfaced for review and reduces confidence.
The conflicting values are left exactly as extracted.

---

## Project layout

```
app.py                     CLI: analyse / review / explain
config/
  settings.py              config loading, validation, env settings
  scoring_weights.yaml     nine category weights and readiness labels
  sector_rules.yaml        sector rules, stage benchmarks, comparable profiles
  probability_rules.yaml   bands, signal deltas, market conditions, confidence
extractors/
  ingest.py                format dispatch
  pdf_extractor.py         pdfplumber with a pypdf fallback
  ppt_extractor.py         .pptx natively; .ppt via LibreOffice
  field_patterns.py        cue table: where each field lives in deck text
  field_extractor.py       candidate collection and resolution
  normalization.py         money / percent / duration / date parsing
  text_repair.py           letter-spacing tolerance
models/                    field registry, evidence, profile, results
analysis/
  pipeline.py              orchestration
  derive.py                calculated fields
  consistency.py           contradiction checks
  sector_classifier.py     sector resolution
  stage_inference.py       declared vs. model-inferred stage
  readiness_score.py       the nine categories
  probability_model.py     bands, signals, modifiers, bridge
  confidence_model.py      data-completeness confidence
  blockers.py              ranked financing blockers
  recommendations.py       exactly five prioritised actions
  ten_capital_fit.py       flags, fit score, next step
reports/
  pdf_report.py            one-page renderer
  theme.py                 palette and the density ladder
  json_report.py           lead record and evidence map
  html_report.py           HTML companion for review
services/llm.py            Claude extraction + quote verification (Null by default)
services/email.py          Resend transport (Null by default)
services/notify.py         what gets emailed, shared by CLI and web
reports/email_report.py    HTML + plain-text notification
web/
  main.py                  FastAPI app: upload, poll, download
  jobs.py                  background job registry with TTL cleanup
  static/app.css           TEN Capital design system
  templates/               upload form and progress/results page
sample_data/               deck generator and generated samples
docs/                      document structure template and companions
tests/                     344 tests
```

---

## How Claude is used

Claude reads the deck. The model does **not** score it.

That split is deliberate. Language understanding is what an LLM is genuinely
good at, and it finds fields a regex never will — a business model described in
prose, a raise buried in a sentence, a runway implied by a milestone chart. On
the AccuBreath sample it takes extraction from 25 fields to 38. But a score
produced by an LLM cannot be audited or reproduced, so readiness, probability,
confidence, blockers and TEN Capital fit are all computed in Python from the
extracted values. Same inputs, same numbers, every point traceable.

**Every value Claude returns is checked against the source before it is
accepted.** The model must supply a verbatim quote and a page number, and the
quote is located in the extracted text:

| Outcome | What happens |
| --- | --- |
| Quote found | Recorded `EXPLICIT` with the verified page |
| Quote not found, but the value's own text is on the cited page | Recorded `EXPLICIT` — this rescues figures split across table cells, where no contiguous quote exists |
| Neither found, numeric field | **Discarded.** A fabricated figure is worse than a missing one |
| Neither found, text field | Downgraded to `ESTIMATED` at reduced confidence |

Rejections and downgrades are reported in the UI and in
`evidence.llm_extraction`, along with token counts and the estimated API cost,
so a run is never a black box.

Rule-based extraction runs first and is the floor: a Claude proposal only
replaces it when the proposal is both verified and more confident.

```bash
python app.py analyse deck.pdf --llm      # CLI opt-in
FP_LLM_ENABLED=on                          # or enable by default
```

Without a key, or with the flag off, the app runs on rules alone — no network
call, no cost, fully reproducible. Typical Claude cost is **$0.07–$0.18 per
deck** on `claude-opus-5` at `FP_LLM_EFFORT=high`; `medium` roughly halves both
latency and spend. The stable system prompt is cached, so repeat analyses cost
less than the raw token count suggests.

---

## Email notifications

Every completed analysis is emailed to `Info@tencapital.group` with the PDF
report and the lead record attached. Failures are emailed too, so an upload
never disappears silently.

```bash
python app.py test-email                      # verify the configuration first
python app.py analyse deck.pdf --email        # CLI opt-in
FP_EMAIL_ENABLED=on                            # web app sends automatically
```

**Delivery is best-effort and never fails an analysis.** The report is the
product; the email is a convenience. Every send is wrapped, returns a result
rather than raising, and a failure shows up as a note on the results page with
the files still downloadable.

### One thing to do before this reaches the inbox

Resend's default sender (`onboarding@resend.dev`) **only delivers to the Resend
account owner's own address**. Sending to `Info@tencapital.group` from it is
rejected:

```
HTTP 403: You can only send testing emails to your own email address
(hallmartin@tencapital.group). To send emails to other recipients, please
verify a domain at resend.com/domains, and change the `from` address.
```

To fix it: verify `tencapital.group` at [resend.com/domains](https://resend.com/domains)
(add the DKIM/SPF records it gives you), then set a sender on that domain:

```bash
FP_EMAIL_FROM="TEN Capital Reports <reports@tencapital.group>"
```

`test-email` warns when the sandbox sender is still in place, so this is
visible before a real report depends on it.

| Variable | Purpose |
| --- | --- |
| `RESEND_API_KEY` | Required. Never commit it |
| `FP_EMAIL_ENABLED` | `off` (default), `auto` (when a key is present), or `on` |
| `FP_EMAIL_TO` | Recipients, comma-separated. Defaults to `Info@tencapital.group` |
| `FP_EMAIL_FROM` | Sender on your verified domain |
| `FP_EMAIL_REPLY_TO` | Optional reply-to |
| `FP_EMAIL_ATTACH` | Artifacts to attach: `pdf`, `data`, `evidence`, `html` (default `pdf,data`) |
| `FP_EMAIL_ON_FAILURE` | Email when an analysis fails (default `on`) |

The email carries the headline probabilities, readiness and confidence scores,
drivers, blockers, the five actions, all ten TEN Capital flags and the
extraction summary — plus a plain-text alternative, because its absence is a
deliberate spam signal and text-only clients render nothing without it.

**Note that this sends company data to a third party.** Deck contents reach
Resend's infrastructure, and attachments carry the full report. That is the
point of the feature, but it is worth being explicit about before pointing it
at a shared inbox.

---

## Deploying to Railway

```bash
railway init
railway variables --set ANTHROPIC_API_KEY=sk-ant-... \
                  --set FP_LLM_ENABLED=on \
                  --set RESEND_API_KEY=re_... \
                  --set FP_EMAIL_ENABLED=on
railway up
```

`railway.json` sets the start command and points the health check at
`/healthz`. Nixpacks reads `requirements.txt` and `.python-version`.

**The app has no authentication.** Anyone who can reach the URL can upload a
deck, and every analysis spends Claude and Resend credits on your account. If
that matters, put it behind a Railway private service, a VPN, or an
authenticating proxy — the application itself will not stop anyone.

| Variable | Purpose |
| --- | --- |
| `ANTHROPIC_API_KEY` | Required for Claude extraction. Never commit it |
| `FP_LLM_ENABLED` | `on` to enable extraction, `off` for rules only |
| `FP_LLM_MODEL` / `FP_LLM_EFFORT` | Model and effort (default `claude-opus-5`, `high`) |
| `FP_MAX_UPLOAD_MB` | Upload ceiling (default 40) |
| `FP_MAX_CONCURRENT` | Simultaneous analyses (default 2) |
| `FP_JOB_DIR` | Where uploads live before expiry (default: a temp dir) |
| `RESEND_API_KEY` | Required for email notifications |
| `FP_EMAIL_ENABLED` | `on` to email results after every analysis |
| `FP_EMAIL_FROM` | Sender on your verified domain (see *Email notifications*) |

Two deployment constraints worth knowing:

* **Jobs are in-memory and files are on local disk**, which suits one Railway
  instance. `numReplicas` is pinned to 1; running more would need shared
  storage, because a poll could land on a replica that does not have the job.
* **Uploaded decks are confidential.** They are written to a per-job directory
  and deleted when the job expires after 30 minutes, along with the generated
  reports. Nothing is written outside that directory.

Legacy `.ppt` needs LibreOffice, which is commented out in `nixpacks.toml`
because it adds ~500MB. Without it the app returns a clear "re-save as .pptx or
.pdf" error rather than failing silently.

---

## Document structure template

[`docs/TEMPLATE.md`](docs/TEMPLATE.md) is the authoritative specification of
the document this app produces: every section, field, format, value domain and
piece of fixed text, with no company data. Alongside it:

| File | What it is |
| --- | --- |
| [`docs/report_template.pdf`](docs/report_template.pdf) | The one-page layout with placeholder tokens |
| `docs/report_template.png` | Preview image |
| `docs/report_template.html` | HTML companion layout |
| [`docs/lead_record_template.json`](docs/lead_record_template.json) | Lead-record structure, values nulled |
| [`docs/evidence_map_template.json`](docs/evidence_map_template.json) | Evidence map listing all 81 registry fields |

```bash
python docs/generate_template.py
```

The PDF template is rendered by the **production renderer** with placeholder
tokens (`XX%`, `$X.XM`, `[COMPANY NAME]`) rather than by a separate mock, so it
cannot drift from real output. Regenerate it whenever the layout or the field
registry changes; the tests assert it stays one page, keeps every fixed
heading, flag label and disclaimer, and contains no company data.

---

## Keeping the report on one page

The renderer builds the report at the roomiest layout, measures it, and steps
down a density ladder (`comfortable → standard → tight → compact → dense`)
until it fits. Long text is truncated on word boundaries, the company name is
auto-sized to its band, and a final `KeepInFrame` scales the block if an
unusually long input defeats every rung. Page count is asserted in the tests
for every sample deck, both page sizes, an 80-character company name and a
deck with no extractable data at all.

To restyle the report, edit [`reports/theme.py`](reports/theme.py); the
analysis is untouched by layout changes.

---

## Configuration

All model tuning is in YAML — no weight, threshold or band is hard-coded.
Configuration is validated at load time, so a file that no longer totals 100
points fails immediately rather than silently mis-scoring:

```
ConfigError: scoring_weights.yaml category weights sum to 130, expected 100.
```

Signal *detection* stays in Python (`SIGNAL_DETECTORS` in
`analysis/probability_model.py`) while *magnitude* lives in YAML. A config edit
can retune or disable a rule; it cannot invent a signal the data does not
support.

Environment settings are documented in [`.env.example`](.env.example).

---

## Tests

```bash
python -m pytest -q                                    # 344 tests
python -m pytest --cov=. --cov-report=term-missing     # with coverage
```

The suite covers the acceptance criteria directly: ingestion of each format and
a useful error for unsupported ones; extraction with source pages retained and
missing fields staying missing; probabilities bounded, ordered and moving in the
right direction for positive and negative signals; a single-page PDF under
layout stress; and JSON that matches the schema, carries evidence quality and
readiness flags, and never converts unknown consent into positive consent.

---

## Limitations

* **No live market data.** The capital market environment is an assumption and
  is labelled as one in every output. The modifier table is ready for a future
  data source; `MarketCondition.is_verified` stays `False` until one exists.
* **No comparable-transaction database.** The comparable financing profile is a
  qualitative archetype with a stated rationale, never a claim that a specific
  company raised a specific amount.
* **No OCR.** Image-only slides are detected, flagged as
  `IMAGE-BASED CONTENT - EXTRACTION CONFIDENCE REDUCED`, and lower the
  confidence score. They are not transcribed.
* **Currency is recorded, not converted.** A EUR figure stays a EUR figure;
  benchmark comparisons assume USD-equivalent amounts.
* **Benchmarks are defaults.** The stage ranges in `sector_rules.yaml` are
  starting points and should be tuned to the deal flow you actually see.
* **Claude extraction is verified, not infallible.** The quote check proves a
  value appears in the deck; it cannot prove the deck is accurate, nor that the
  right one of two similar figures was chosen. The `review` command and the
  evidence map exist so a human can check the values that matter.
* **The web app holds jobs in memory on a single instance.** Scaling past one
  replica needs shared job storage.
* **Email requires a verified sending domain.** Until `tencapital.group` is
  verified in Resend, notifications only reach the Resend account owner's own
  address. See *Email notifications*.

---

## Disclaimer

This report estimates financing probability based on company-supplied
information, model assumptions, and available market context. It does not
predict investor behavior with certainty and does not constitute investment,
legal, tax, or securities advice.
