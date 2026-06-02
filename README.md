# Gifts & Hospitality Analyser

A web-based tool for analysing gifts and hospitality registers. Upload your register data, clean and standardise it, and produce ready-to-use compliance reports — no technical knowledge required.

---

## What does it do?

Government departments are required to maintain registers of gifts and hospitality offered to staff. These registers are often held as spreadsheets, and the data within them can be inconsistent: names spelled differently across entries, organisation names abbreviated or formatted in different ways, values recorded in varying formats, and so on.

This tool takes your register data, tidies it up automatically, and produces a set of analytical reports that help you:

- **Understand spend patterns** — total accepted values broken down by directorate, type of hospitality/gift, and staff grade
- **Check compliance** — identify entries where approvals may have been self-granted, where declaration timelines are unusually long, or where values have been entered in a format that cannot be read
- **Spot risks** — flag high-frequency relationships between specific staff and specific external organisations, and identify potential group events where multiple staff accepted hospitality from the same source on the same date
- **Export clean data** — download a standardised, de-duplicated version of your register alongside the analytical reports, both as Excel files

---

## Who is it for?

Any government department or public body that holds a gifts and hospitality register in spreadsheet (CSV) format. The tool is designed to work with registers that have not been centrally standardised — it handles inconsistent formatting, variant spellings, and mixed value formats automatically.

---

## What do you need to run it?

- Python 3.9 or later
- The following Python packages (install via `pip install -r requirements.txt` if a requirements file is provided, or individually):
  - `streamlit`
  - `pandas`
  - `openpyxl`
  - `thefuzz` (optional, but strongly recommended — enables fuzzy name matching)
  - `python-Levenshtein` (optional, speeds up fuzzy matching)

Once installed, start the tool by running:

```
streamlit run gifts_hospitality_analyser.py
```

This will open the dashboard in your web browser.

---

## How to use it — step by step

### Step 1: Upload your register

Click **Browse files** and upload one or more CSV files. If you have separate registers for gifts and hospitality, or registers from multiple years, you can upload them all at once and they will be combined automatically.

Your CSV file should have column headers in the first row. The exact column names do not need to match any particular format — you will map them to the right fields in the next step.

Click **Load and Combine Files** to proceed.

---

### Step 2: Map your columns

The tool needs to know which column in your spreadsheet corresponds to each type of information. Use the dropdown menus to match your column names to the standard fields:

| Field | Description |
|---|---|
| Timestamp / Date Logged | The date the entry was added to the register |
| Recipient Name | The member of staff who received the gift or hospitality |
| Date Received / Event Date | The date the gift was received or the event took place |
| Offered By Organisation | The external organisation or individual making the offer |
| Directorate / Department | The team or business area the recipient belongs to |
| Status | Whether the offer was accepted or declined |
| Reason for Acceptance | The justification given for accepting |
| Gift Value | The monetary value of any gift (can be left as None if your register does not separate gifts and hospitality) |
| Gift Description | A description of the gift |
| Hospitality Value | The monetary value of any hospitality |
| Hospitality Description | A description of the hospitality |
| Grade | The recipient's grade or pay band (optional) |
| Approver Name | The name of the person who approved the acceptance (optional) |

The tool will try to pre-select the most likely column for each field. Check the selections and adjust where needed, then click **Lock Schema & Prepare Normalisations**.

---

### Step 3: Review name standardisation

Registers frequently contain the same organisation or person recorded in multiple ways — "DCMS", "Dept for Culture, Media and Sport", and "Department for Culture Media & Sport" might all refer to the same body. The tool automatically detects these variations and suggests which entries should be treated as the same.

This step shows you the suggested groupings so you can confirm, adjust, or override them:

- **Green tick entries** (shown in a collapsible log) were merged automatically with high confidence and do not need your attention.
- **Cluster review tables** show suggested groupings that the tool was less certain about. For each cluster, you can:
  - Edit the canonical name (the single version that will appear in all reports)
  - Untick any entry that should *not* be merged into the group

The same review process runs for organisation names, staff names, and directorate names.

Once you are satisfied, click **Apply Normalisations & Run Pipeline**.

If you untick any entries from a cluster, you will be asked how to handle those entries — whether to keep them as they are, assign them to a different existing group, or give them a new name.

---

### Step 4: View and download reports

The analysis results are displayed across several tabs:

| Tab | Contents |
|---|---|
| **Directorate & Category** | Total accepted and declined values by directorate and by category (events/entertainment, food & drink, accommodation, other) |
| **Grade Analysis** | Breakdown of values by staff grade, if grade data was provided |
| **Compliance Metrics** | Per-directorate indicators: value entry quality, approver compliance rate, and declaration lag times |
| **Risk Rankings** | Top 10 individuals with value entry errors; top 10 individuals with the longest declaration lags |
| **Group Events & High-Freq** | Entries where multiple staff attended the same event; most frequent recipient–donor pairings |
| **Data Quality Logs** | Rows flagged for data issues (missing values, unrecognised status entries, missing directorate) |
| **Recipient Analysis** | Per-person summary with drill-down to individual entries |
| **Offerer Analysis** | Per-organisation summary with drill-down to individual entries |

Use the **Download** buttons at the bottom of the page to export:
- A **Cleaned Register** — your original data with names standardised and values parsed, ready for publication or further analysis
- **Analytical Reports** — all the summary tables in a single Excel workbook

---

## Saving your work

The tool automatically remembers name standardisation decisions between sessions. When you approve a merge (e.g. "The FA" → "Football Association"), that mapping is saved to a file called `name_mappings.json` in the same folder as the application. The next time you load a register, those decisions are applied automatically.

You can view and edit saved mappings at any time using the **Manage name mappings** panel in the sidebar.

---

## Notes on data format

- Values can be entered in a wide range of formats: `£25`, `25.00`, `£20-£30`, `approx £50 including accommodation`. The tool will extract a numeric figure where possible and flag entries it cannot parse.
- Status fields are recognised in many common forms: Accepted, Approved, Yes, Declined, Rejected, Refused, Returned, and similar.
- Date columns are parsed automatically and accept most common formats.
- If your register uses different column names for gifts and hospitality (e.g. a combined "Value" column), you can map both gift and hospitality fields to the same column.
