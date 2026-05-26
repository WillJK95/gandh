import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import json
from datetime import datetime
from pathlib import Path

try:
    from thefuzz import process, fuzz
except ImportError:
    process = None
    fuzz = None

st.set_page_config(page_title="Hospitality & Gifts Analytics Engine", layout="wide")

# =============================================================================
# PERSISTENT NAME MAPPING DICTIONARY
# =============================================================================

MAPPING_FILE = Path(__file__).resolve().parent / "name_mappings.json"

SEED_ORG_MAP = {
    'FA': 'Football Association',
    'The FA': 'Football Association',
    'The Football Association': 'Football Association',
    'The English Football Association': 'Football Association',
    'UEFA/FA': 'Football Association',
    'The English Fottball Association': 'Football Association',
    'English FA': 'Football Association',
    'The FA (Football Association)': 'Football Association',
    'pwc': 'PwC',
    'National Lottery': 'National Lottery',
    'The National Lottery': 'National Lottery',
    'National Lottery Heritage Fund': 'National Lottery Heritage Fund',
    'The EFL': 'English Football League',
    'BFI': 'British Film Institute',
    'RFU': 'Rugby Football Union',
    'The National Theatre': 'National Theatre',
}


def load_name_mappings():
    if MAPPING_FILE.exists():
        try:
            with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for key in ('organizations', 'recipients', 'approvers'):
                data.setdefault(key, {})
            return data
        except (json.JSONDecodeError, OSError):
            pass
    data = {'organizations': dict(SEED_ORG_MAP), 'recipients': {}, 'approvers': {}}
    save_name_mappings(data)
    return data


def save_name_mappings(data):
    try:
        with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
        return True
    except OSError as e:
        st.warning(f"Could not save name mappings to {MAPPING_FILE}: {e}")
        return False


# =============================================================================
# CORE PARSING / CATEGORISATION
# =============================================================================

def parse_value(value_str):
    if pd.isna(value_str):
        return np.nan
    value_str = str(value_str).lower().strip()
    if not value_str:
        return np.nan
    if any(term in value_str for term in ['n/a', 'unknown', 'nil', 'unclear', 'unsure', 'free', 'no gift', 'not known']):
        return np.nan
    if '65 gbp' in value_str:
        return 65.0
    if '47 or 35' in value_str:
        return 47.0

    currency_numbers = [float(num.replace(',', '')) for num in re.findall(r'(?:£|gbp)\s*(\d[\d,.]*)', value_str)]
    if not currency_numbers:
        currency_numbers = [float(num.replace(',', '')) for num in re.findall(r'(\d[\d,.]*)\s*(?:£|gbp)', value_str)]

    all_numbers = [float(num.replace(',', '')) for num in re.findall(r'(\d[\d,.]*)', value_str)]
    numbers = currency_numbers if currency_numbers else all_numbers

    if len(numbers) > 1:
        filtered_numbers = [n for n in numbers if not (2000 <= n <= 2035 and n == int(n))]
        if filtered_numbers:
            numbers = filtered_numbers

    if not numbers:
        return np.nan
    if any(term in value_str for term in ['-', 'to']) and len(numbers) > 1:
        return np.mean(numbers)
    if ('£' in value_str and value_str.count('£') > 1) or (' and ' in value_str) or ('+' in value_str) or ('accomm' in value_str):
        return sum(numbers)

    return max(numbers) if numbers else np.nan


def categorize_hospitality(detail_str):
    detail_str = str(detail_str).lower()
    if any(w in detail_str for w in ['ticket', 'match', 'stadium', 'concert', 'theatre', 'show', 'opera', 'event', 'festival', 'sport']):
        return 'Event/Entertainment'
    if any(w in detail_str for w in ['dinner', 'lunch', 'breakfast', 'reception', 'meal', 'wine', 'gala', 'banquet', 'hospitality', 'drinks']):
        return 'Food & Drink'
    if any(w in detail_str for w in ['hotel', 'stay', 'overnight', 'accommodation', 'flight', 'travel', 'train']):
        return 'Accommodation'
    return 'Other / Gift Item'


EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')


def is_email(s):
    return isinstance(s, str) and bool(EMAIL_RE.match(s.strip()))


def clean_person_name(name_val):
    # Persistent person-name map keys are pre-cleaning raw entries.
    if pd.isna(name_val) or not isinstance(name_val, str):
        return name_val
    s = name_val.strip()
    if not s:
        return name_val

    if is_email(s):
        return s.lower()

    s = s.replace(' & ', ' and ').replace('-', ' ')
    s = re.sub(r'(?<=\w)[.,;:]+(?=\s|/|$)', '', s)
    s = re.sub(r'\s*/\s*', ' / ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.title().strip()


def to_comparable_token(s):
    """Convert a name or email into a comparable token so 'Jane Smith',
    'jane.smith@dept.gov.uk', and 'J.Smith' all collapse to the same form
    for the self-approval equality check."""
    if pd.isna(s) or not isinstance(s, str):
        return ''
    s = s.strip().lower()
    if is_email(s):
        local = s.split('@')[0]
        local = re.sub(r'[._\-]+', ' ', local)
        local = re.sub(r'\d+$', '', local)
        return local.strip()
    return re.sub(r'[^a-z\s]', ' ', s).strip()


# =============================================================================
# ORG-NAME NORMALISATION HELPERS
# =============================================================================

STOP_WORDS = {'the', 'ltd', 'limited', 'llp', 'inc', 'plc', 'group', 'co', 'company', 'corporation', 'corp'}

ABBREV_PAIRS = [
    (r'\bltd\b', 'limited'),
    (r'\b&\b', 'and'),
    (r'\bco\b', 'company'),
]

INTERMEDIARY_PATTERNS = [
    re.compile(r'^(.*?)\s+(?:courtesy of|on behalf of|via|guest of|hosted by)\s+(.+)$', re.IGNORECASE),
]

PAREN_RE = re.compile(r'\(([^)]*)\)')
HUMAN_INTERVENTION_RE = re.compile(r'courtesy of|on behalf of|\bvia\b|guest of|\bhosted by\b|&|\band\b|,', re.IGNORECASE)


def expand_abbrevs(s):
    if not s:
        return s
    out = s
    for pat, repl in ABBREV_PAIRS:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    return out


def extract_org_candidate(raw):
    """('X courtesy of Y', '(the Z)') -> ('Y', 'the Z'). Falls through to identity."""
    if not isinstance(raw, str) or not raw.strip():
        return ('', '')
    s = raw.strip()
    note_match = PAREN_RE.search(s)
    note = note_match.group(1).strip() if note_match else ''
    s_no_paren = PAREN_RE.sub('', s).strip()
    for pat in INTERMEDIARY_PATTERNS:
        m = pat.match(s_no_paren)
        if m:
            return (m.group(2).strip(), note)
    return (s_no_paren, note)


def light_normalise(s):
    if not isinstance(s, str):
        return ''
    out = expand_abbrevs(s.lower())
    out = re.sub(r'[^\w\s]', ' ', out)
    tokens = [t for t in out.split() if t and t not in STOP_WORDS]
    return ' '.join(tokens)


def first_token_block_key(s):
    norm = light_normalise(s)
    if not norm:
        return ''
    first = norm.split()[0]
    return first[:3] if len(first) >= 3 else first


def _len_ratio(a, b):
    la, lb = max(len(a or ''), 1), max(len(b or ''), 1)
    return min(la, lb) / max(la, lb)


def _classify_tier(score, len_ratio, cand_info, matched_canon, cluster_size):
    """Return ('auto', reason) | ('review', flags_list) | ('leave', None)."""
    human = cand_info['has_human_intervention']
    nrm = cand_info['normalised']
    matched_nrm = light_normalise(matched_canon)
    is_substring = bool(nrm and matched_nrm and (nrm in matched_nrm or matched_nrm in nrm))

    if score < 75:
        return ('leave', None)

    if not human:
        if score >= 92 and 0.7 <= len_ratio <= 1.3:
            return ('auto', 'fuzzy_high_score_balanced')
        if score >= 88 and is_substring and len_ratio >= 0.7:
            return ('auto', 'substring_high_score')

    flags = []
    if human:
        flags.append('human_intervention')
    if len_ratio < 0.5:
        flags.append('length_asymmetry')
    if 75 <= score <= 91:
        flags.append('mid_confidence')
    if cluster_size >= 5 and 70 <= score <= 84:
        flags.append('singleton_vs_cluster')
    if not flags:
        flags.append('flagged')
    return ('review', flags)


def compute_org_normalisations(raw_orgs, persistent_org_map):
    """Tiered org normalisation.

    Returns:
        auto_map: dict raw -> canonical (silent merges + manual map hits)
        auto_log: list of {Raw, Canonical, Score, Reason} for transparency
        review_records: list of {Raw, Proposed_Canonical, Score, Length_Ratio, Flags, Parenthetical}
    """
    raw_orgs = [o for o in raw_orgs if isinstance(o, str) and o.strip()]
    unique_raw = sorted(set(raw_orgs))

    auto_map = {}
    auto_log = []
    review_records = []
    review_seen = set()

    def add_review(raw, canon, score, len_ratio, flags, parenthetical):
        if raw in review_seen:
            return
        review_seen.add(raw)
        review_records.append({
            'Raw': raw,
            'Proposed_Canonical': canon,
            'Score': score,
            'Length_Ratio': round(len_ratio, 2),
            'Flags': ', '.join(flags) if isinstance(flags, (list, tuple, set)) else flags,
            'Parenthetical': parenthetical,
        })

    candidates = {}
    for raw in unique_raw:
        cand, note = extract_org_candidate(raw)
        candidates[raw] = {
            'candidate': cand,
            'parenthetical': note,
            'normalised': light_normalise(cand),
            'has_human_intervention': bool(HUMAN_INTERVENTION_RE.search(raw)),
        }

    # Persistent manual map (raw, candidate, and normalised forms)
    norm_to_persistent = {light_normalise(src): dst for src, dst in persistent_org_map.items()}

    pending = []
    for raw in unique_raw:
        c = candidates[raw]
        if raw in persistent_org_map:
            auto_map[raw] = persistent_org_map[raw]
            auto_log.append({'Raw': raw, 'Canonical': persistent_org_map[raw], 'Score': 100, 'Reason': 'manual_map'})
            continue
        if c['candidate'] and c['candidate'] in persistent_org_map:
            auto_map[raw] = persistent_org_map[c['candidate']]
            auto_log.append({'Raw': raw, 'Canonical': persistent_org_map[c['candidate']], 'Score': 100, 'Reason': 'manual_map (after extract)'})
            continue
        if c['normalised'] and c['normalised'] in norm_to_persistent:
            auto_map[raw] = norm_to_persistent[c['normalised']]
            auto_log.append({'Raw': raw, 'Canonical': norm_to_persistent[c['normalised']], 'Score': 100, 'Reason': 'manual_map (normalised)'})
            continue
        pending.append(raw)

    # Group pending by light-normalised candidate
    norm_groups = {}
    for raw in pending:
        nrm = candidates[raw]['normalised']
        if not nrm:
            continue
        norm_groups.setdefault(nrm, []).append(raw)

    # Pick one representative canonical per group (prefer no-human-intervention, then longest)
    group_canon = {}
    for nrm, members in norm_groups.items():
        members_sorted = sorted(
            members,
            key=lambda r: (candidates[r]['has_human_intervention'], -len(candidates[r]['candidate'])),
        )
        group_canon[nrm] = candidates[members_sorted[0]]['candidate']

    # Seed canonical pool with persistent map destinations so fuzzy can land on them
    canonical_pool = {}  # normalised -> display canonical
    for dst in set(persistent_org_map.values()):
        canonical_pool[light_normalise(dst)] = dst
    for nrm, canon in group_canon.items():
        canonical_pool.setdefault(nrm, canon)

    # Auto-merge exact-normalised duplicates within group; surface human-intervention members for review
    for nrm, members in norm_groups.items():
        canon = canonical_pool[nrm]
        for raw in members:
            if raw == canon:
                continue
            if candidates[raw]['has_human_intervention']:
                add_review(raw, canon, 100, 1.0, ['human_intervention'], candidates[raw]['parenthetical'])
                continue
            auto_map[raw] = canon
            auto_log.append({'Raw': raw, 'Canonical': canon, 'Score': 100, 'Reason': 'exact after light_normalise'})

    # Pool-level pairwise fuzzy: compare distinct canonicals to each other within block
    persistent_dest_nrms = {light_normalise(dst) for dst in set(persistent_org_map.values())}
    if process is not None and len(canonical_pool) > 1:
        pool_by_block = {}
        for nrm, canon in canonical_pool.items():
            pool_by_block.setdefault(first_token_block_key(canon), []).append(nrm)

        # Union-find of merges
        parent = {nrm: nrm for nrm in canonical_pool}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            # The longer canonical wins as the cluster root
            if len(canonical_pool[ra]) >= len(canonical_pool[rb]):
                parent[rb] = ra
            else:
                parent[ra] = rb

        # Compare canonicals within each block
        for block, nrms in pool_by_block.items():
            if len(nrms) < 2:
                continue
            # Sort by length desc so the longest canonical anchors comparisons
            nrms_sorted = sorted(nrms, key=lambda n: -len(canonical_pool[n]))
            for i, a in enumerate(nrms_sorted):
                for b in nrms_sorted[i + 1:]:
                    canon_a, canon_b = canonical_pool[a], canonical_pool[b]
                    score = fuzz.token_set_ratio(a, b)
                    len_ratio = _len_ratio(canon_a, canon_b)
                    if score < 75:
                        continue
                    if a in persistent_dest_nrms and b in persistent_dest_nrms:
                        # Both sides are user-asserted distinct destinations; don't merge.
                        continue
                    # The "raws" merged into the b-group inherit any human-intervention flags
                    b_members = norm_groups.get(b, [])
                    b_has_human = any(candidates[r]['has_human_intervention'] for r in b_members)
                    is_substring = (a in b or b in a)

                    if not b_has_human and score >= 92 and 0.7 <= len_ratio <= 1.3:
                        union(a, b)
                        for raw in b_members:
                            if candidates[raw]['has_human_intervention'] or raw in auto_map:
                                continue
                            auto_map[raw] = canon_a
                            auto_log.append({'Raw': raw, 'Canonical': canon_a, 'Score': score, 'Reason': 'pool-merge balanced'})
                        # Map the canonical-as-raw itself if it appears in the input
                        if canon_b in unique_raw and canon_b != canon_a and canon_b not in auto_map:
                            auto_map[canon_b] = canon_a
                            auto_log.append({'Raw': canon_b, 'Canonical': canon_a, 'Score': score, 'Reason': 'pool-merge balanced'})
                        continue
                    if not b_has_human and score >= 88 and is_substring and len_ratio >= 0.7:
                        union(a, b)
                        for raw in b_members:
                            if candidates[raw]['has_human_intervention'] or raw in auto_map:
                                continue
                            auto_map[raw] = canon_a
                            auto_log.append({'Raw': raw, 'Canonical': canon_a, 'Score': score, 'Reason': 'pool-merge substring'})
                        if canon_b in unique_raw and canon_b != canon_a and canon_b not in auto_map:
                            auto_map[canon_b] = canon_a
                            auto_log.append({'Raw': canon_b, 'Canonical': canon_a, 'Score': score, 'Reason': 'pool-merge substring'})
                        continue

                    # Otherwise classify per tier rules and surface for review
                    cluster_size = len(b_members) or 1
                    # Use the b-group's representative member info for human-intervention flag
                    rep_info = {
                        'candidate': canon_b,
                        'normalised': b,
                        'has_human_intervention': b_has_human,
                        'parenthetical': '',
                    }
                    tier, info = _classify_tier(score, len_ratio, rep_info, canon_a, cluster_size)
                    if tier == 'review':
                        # Surface each raw in the b-group as a review record pointing at canon_a
                        for raw in b_members:
                            add_review(raw, canon_a, score, len_ratio, info, candidates[raw]['parenthetical'])
                        # Also surface canon_b itself if it appears as a raw
                        if canon_b in unique_raw and canon_b not in review_seen and canon_b not in auto_map:
                            add_review(canon_b, canon_a, score, len_ratio, info, '')

    return auto_map, auto_log, review_records


def compute_person_normalisations(values, persistent_map, label):
    records = []
    for raw in values:
        if not isinstance(raw, str) or not raw.strip():
            continue
        if raw in persistent_map:
            proposed = persistent_map[raw]
        else:
            proposed = clean_person_name(raw)
        if raw.strip() != str(proposed).strip():
            records.append({'Type': label, 'Raw Entry': raw, 'Normalized Entry': proposed})
    return records


# =============================================================================
# COMPLIANCE METRICS
# =============================================================================

def calculate_single_group_compliance(group):
    if group.empty:
        return pd.Series(dtype='float64')
    total_entries = len(group)

    parsable_value_count = group['value_parsed_gbp'].notna().sum()
    value_errors_df = group[group['value_parsed_gbp'].isna()]
    val_offender_name, val_offender_share = ('N/A', 0)
    if not value_errors_df.empty:
        top_offender = value_errors_df['recipient_name_clean'].value_counts()
        if not top_offender.empty:
            val_offender_name = top_offender.index[0]
            val_offender_share = (top_offender.iloc[0] / len(value_errors_df)) * 100

    approver_compliance_pct = np.nan
    app_offender_name = 'Data Missing'
    app_offender_share = 0

    if 'approver_name' in group.columns and 'approver_name_clean' in group.columns:
        accepted_group = group[group['status'].str.lower() == 'accepted']
        self_approved_flags = ['self approved', 'self-approved', 'on my own authority']

        recip_token = accepted_group['recipient_name'].apply(to_comparable_token)
        appr_token = accepted_group['approver_name'].apply(to_comparable_token)
        is_name_match = (recip_token == appr_token) & (recip_token != '')
        is_flag_match = accepted_group['approver_name'].str.lower().isin(self_approved_flags)
        self_approved_df = accepted_group[is_name_match | is_flag_match]

        total_self_approvals = len(self_approved_df)
        non_self_approved_count = total_entries - total_self_approvals
        approver_compliance_pct = (non_self_approved_count / total_entries) * 100 if total_entries > 0 else 0

        if total_self_approvals > 0:
            top_app = self_approved_df['recipient_name_clean'].value_counts()
            if not top_app.empty:
                app_offender_name = top_app.index[0]
                app_offender_share = (top_app.iloc[0] / total_self_approvals) * 100

    lag_offender_name = 'N/A'
    median_lag = group['declaration_lag_days'].median()
    if group['declaration_lag_days'].notna().any():
        median_lags = group.groupby('recipient_name_clean')['declaration_lag_days'].median()
        if not median_lags.empty:
            lag_offender_name = median_lags.idxmax()

    return pd.Series({
        'Total_Entries': total_entries,
        'Value_Compliance_%': (parsable_value_count / total_entries) * 100 if total_entries > 0 else 0,
        'Approver_Compliance_%': approver_compliance_pct,
        'Avg_Lag_Days': group['declaration_lag_days'].mean(),
        'Median_Lag_Days': median_lag,
        'Value_Worst_Offender': val_offender_name,
        'Value_Offender_%_Share': val_offender_share,
        'Approver_Worst_Offender': app_offender_name,
        'Approver_Offender_%_Share': app_offender_share,
        'Lag_Worst_Offender_by_Median': lag_offender_name,
    })


def generate_compliance_metrics(df):
    report_dfs = {}
    time_grouping = ['year_declared']
    accepted_df = df[df['status'].str.lower() == 'accepted'].copy()

    accepted_dir = accepted_df.groupby(['directorate_clean'] + time_grouping, dropna=False)['value_parsed_gbp'].agg(['sum', 'mean', 'median', 'max'])
    accepted_dir.columns = ['Accepted_' + col for col in accepted_dir.columns]

    declined_dir = df[df['status'].str.lower() == 'declined'].groupby(['directorate_clean'] + time_grouping, dropna=False)['value_parsed_gbp'].agg(['sum', 'mean', 'median', 'max'])
    declined_dir.columns = ['Declined_' + col for col in declined_dir.columns]

    val_dir = pd.concat([accepted_dir, declined_dir], axis=1, sort=True).fillna(0)

    if not accepted_df.empty:
        group_cols = ['directorate_clean'] + time_grouping
        recipient_values = accepted_df.groupby(group_cols + ['recipient_name_clean'])['value_parsed_gbp'].sum()

        if not recipient_values.empty:
            top_recipient_values = recipient_values.groupby(level=list(range(len(group_cols)))).max()
            top_recipient_values.name = 'Highest_Individual_Total_GBP'

            top_recipient_idx = recipient_values.groupby(level=list(range(len(group_cols)))).idxmax()
            top_recipient_names = top_recipient_idx.apply(lambda x: x[-1] if isinstance(x, tuple) else 'N/A')
            top_recipient_names.name = 'Highest_Individual_Name'

            val_dir = val_dir.join(top_recipient_names).join(top_recipient_values)
            val_dir['Highest_Individual_Share_%'] = np.where(
                val_dir['Accepted_sum'] > 0,
                (val_dir['Highest_Individual_Total_GBP'] / val_dir['Accepted_sum']) * 100,
                0
            )

    for col in ['Highest_Individual_Name', 'Highest_Individual_Total_GBP', 'Highest_Individual_Share_%']:
        if col in val_dir.columns:
            val_dir[col] = val_dir[col].fillna('N/A' if 'Name' in col else 0)

    report_dfs['Value_by_Directorate_YR'] = val_dir.round(2)

    accepted_cat = accepted_df.groupby(['hospitality_category'] + time_grouping, dropna=False)['value_parsed_gbp'].agg(['count', 'sum', 'mean'])
    accepted_cat.columns = ['Accepted_Count', 'Accepted_Total_GBP', 'Accepted_Avg_GBP']
    report_dfs['Value_by_Category'] = accepted_cat.round(2)

    if 'record_type' in df.columns:
        rt = accepted_df.groupby(['record_type'] + time_grouping, dropna=False)['value_parsed_gbp'].agg(['count', 'sum', 'mean'])
        rt.columns = ['Accepted_Count', 'Accepted_Total_GBP', 'Accepted_Avg_GBP']
        report_dfs['Value_by_RecordType'] = rt.round(2)

    if 'grade' in df.columns:
        accepted_grade = accepted_df.groupby(['grade'] + time_grouping)['value_parsed_gbp'].agg(['sum', 'mean', 'median', 'max'])
        accepted_grade.columns = ['Accepted_' + col for col in accepted_grade.columns]
        declined_grade = df[df['status'].str.lower() == 'declined'].groupby(['grade'] + time_grouping)['value_parsed_gbp'].agg(['sum', 'mean', 'median', 'max'])
        declined_grade.columns = ['Declined_' + col for col in declined_grade.columns]
        val_grade = pd.concat([accepted_grade, declined_grade], axis=1, sort=True).fillna(0)
        report_dfs['Value_by_Grade_YR'] = val_grade.round(2)

    dir_comp = df.groupby(['directorate_clean'] + time_grouping, dropna=False).apply(calculate_single_group_compliance).round(2).reset_index()
    report_dfs['Compliance_by_Directorate'] = dir_comp

    rankings = {}
    non_compliant_val = df[df['value_parsed_gbp'].isna()]
    if not non_compliant_val.empty:
        rankings['top_val_err'] = non_compliant_val['recipient_name_clean'].value_counts().nlargest(10).to_frame('Entries_With_Value_Errors').reset_index()

    rankings['top_lag'] = df.groupby('recipient_name_clean')['declaration_lag_days'].max().nlargest(10).to_frame('Max_Lag_Days').reset_index()
    report_dfs['Individual_Compliance_Data'] = rankings

    group_cols = ['timestamp', 'date_received', 'recipient_name_clean', 'directorate_clean', 'offered_by_org_clean', 'details', 'value_parsed_gbp']
    potential_groups = accepted_df[accepted_df.duplicated(subset=['date_received', 'offered_by_org_clean'], keep=False)].sort_values(by=['offered_by_org_clean', 'date_received'])
    report_dfs['Potential_Group_Events'] = potential_groups[[c for c in group_cols if c in potential_groups.columns]]

    status_grouped = df.groupby(['recipient_name_clean', 'offered_by_org_clean', 'status']).size().unstack(fill_value=0)
    status_grouped['Total'] = status_grouped.sum(axis=1)
    report_dfs['High_Freq_Pairings'] = status_grouped.sort_values('Total', ascending=False).head(50)

    issue_mask = df['timestamp'].isna() | df['value_parsed_gbp'].isna() | df['directorate_clean'].isna()
    if issue_mask.any():
        log_df = df[issue_mask].copy()
        log_df['Issue_Reason'] = ''
        log_df.loc[log_df['timestamp'].isna(), 'Issue_Reason'] += 'Invalid Timestamp; '
        log_df.loc[log_df['value_parsed_gbp'].isna(), 'Issue_Reason'] += 'Unparsable Value; '
        log_df.loc[log_df['directorate_clean'].isna(), 'Issue_Reason'] += 'Missing Directorate; '
        log_cols = ['Issue_Reason', 'timestamp', 'recipient_name', 'value_raw', 'directorate', 'details', 'record_type']
        report_dfs['Data_Quality_Log'] = log_df[[c for c in log_cols if c in log_df.columns]]

    return report_dfs


def run_analysis_pipeline():
    """Apply normalisations from session state, derive analytical columns, persist
    new mappings, and produce the report dataframes. Reachable both from the
    no-orphan Apply button and from the orphan-review Confirm button."""
    df_processed = st.session_state.mapped_working_df.copy()
    persistent = st.session_state.name_mappings

    # Build final org_map: auto-applied + cluster acceptances + orphan overrides
    org_map = dict(st.session_state.get('org_auto_map', {}))
    new_persistent_orgs = {}
    cluster_count = st.session_state.get('cluster_count', 0)
    for i in range(cluster_count):
        canonical = st.session_state.get(f"cluster_canon_{i}", '').strip()
        resolved = st.session_state.get(f"cluster_resolved_{i}")
        if resolved is None or not canonical:
            continue
        for _, row in resolved.iterrows():
            if not row.get('Include', True):
                continue
            raw = row['Raw']
            org_map[raw] = canonical
            new_persistent_orgs[raw] = canonical

    # Orphan overrides: only persist if mapped onto an existing canonical (not
    # singleton, not freshly-coined). Existing canonicals = anything already in
    # org_map.values() OR in the persistent organisations map.
    existing_canonicals = set(org_map.values()) | set(persistent.get('organizations', {}).values())
    for raw, target in st.session_state.get('orphan_overrides', {}).items():
        if not target or target == raw:
            continue
        org_map[raw] = target
        if target in existing_canonicals:
            new_persistent_orgs[raw] = target

    # Person maps from edited table
    recipient_map = {}
    approver_map = {}
    person_edited = st.session_state.get('person_review_df_edited', pd.DataFrame())
    if not person_edited.empty:
        for _, row in person_edited.iterrows():
            if row['Type'] == 'Recipient Name':
                recipient_map[row['Raw Entry']] = row['Normalized Entry']
            elif row['Type'] == 'Approver Name':
                approver_map[row['Raw Entry']] = row['Normalized Entry']

    if 'recipient_name' in df_processed.columns:
        df_processed['recipient_name_clean'] = df_processed['recipient_name'].map(recipient_map).fillna(
            df_processed['recipient_name'].apply(clean_person_name))
    if 'approver_name' in df_processed.columns:
        df_processed['approver_name_clean'] = df_processed['approver_name'].map(approver_map).fillna(
            df_processed['approver_name'].apply(clean_person_name))
    if 'offered_by_org' in df_processed.columns:
        df_processed['offered_by_org_clean'] = df_processed['offered_by_org'].map(org_map).fillna(
            df_processed['offered_by_org'])

    df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'], dayfirst=True, errors='coerce')
    df_processed['date_received'] = pd.to_datetime(df_processed['date_received'], dayfirst=True, errors='coerce')
    df_processed['year_declared'] = df_processed['timestamp'].dt.year
    df_processed['month_declared'] = df_processed['timestamp'].dt.month
    df_processed['quarter_declared'] = df_processed['timestamp'].dt.quarter
    df_processed['declaration_lag_days'] = (df_processed['timestamp'] - df_processed['date_received']).dt.days

    df_processed['hospitality_category'] = df_processed['details'].apply(categorize_hospitality)
    df_processed['directorate_clean'] = df_processed['directorate'].astype(str).str.replace(' & ', ' and ', regex=False).str.strip()

    if new_persistent_orgs:
        persistent['organizations'].update(new_persistent_orgs)
        save_name_mappings(persistent)
        st.session_state.name_mappings = persistent

    new_recips = {k: v for k, v in recipient_map.items() if str(k).strip() != str(v).strip() and v}
    new_approvers = {k: v for k, v in approver_map.items() if str(k).strip() != str(v).strip() and v}
    if new_recips or new_approvers:
        persistent['recipients'].update(new_recips)
        persistent['approvers'].update(new_approvers)
        save_name_mappings(persistent)

    st.session_state.final_reports = generate_compliance_metrics(df_processed)
    st.session_state.df_processed = df_processed
    st.session_state.stage = "analysis"


# =============================================================================
# STREAMLIT UI
# =============================================================================

st.title("🎁 Hospitality & Gifts Analytics Dashboard")
st.markdown("Upload single or multiple registers, map gift/hospitality columns separately, "
            "review tiered org normalisations as clusters, then run the compliance pipeline.")

if "stage" not in st.session_state:
    st.session_state.stage = "upload"
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "mapped_working_df" not in st.session_state:
    st.session_state.mapped_working_df = None
if "name_mappings" not in st.session_state:
    st.session_state.name_mappings = load_name_mappings()

# --- STEP 1: FILE UPLOADER ---
uploaded_files = st.file_uploader("Upload Hospitality CSV Registers", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Load and Combine Files", type="primary") or st.session_state.raw_df is not None:
        if st.session_state.raw_df is None:
            dfs = []
            for file in uploaded_files:
                try:
                    dfs.append(pd.read_csv(file))
                except Exception as e:
                    st.error(f"Error loading {file.name}: {e}")
            if dfs:
                st.session_state.raw_df = pd.concat(dfs, ignore_index=True).dropna(how='all')
                st.session_state.stage = "mapping"
                st.success(f"Combined {len(dfs)} file(s). Total combined rows: {len(st.session_state.raw_df)}")

# --- STEP 2: COLUMN MAPPING UI ---
if st.session_state.stage in ["mapping", "normalization", "analysis"] and st.session_state.raw_df is not None:
    st.divider()
    st.header("1. Core Column Schema Mapping")

    all_columns = list(st.session_state.raw_df.columns)
    options_with_none = ["None"] + all_columns

    def auto_detect(options, keywords, default_idx=0):
        for i, opt in enumerate(options):
            opt_l = str(opt).lower()
            if all(k in opt_l for k in keywords):
                return i
        for i, opt in enumerate(options):
            if any(k in str(opt).lower() for k in keywords):
                return i
        return default_idx if default_idx < len(options) else 0

    core_mapping = {}
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Common fields**")
        core_mapping['timestamp'] = st.selectbox("Timestamp / Date Logged", all_columns, index=auto_detect(all_columns, ['timestamp']))
        core_mapping['recipient_name'] = st.selectbox("Recipient Name", all_columns, index=auto_detect(all_columns, ['recipient']))
        core_mapping['date_received'] = st.selectbox("Date Received / Event Date", all_columns, index=auto_detect(all_columns, ['received']))
        core_mapping['offered_by_org'] = st.selectbox("Offered By Organisation", all_columns, index=auto_detect(all_columns, ['offered']))
        core_mapping['directorate'] = st.selectbox("Directorate / Department", all_columns, index=auto_detect(all_columns, ['directorate']))
        core_mapping['status'] = st.selectbox("Status (Accepted/Declined)", all_columns, index=auto_detect(all_columns, ['status']))
        core_mapping['reason'] = st.selectbox("Reason for Acceptance", all_columns, index=auto_detect(all_columns, ['reason']))

    with col2:
        st.markdown("**🎁 Gifts**")
        core_mapping['gift_value'] = st.selectbox("Gift Value", options_with_none, index=auto_detect(options_with_none, ['gift', 'value']))
        core_mapping['gift_description'] = st.selectbox("Gift Description", options_with_none, index=auto_detect(options_with_none, ['gift', 'desc']))

    with col3:
        st.markdown("**🍽️ Hospitality**")
        core_mapping['hospitality_value'] = st.selectbox("Hospitality Value", options_with_none, index=auto_detect(options_with_none, ['hospitality', 'value']))
        core_mapping['hospitality_description'] = st.selectbox("Hospitality Description", options_with_none, index=auto_detect(options_with_none, ['hospitality', 'desc']))

    st.markdown("**Optional layers**")
    oc1, oc2 = st.columns(2)
    with oc1:
        core_mapping['grade'] = st.selectbox("Grade (Optional)", options_with_none, index=auto_detect(options_with_none, ['grade']))
    with oc2:
        core_mapping['approver_name'] = st.selectbox("Approver Name (Optional)", options_with_none, index=auto_detect(options_with_none, ['approver']))

    if st.button("Lock Schema & Prepare Normalisations"):
        if core_mapping['gift_value'] == "None" and core_mapping['hospitality_value'] == "None":
            st.error("At least one of Gift Value or Hospitality Value must be mapped.")
        else:
            mapped_df = st.session_state.raw_df.copy()
            inv_map = {v: k for k, v in core_mapping.items() if v != "None"}
            mapped_df = mapped_df.rename(columns=inv_map)
            keep_cols = list(set(inv_map.values()))
            mapped_df = mapped_df[keep_cols]

            # Whitespace strip — runs once, on the raw text columns, before any normalisation
            text_cols_to_strip = ['recipient_name', 'offered_by_org', 'directorate', 'status',
                                  'gift_description', 'hospitality_description', 'reason', 'approver_name']
            for col in text_cols_to_strip:
                if col in mapped_df.columns:
                    mapped_df[col] = mapped_df[col].astype(str).str.strip()
                    mapped_df[col] = mapped_df[col].replace({'nan': pd.NA, '': pd.NA, 'None': pd.NA})

            # Parse gift + hospitality values separately
            if 'gift_value' in mapped_df.columns:
                mapped_df['gift_value_gbp'] = mapped_df['gift_value'].apply(parse_value)
            if 'hospitality_value' in mapped_df.columns:
                mapped_df['hospitality_value_gbp'] = mapped_df['hospitality_value'].apply(parse_value)

            g = mapped_df['gift_value_gbp'] if 'gift_value_gbp' in mapped_df.columns else pd.Series(np.nan, index=mapped_df.index)
            h = mapped_df['hospitality_value_gbp'] if 'hospitality_value_gbp' in mapped_df.columns else pd.Series(np.nan, index=mapped_df.index)
            mapped_df['value_parsed_gbp'] = g.fillna(0) + h.fillna(0)
            mapped_df.loc[g.isna() & h.isna(), 'value_parsed_gbp'] = np.nan

            # record_type
            has_gift = g.notna() & (g > 0) if 'gift_value_gbp' in mapped_df.columns else pd.Series(False, index=mapped_df.index)
            has_hosp = h.notna() & (h > 0) if 'hospitality_value_gbp' in mapped_df.columns else pd.Series(False, index=mapped_df.index)
            if 'gift_description' in mapped_df.columns:
                has_gift = has_gift | mapped_df['gift_description'].notna()
            if 'hospitality_description' in mapped_df.columns:
                has_hosp = has_hosp | mapped_df['hospitality_description'].notna()
            rt = np.where(has_gift & has_hosp, 'Both', np.where(has_gift, 'Gift', np.where(has_hosp, 'Hospitality', 'Unknown')))
            mapped_df['record_type'] = rt

            # Combined details (used by categorize_hospitality)
            def _combine(row):
                parts = []
                if 'gift_description' in mapped_df.columns and pd.notna(row.get('gift_description')):
                    parts.append(str(row['gift_description']))
                if 'hospitality_description' in mapped_df.columns and pd.notna(row.get('hospitality_description')):
                    parts.append(str(row['hospitality_description']))
                return ' | '.join(parts) if parts else ''
            mapped_df['details'] = mapped_df.apply(_combine, axis=1)

            # Combined raw value (for display in data quality log)
            def _combine_raw(row):
                parts = []
                if 'gift_value' in mapped_df.columns and pd.notna(row.get('gift_value')):
                    parts.append(f"gift: {row['gift_value']}")
                if 'hospitality_value' in mapped_df.columns and pd.notna(row.get('hospitality_value')):
                    parts.append(f"hosp: {row['hospitality_value']}")
                return ' | '.join(parts) if parts else ''
            mapped_df['value_raw'] = mapped_df.apply(_combine_raw, axis=1)

            st.session_state.mapped_working_df = mapped_df

            # Run tiered org normalisation
            persistent = st.session_state.name_mappings
            raw_orgs = mapped_df['offered_by_org'].dropna().unique().tolist() if 'offered_by_org' in mapped_df.columns else []
            auto_map, auto_log, review_records = compute_org_normalisations(raw_orgs, persistent.get('organizations', {}))

            # Annotate review records with value + directorate metrics for risk-weighted sorting
            value_lookup = mapped_df.groupby('offered_by_org')['value_parsed_gbp'].sum().to_dict() if 'offered_by_org' in mapped_df.columns else {}
            dir_lookup = mapped_df.groupby('offered_by_org')['directorate'].nunique().to_dict() if {'offered_by_org', 'directorate'}.issubset(mapped_df.columns) else {}
            count_lookup = mapped_df['offered_by_org'].value_counts().to_dict() if 'offered_by_org' in mapped_df.columns else {}

            st.session_state.org_auto_map = auto_map
            st.session_state.org_auto_log = pd.DataFrame(auto_log)
            st.session_state.org_review_records = review_records
            st.session_state.org_value_lookup = value_lookup
            st.session_state.org_dir_lookup = dir_lookup
            st.session_state.org_count_lookup = count_lookup

            # Recipient + Approver row-by-row suggestions
            recip_recs = []
            if 'recipient_name' in mapped_df.columns:
                recip_recs = compute_person_normalisations(
                    mapped_df['recipient_name'].dropna().unique(),
                    persistent.get('recipients', {}),
                    'Recipient Name',
                )
            approver_recs = []
            if 'approver_name' in mapped_df.columns:
                approver_recs = compute_person_normalisations(
                    mapped_df['approver_name'].dropna().unique(),
                    persistent.get('approvers', {}),
                    'Approver Name',
                )
            st.session_state.person_review_df = pd.DataFrame(recip_recs + approver_recs)

            st.session_state.stage = "normalization"

# --- STEP 3: NORMALISATION REVIEW ---
if st.session_state.stage in ["normalization", "analysis"] and st.session_state.mapped_working_df is not None:
    st.divider()
    st.header("2. Review Normalisations")

    persistent = st.session_state.name_mappings

    # Auto-applied org merges (transparent log)
    auto_log_df = st.session_state.get('org_auto_log', pd.DataFrame())
    if not auto_log_df.empty:
        with st.expander(f"✅ Auto-applied org merges ({len(auto_log_df)} entries)", expanded=False):
            st.caption("These were merged silently because they were exact matches after light normalisation, "
                       "manual-map hits, or high-confidence fuzzy matches with no length/human-content red flags.")
            st.dataframe(auto_log_df, use_container_width=True)

    # Cluster review for orgs
    st.subheader("Org cluster review")
    review_records = st.session_state.get('org_review_records', [])
    value_lookup = st.session_state.get('org_value_lookup', {})
    dir_lookup = st.session_state.get('org_dir_lookup', {})
    count_lookup = st.session_state.get('org_count_lookup', {})

    if not review_records:
        st.success("No org name pairs require review. Run the pipeline below.")
    else:
        # Group review records by proposed canonical, ranked by total £ value
        clusters = {}
        for rec in review_records:
            clusters.setdefault(rec['Proposed_Canonical'], []).append(rec)

        def cluster_metrics(members):
            total_value = sum(value_lookup.get(m['Raw'], 0) or 0 for m in members)
            total_count = sum(count_lookup.get(m['Raw'], 0) or 0 for m in members)
            directorates = set()
            for m in members:
                # nunique returns int; we cannot get distinct dir names here cheaply, so use count as proxy
                directorates.add(dir_lookup.get(m['Raw'], 0))
            return total_value, total_count, max(directorates) if directorates else 0

        sorted_clusters = sorted(
            clusters.items(),
            key=lambda kv: (cluster_metrics(kv[1])[0], cluster_metrics(kv[1])[1]),
            reverse=True,
        )

        st.caption("Clusters sorted by total £ value (counter-fraud risk weighting). "
                   "Edit the canonical text or untick members to split them out of the cluster.")

        for i, (proposed, members) in enumerate(sorted_clusters):
            total_value, total_count, dir_spread = cluster_metrics(members)
            flags = sorted({f for m in members for f in m['Flags'].split(', ') if f})
            header = f"**{proposed}** — {len(members)} variant(s), {total_count} rows, £{total_value:,.0f} accepted/declared"
            if flags:
                header += f" · flags: {', '.join(flags)}"
            with st.expander(header, expanded=False):
                st.text_input(
                    "Canonical name",
                    value=proposed,
                    key=f"cluster_canon_{i}",
                )
                member_df = pd.DataFrame([{
                    'Include': True,
                    'Raw': m['Raw'],
                    'Score': m['Score'],
                    'Len ratio': m['Length_Ratio'],
                    'Flags': m['Flags'],
                    'Rows': count_lookup.get(m['Raw'], 0),
                    '£ total': value_lookup.get(m['Raw'], 0) or 0,
                    'Parenthetical': m['Parenthetical'],
                } for m in members])
                edited = st.data_editor(
                    member_df,
                    column_config={
                        'Include': st.column_config.CheckboxColumn(help="Untick to split this raw out of the cluster"),
                        'Raw': st.column_config.TextColumn(disabled=True),
                        'Score': st.column_config.NumberColumn(disabled=True),
                        'Len ratio': st.column_config.NumberColumn(disabled=True),
                        'Flags': st.column_config.TextColumn(disabled=True),
                        'Rows': st.column_config.NumberColumn(disabled=True),
                        '£ total': st.column_config.NumberColumn(format='£%.0f', disabled=True),
                        'Parenthetical': st.column_config.TextColumn(disabled=True),
                    },
                    hide_index=True,
                    use_container_width=True,
                    key=f"cluster_members_{i}",
                )
                # Stash the resolved members for the apply step
                st.session_state[f"cluster_resolved_{i}"] = edited
        st.session_state.cluster_count = len(sorted_clusters)
        st.session_state.cluster_proposed = [p for p, _ in sorted_clusters]
        # Per-canonical (member_count, total_value) — used by the orphan-review stage
        st.session_state.cluster_metrics_map = {
            proposed: (len(members), cluster_metrics(members)[0])
            for proposed, members in sorted_clusters
        }

    # Recipient / Approver row-by-row review
    person_df = st.session_state.get('person_review_df', pd.DataFrame())
    if not person_df.empty:
        st.subheader("Recipient / Approver name review")
        st.caption("Edit the Normalized Entry cell to override. Leave as-is to accept the proposed value.")
        edited_person = st.data_editor(
            person_df,
            column_config={
                'Type': st.column_config.TextColumn(disabled=True),
                'Raw Entry': st.column_config.TextColumn(disabled=True),
                'Normalized Entry': st.column_config.TextColumn(disabled=False),
            },
            hide_index=True,
            use_container_width=True,
            key='person_editor',
        )
        st.session_state.person_review_df_edited = edited_person
    else:
        st.session_state.person_review_df_edited = pd.DataFrame()

    # Persistent dictionary management
    with st.expander("📚 Manual mappings (persistent across sessions)", expanded=False):
        st.caption(f"Stored at `{MAPPING_FILE}`. Edits here are written immediately on Save.")
        tab_o, tab_r, tab_a = st.tabs(["Organizations", "Recipients", "Approvers"])
        for label, key, tab in [("Organizations", "organizations", tab_o),
                                 ("Recipients", "recipients", tab_r),
                                 ("Approvers", "approvers", tab_a)]:
            with tab:
                current = persistent.get(key, {})
                edit_df = pd.DataFrame(
                    [{'Raw Entry': k, 'Canonical': v} for k, v in sorted(current.items())]
                    or [{'Raw Entry': '', 'Canonical': ''}]
                )
                edited = st.data_editor(
                    edit_df,
                    num_rows='dynamic',
                    use_container_width=True,
                    key=f'persistent_editor_{key}',
                )
                if st.button(f"Save {label}", key=f'save_{key}'):
                    new_map = {
                        str(row['Raw Entry']).strip(): str(row['Canonical']).strip()
                        for _, row in edited.iterrows()
                        if str(row.get('Raw Entry', '')).strip() and str(row.get('Canonical', '')).strip()
                    }
                    persistent[key] = new_map
                    if save_name_mappings(persistent):
                        st.success(f"Saved {len(new_map)} {label.lower()} mapping(s) to {MAPPING_FILE.name}")

    if st.button("Apply Normalisations & Run Pipeline", type="primary"):
        # Collect orphans before applying. An "orphan" is a cluster member the
        # user explicitly unticked — it floats with no proposed canonical.
        orphans = []
        cluster_count = st.session_state.get('cluster_count', 0)
        cluster_proposed = st.session_state.get('cluster_proposed', [])
        for i in range(cluster_count):
            proposed = cluster_proposed[i] if i < len(cluster_proposed) else ''
            resolved = st.session_state.get(f"cluster_resolved_{i}")
            if resolved is None:
                continue
            for _, row in resolved.iterrows():
                if not row.get('Include', True):
                    orphans.append({
                        'Raw': row['Raw'],
                        'Original': proposed,
                        'Score_To_Original': row.get('Score', 0),
                        'Value': row.get('£ total', 0) or 0,
                        'Rows': row.get('Rows', 0) or 0,
                    })

        if orphans:
            st.session_state.orphans = orphans
            st.session_state.stage = "orphan_review"
        else:
            run_analysis_pipeline()


# --- STEP 2b: ORPHAN REASSIGNMENT ---
if st.session_state.stage == "orphan_review" and st.session_state.get('orphans'):
    st.divider()
    st.header("2b. Reassign orphaned offerer names")
    st.caption("These were unticked from their proposed clusters. "
               "Pick a destination, leave as a singleton, or create a new canonical. "
               "Re-clustering by fuzzy match alone would just put them back where you took them out from — "
               "so the choice here is human-led.")

    persistent = st.session_state.name_mappings
    metrics_map = st.session_state.get('cluster_metrics_map', {})
    auto_map = st.session_state.get('org_auto_map', {})

    # Build the surviving canonical pool with cluster-size + £ metadata.
    canonical_metadata = {}
    for canon in set(auto_map.values()):
        canonical_metadata.setdefault(canon, (0, 0.0))
    for i in range(st.session_state.get('cluster_count', 0)):
        canon = st.session_state.get(f"cluster_canon_{i}", '').strip()
        resolved = st.session_state.get(f"cluster_resolved_{i}")
        if not canon or resolved is None:
            continue
        if not resolved['Include'].any():
            continue
        proposed = st.session_state.cluster_proposed[i]
        size, value = metrics_map.get(proposed, (0, 0.0))
        canonical_metadata[canon] = (size, value)
    for dst in persistent.get('organizations', {}).values():
        canonical_metadata.setdefault(dst, (0, 0.0))

    include_original = st.checkbox(
        "Allow the original cluster as a suggestion for these orphans",
        value=False,
        help="Hidden by default so you don't accidentally re-add a raw to the cluster you just unticked from.",
    )

    orphans_sorted = sorted(st.session_state.orphans, key=lambda o: -(o['Value'] or 0))

    for i, o in enumerate(orphans_sorted):
        raw = o['Raw']
        original = o['Original']
        nrm_raw = light_normalise(raw)

        scored = []
        for canon, (size, value) in canonical_metadata.items():
            if canon == original and not include_original:
                continue
            if fuzz is not None:
                score = fuzz.token_set_ratio(nrm_raw, light_normalise(canon))
            else:
                score = 0
            scored.append((canon, size, value, score))
        scored.sort(key=lambda x: -x[3])

        def _fmt(canon, size, value, score):
            return f"{canon}   [{size} variants, £{value:,.0f}]   ·   match {score}"

        options = (
            ["(Keep as singleton)"]
            + [_fmt(c, n, v, s) for c, n, v, s in scored[:50]]
            + ["(Create new canonical →)"]
        )

        st.markdown(
            f"**`{raw}`** — was in *{original}*, "
            f"£{o['Value']:,.0f}, {o['Rows']} rows"
        )
        choice = st.selectbox(
            "Reassign to:",
            options,
            key=f"orphan_choice_{i}",
            label_visibility="collapsed",
        )
        if choice == "(Create new canonical →)":
            new_name = st.text_input("New canonical name:", key=f"orphan_new_{i}")
            st.session_state[f"orphan_resolved_{i}"] = new_name.strip() or raw
        elif choice == "(Keep as singleton)":
            st.session_state[f"orphan_resolved_{i}"] = raw
        else:
            st.session_state[f"orphan_resolved_{i}"] = choice.split("   ·   ")[0].split("   [")[0].strip()

    if st.button("Confirm Reassignments & Run Pipeline", type="primary"):
        st.session_state.orphan_overrides = {
            o['Raw']: st.session_state.get(f"orphan_resolved_{i}", o['Raw'])
            for i, o in enumerate(orphans_sorted)
        }
        run_analysis_pipeline()

# --- STEP 4: REPORTS & EXPORTS ---
if st.session_state.stage == "analysis" and "final_reports" in st.session_state:
    st.divider()
    st.header("3. Quality Logs & Compliance Analysis Reports")

    proc_df = st.session_state.df_processed
    reports = st.session_state.final_reports

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Rows Evaluated", len(proc_df))
    with c2:
        accepted_sum = proc_df[proc_df['status'].astype(str).str.lower() == 'accepted']['value_parsed_gbp'].sum()
        st.metric("Total Accepted Spend", f"£{accepted_sum:,.2f}")
    with c3:
        st.metric("Unparsable Value Failures", int(proc_df['value_parsed_gbp'].isna().sum()))
    with c4:
        med_lag = proc_df['declaration_lag_days'].median()
        st.metric("Median Declaration Lag", f"{med_lag:.1f} Days" if pd.notna(med_lag) else "N/A")

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "Directorate & Category",
        "Grade Analysis",
        "Compliance Metrics",
        "Risk Rankings",
        "Group Events & High-Freq",
        "Data Quality Logs",
    ])

    with t1:
        st.subheader("Value Summary by Directorate")
        st.dataframe(reports.get('Value_by_Directorate_YR'), use_container_width=True)
        st.subheader("Value Summary by Category")
        st.dataframe(reports.get('Value_by_Category'), use_container_width=True)
        if 'Value_by_RecordType' in reports:
            st.subheader("Value Summary by Gift / Hospitality Type")
            st.dataframe(reports.get('Value_by_RecordType'), use_container_width=True)

    with t2:
        if 'Value_by_Grade_YR' in reports:
            st.subheader("Value Breakdown Across Grades")
            st.dataframe(reports.get('Value_by_Grade_YR'), use_container_width=True)
        else:
            st.info("Grade analysis was omitted or grade column was not mapped.")

    with t3:
        st.subheader("Directorate Compliance Indicators")
        st.dataframe(reports.get('Compliance_by_Directorate'), use_container_width=True)

    with t4:
        ind_data = reports.get('Individual_Compliance_Data', {})
        cl, cr = st.columns(2)
        with cl:
            st.subheader("Top 10 Value-Parsing Issues")
            if 'top_val_err' in ind_data:
                st.dataframe(ind_data['top_val_err'], use_container_width=True)
            else:
                st.write("No value errors found.")
        with cr:
            st.subheader("Top 10 Highest Declaration Lags")
            st.dataframe(ind_data.get('top_lag'), use_container_width=True)

    with t5:
        st.subheader("Potential Group Events (Date & Org match)")
        st.dataframe(reports.get('Potential_Group_Events'), use_container_width=True)
        st.subheader("High-Frequency Recipient-Donor Pairings (Top 50)")
        st.dataframe(reports.get('High_Freq_Pairings'), use_container_width=True)

    with t6:
        st.subheader("Data Quality Log")
        if 'Data_Quality_Log' in reports:
            st.dataframe(reports.get('Data_Quality_Log'), use_container_width=True)
        else:
            st.success("Clean data — no quality issues detected.")

    # ---- EXPORTS ----
    st.divider()
    st.subheader("📥 Exports")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    ec1, ec2 = st.columns(2)

    with ec1:
        cleaned_buffer = io.BytesIO()
        with pd.ExcelWriter(cleaned_buffer, engine='openpyxl') as writer:
            proc_df.to_excel(writer, sheet_name='cleaned_register', index=False)
        st.download_button(
            label="Download Cleaned Register (.xlsx)",
            data=cleaned_buffer.getvalue(),
            file_name=f"cleaned_register_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Post-clean, post-normalisation register: one row per input row with all derived "
                 "columns (*_clean, value_parsed_gbp, hospitality_category, record_type).",
        )

    with ec2:
        reports_buffer = io.BytesIO()
        with pd.ExcelWriter(reports_buffer, engine='openpyxl') as writer:
            for sheet_name, data in reports.items():
                if isinstance(data, dict):
                    for sub_key, sub_df in data.items():
                        sub_df.to_excel(writer, sheet_name=sub_key[:31], index=False)
                else:
                    data.to_excel(writer, sheet_name=sheet_name[:31])
        st.download_button(
            label="Download Analytical Reports (.xlsx)",
            data=reports_buffer.getvalue(),
            file_name=f"hospitality_analysis_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
