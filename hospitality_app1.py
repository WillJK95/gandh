import streamlit as st
import pandas as pd
import numpy as np
import re
import io

# Handle fuzzy matching imports gracefully
try:
    from thefuzz import process, fuzz
except ImportError:
    process = None

st.set_page_config(page_title="Hospitality & Gifts Analytics Engine", layout="wide")

# -----------------------------------------------------------------------------
# CORE ANALYTICAL FUNCTIONS (Adapted from original script)
# -----------------------------------------------------------------------------

def parse_value(value_str):
    if pd.isna(value_str): return np.nan
    value_str = str(value_str).lower().strip()
    if any(term in value_str for term in ['n/a', 'unknown', 'nil', 'unclear', 'unsure', 'free', 'no gift', 'not known']): 
        return np.nan
    if '65 gbp' in value_str: return 65.0
    if '47 or 35' in value_str: return 47.0
    
    currency_numbers = [float(num.replace(',', '')) for num in re.findall(r'(?:£|gbp)\s*(\d[\d,.]*)', value_str)]
    if not currency_numbers:
        currency_numbers = [float(num.replace(',', '')) for num in re.findall(r'(\d[\d,.]*)\s*(?:£|gbp)', value_str)]
        
    all_numbers = [float(num.replace(',', '')) for num in re.findall(r'(\d[\d,.]*)', value_str)]
    numbers = currency_numbers if currency_numbers else all_numbers
    
    if len(numbers) > 1:
        filtered_numbers = [n for n in numbers if not (2000 <= n <= 2035 and n == int(n))]
        if filtered_numbers:
            numbers = filtered_numbers
            
    if not numbers: return np.nan
    if any(term in value_str for term in ['-', 'to']) and len(numbers) > 1: return np.mean(numbers)
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


def clean_person_name(name_val):
    if pd.isna(name_val) or not isinstance(name_val, str):
        return name_val
    return name_val.replace(' & ', ' and ').replace('-', ' ').title().strip()


def compute_suggested_normalizations(df):
    """
    Pre-calculates suggested normalizations for Recipient, Approver, and Orgs 
    so the user can review and override them in the UI.
    """
    normalization_records = []

    # 1. Recipient Names
    if 'recipient_name' in df.columns:
        for raw in df['recipient_name'].dropna().unique():
            clean = clean_person_name(str(raw))
            if str(raw).strip() != str(clean).strip():
                normalization_records.append({'Type': 'Recipient Name', 'Raw Entry': raw, 'Normalized Entry': clean})

    # 2. Approver Names (Optional)
    if 'approver_name' in df.columns:
        for raw in df['approver_name'].dropna().unique():
            clean = clean_person_name(str(raw))
            if str(raw).strip() != str(clean).strip():
                normalization_records.append({'Type': 'Approver Name', 'Raw Entry': raw, 'Normalized Entry': clean})

    # 3. Organizations (Manual Map + Fuzzy matching)
    if 'offered_by_org' in df.columns:
        manual_org_map = {
            'FA': 'Football Association', 'The FA': 'Football Association', 'The Football Association': 'Football Association',
            'The English Football Association': 'Football Association', 'UEFA/FA': 'Football Association',
            'The English Fottball Association': 'Football Association', 'English FA': 'Football Association',
            'The FA (Football Association)': 'Football Association', 'pwc': 'PwC', 'National Lottery': 'National Lottery',
            'The National Lottery': 'National Lottery', 'National Lottery Heritage Fund': 'National Lottery Heritage Fund',
            'The EFL': 'English Football League', 'BFI': 'British Film Institute', 'BFI ': 'British Film Institute',
            'RFU': 'Rugby Football Union', 'The National Theatre': 'National Theatre'
        }
        
        unique_orgs = df['offered_by_org'].dropna().unique()
        # Initial Pass with manual map
        mapped_orgs = {org: manual_org_map.get(org, org) for org in unique_orgs}
        
        if process:
            canonical_org_map = {}
            for org in set(mapped_orgs.values()):
                normalized = re.sub(r'[^\w\s]', '', org.lower())
                normalized = re.sub(r'\b(the|ltd|llp|inc|plc|group|and)\b', '', normalized).strip()
                if not normalized: continue
                if canonical_org_map:
                    match, score = process.extractOne(normalized, list(canonical_org_map.keys()), scorer=fuzz.token_set_ratio)
                    if score > 85: continue
                canonical_org_map[normalized] = org
            
            for raw_org, current_org in mapped_orgs.items():
                normalized = re.sub(r'[^\w\s]', '', current_org.lower())
                normalized = re.sub(r'\b(the|ltd|llp|inc|plc|group|and)\b', '', normalized).strip()
                if not normalized: continue
                match, score = process.extractOne(normalized, list(canonical_org_map.keys()), scorer=fuzz.token_set_ratio)
                if score > 85:
                    mapped_orgs[raw_org] = canonical_org_map[match]

        for raw, clean in mapped_orgs.items():
            if str(raw).strip() != str(clean).strip():
                normalization_records.append({'Type': 'Offered By Organization', 'Raw Entry': raw, 'Normalized Entry': clean})

    return pd.DataFrame(normalization_records)


def calculate_single_group_compliance(group):
    if group.empty: return pd.Series(dtype='float64')
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
        
        is_name_match = accepted_group['recipient_name_clean'].str.lower() == accepted_group['approver_name_clean'].str.lower()
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
    
    # Tab 1: Directorate Calculations
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

    # Tab 1b: Analytics Metrics by Categorization Layer
    accepted_cat = accepted_df.groupby(['hospitality_category'] + time_grouping, dropna=False)['value_parsed_gbp'].agg(['count', 'sum', 'mean'])
    accepted_cat.columns = ['Accepted_Count', 'Accepted_Total_GBP', 'Accepted_Avg_GBP']
    report_dfs['Value_by_Category'] = accepted_cat.round(2)

    # Tab 2: Grade Analysis Framework (Modular Rule)
    if 'grade' in df.columns:
        accepted_grade = accepted_df.groupby(['grade'] + time_grouping)['value_parsed_gbp'].agg(['sum', 'mean', 'median', 'max'])
        accepted_grade.columns = ['Accepted_' + col for col in accepted_grade.columns]
        declined_grade = df[df['status'].str.lower() == 'declined'].groupby(['grade'] + time_grouping)['value_parsed_gbp'].agg(['sum', 'mean', 'median', 'max'])
        declined_grade.columns = ['Declined_' + col for col in declined_grade.columns]
        val_grade = pd.concat([accepted_grade, declined_grade], axis=1, sort=True).fillna(0)
        report_dfs['Value_by_Grade_YR'] = val_grade.round(2)

    # Tab 3: Compliance Rates Summaries
    dir_comp = df.groupby(['directorate_clean'] + time_grouping, dropna=False).apply(calculate_single_group_compliance).round(2).reset_index()
    report_dfs['Compliance_by_Directorate'] = dir_comp

    # Tab 4: Individual Standings Mapping
    rankings = {}
    non_compliant_val = df[df['value_parsed_gbp'].isna()]
    if not non_compliant_val.empty:
        rankings['top_val_err'] = non_compliant_val['recipient_name_clean'].value_counts().nlargest(10).to_frame('Entries_With_Value_Errors').reset_index()
    
    rankings['top_lag'] = df.groupby('recipient_name_clean')['declaration_lag_days'].max().nlargest(10).to_frame('Max_Lag_Days').reset_index()
    report_dfs['Individual_Compliance_Data'] = rankings

    # Tab 5: Concentrated Group Configurations
    group_cols = ['timestamp', 'date_received', 'recipient_name_clean', 'directorate_clean', 'offered_by_org_clean', 'details', 'value_parsed_gbp']
    potential_groups = accepted_df[accepted_df.duplicated(subset=['date_received', 'offered_by_org_clean'], keep=False)].sort_values(by=['offered_by_org_clean', 'date_received'])
    report_dfs['Potential_Group_Events'] = potential_groups[[c for c in group_cols if c in potential_groups.columns]]

    # Tab 6: High Frequency Vectors
    status_grouped = df.groupby(['recipient_name_clean', 'offered_by_org_clean', 'status']).size().unstack(fill_value=0)
    status_grouped['Total'] = status_grouped.sum(axis=1)
    report_dfs['High_Freq_Pairings'] = status_grouped.sort_values('Total', ascending=False).head(50)

    # Tab 7: Isolation Quality Logs
    issue_mask = df['timestamp'].isna() | df['value_parsed_gbp'].isna() | df['directorate_clean'].isna()
    if issue_mask.any():
        log_df = df[issue_mask].copy()
        log_df['Issue_Reason'] = ''
        log_df.loc[log_df['timestamp'].isna(), 'Issue_Reason'] += 'Invalid Timestamp; '
        log_df.loc[log_df['value_parsed_gbp'].isna(), 'Issue_Reason'] += 'Unparsable Value; '
        log_df.loc[log_df['directorate_clean'].isna(), 'Issue_Reason'] += 'Missing Directorate; '
        log_cols = ['Issue_Reason', 'timestamp', 'recipient_name', 'value_raw', 'directorate', 'details']
        report_dfs['Data_Quality_Log'] = log_df[[c for c in log_cols if c in log_df.columns]]

    return report_dfs

# -----------------------------------------------------------------------------
# STREAMLIT UI & APPS ENGINE
# -----------------------------------------------------------------------------

st.title("🎁 Hospitality & Gifts Analytics Dashboard")
st.markdown("Upload single or multiple registers, adjust schema mapping, and override normalizations interactively.")

if "stage" not in st.session_state:
    st.session_state.stage = "upload"
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "norm_df" not in st.session_state:
    st.session_state.norm_df = None

# --- STEP 1: FILE UPLOADER & CONCATENATION ---
uploaded_files = st.file_uploader("Upload Hospitality CSV Registers", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Load and Combine Files", type="primary") or st.session_state.raw_df is not None:
        if st.session_state.raw_df is None:
            dfs = []
            for file in uploaded_files:
                try:
                    loaded_df = pd.read_csv(file)
                    dfs.append(loaded_df)
                except Exception as e:
                    st.error(f"Error loading {file.name}: {e}")
            if dfs:
                st.session_state.raw_df = pd.concat(dfs, ignore_index=True).dropna(how='all')
                st.session_state.stage = "mapping"
                st.success(f"Successfully combined {len(dfs)} file(s). Total combined rows: {len(st.session_state.raw_df)}")

# --- STEP 2: MODULAR COLUMN MAPPING UI ---
if st.session_state.stage in ["mapping", "normalization", "analysis"] and st.session_state.raw_df is not None:
    st.divider()
    st.header("1. Core Column Schema Mapping")
    
    all_columns = list(st.session_state.raw_df.columns)
    
    # Layout schema mapping selectors
    col1, col2, col3 = st.columns(3)
    
    core_mapping = {}
    
    # Auto-detection helpers
    def auto_detect(options, keywords, default_idx=0):
        for i, opt in enumerate(options):
            if any(k in str(opt).lower() for k in keywords):
                return i
        return default_idx if default_idx < len(options) else 0

    with col1:
        core_mapping['timestamp'] = st.selectbox("Timestamp / Date Logged", all_columns, index=auto_detect(all_columns, ['timestamp', 'logged']))
        core_mapping['recipient_name'] = st.selectbox("Recipient Name", all_columns, index=auto_detect(all_columns, ['recipient', 'name']))
        core_mapping['date_received'] = st.selectbox("Date Received / Event Date", all_columns, index=auto_detect(all_columns, ['received', 'event']))
        core_mapping['details'] = st.selectbox("Details / Hospitality Description", all_columns, index=auto_detect(all_columns, ['detail', 'description']))

    with col2:
        core_mapping['offered_by_org'] = st.selectbox("Offered By Organization", all_columns, index=auto_detect(all_columns, ['org', 'offered', 'company']))
        core_mapping['directorate'] = st.selectbox("Directorate / Department", all_columns, index=auto_detect(all_columns, ['directorate', 'dept', 'department']))
        core_mapping['status'] = st.selectbox("Status (Accepted/Declined)", all_columns, index=auto_detect(all_columns, ['status']))
        core_mapping['reason'] = st.selectbox("Reason for Acceptance", all_columns, index=auto_detect(all_columns, ['reason']))

    with col3:
        core_mapping['value_raw'] = st.selectbox("Raw Estimated Value Field", all_columns, index=auto_detect(all_columns, ['value', 'cost', 'price']))
        
        # Optional modular layers
        use_grade = st.checkbox("Include Grade Analysis Layer?", value='grade' in [c.lower() for c in all_columns])
        core_mapping['grade'] = st.selectbox("Grade (Optional)", ["None"] + all_columns, index=auto_detect(["None"] + all_columns, ['grade']) if use_grade else 0)
        
        use_approver = st.checkbox("Include Approver Compliance Layer?", value='approver' in [c.lower() for c in all_columns])
        core_mapping['approver_name'] = st.selectbox("Approver Name (Optional)", ["None"] + all_columns, index=auto_detect(["None"] + all_columns, ['approver']) if use_approver else 0)

    if st.button("Lock Schema & Extract Normalizations"):
        # Map columns safely without breaking source structures
        mapped_df = st.session_state.raw_df.copy()
        inv_map = {v: k for k, v in core_mapping.items() if v != "None"}
        mapped_df = mapped_df.rename(columns=inv_map)
        
        # Keep only the columns we successfully mapped
        keep_cols = list(inv_map.values())
        st.session_state.mapped_working_df = mapped_df[keep_cols]
        
        # Generate proposed updates
        st.session_state.norm_df = compute_suggested_normalizations(st.session_state.mapped_working_df)
        st.session_state.stage = "normalization"

# --- STEP 3: INTERACTIVE REJECTION / OVERRIDE TABLE ---
if st.session_state.stage in ["normalization", "analysis"] and st.session_state.norm_df is not None:
    st.divider()
    st.header("2. Review Name & Organization Normalization Filters")
    st.markdown(
        "> **Reviewing Changes Below:** To reject a suggested variation, copy the value from **Raw Entry** into the editable **Normalized Entry** cell. "
        "You can also freetext write any alternative custom assignment directly into the **Normalized Entry** cell."
    )
    
    if not st.session_state.norm_df.empty:
        # Display editable table view
        edited_norm_df = st.data_editor(
            st.session_state.norm_df,
            column_config={
                "Type": st.column_config.TextColumn(disabled=True),
                "Raw Entry": st.column_config.TextColumn(disabled=True),
                "Normalized Entry": st.column_config.TextColumn(disabled=False)
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No text variances required structural string standardization adjustments.")
        edited_norm_df = st.session_state.norm_df

    if st.button("Run Full Analytics Pipeline", type="primary"):
        # Build normalization lookup frames from finalized user adjustments
        df_processed = st.session_state.mapped_working_df.copy()
        
        recipient_map = edited_norm_df[edited_norm_df['Type'] == 'Recipient Name'].set_index('Raw Entry')['Normalized Entry'].to_dict()
        approver_map = edited_norm_df[edited_norm_df['Type'] == 'Approver Name'].set_index('Raw Entry')['Normalized Entry'].to_dict()
        org_map = edited_norm_df[edited_norm_df['Type'] == 'Offered By Organization'].set_index('Raw Entry')['Normalized Entry'].to_dict()

        # Apply basic cleaning falls back to user choices
        df_processed['recipient_name_clean'] = df_processed['recipient_name'].map(recipient_map).fillna(df_processed['recipient_name'].apply(clean_person_name))
        if 'approver_name' in df_processed.columns:
            df_processed['approver_name_clean'] = df_processed['approver_name'].map(approver_map).fillna(df_processed['approver_name'].apply(clean_person_name))
        df_processed['offered_by_org_clean'] = df_processed['offered_by_org'].map(org_map).fillna(df_processed['offered_by_org'])

        # Time Series, Delays & Categories Pipeline
        df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'], dayfirst=True, errors='coerce')
        df_processed['date_received'] = pd.to_datetime(df_processed['date_received'], dayfirst=True, errors='coerce')
        df_processed['year_declared'] = df_processed['timestamp'].dt.year
        df_processed['month_declared'] = df_processed['timestamp'].dt.month
        df_processed['quarter_declared'] = df_processed['timestamp'].dt.quarter
        df_processed['declaration_lag_days'] = (df_processed['timestamp'] - df_processed['date_received']).dt.days

        df_processed['value_parsed_gbp'] = df_processed['value_raw'].apply(parse_value)
        df_processed['hospitality_category'] = df_processed['details'].apply(categorize_hospitality)
        df_processed['directorate_clean'] = df_processed['directorate'].astype(str).str.replace(' & ', ' and ', regex=False).str.strip()

        # Generate metrics
        st.session_state.final_reports = generate_compliance_metrics(df_processed)
        st.session_state.df_processed = df_processed
        st.session_state.stage = "analysis"

# --- STEP 4: METRICS INTERFACE & REPORT VISUALIZATIONS ---
if st.session_state.stage == "analysis" and "final_reports" in st.session_state:
    st.divider()
    st.header("3. Quality Logs & Compliance Analysis Reports")
    
    proc_df = st.session_state.df_processed
    reports = st.session_state.final_reports

    # Summary Statistics Widgets
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Rows Evaluated", len(proc_df))
    with c2:
        st.metric("Total Accepted Spend", f"£{proc_df[proc_df['status'].str.lower()=='accepted']['value_parsed_gbp'].sum():,.2f}")
    with c3:
        st.metric("Unparsable Value Failures", proc_df['value_parsed_gbp'].isna().sum())
    with c4:
        st.metric("Median Declaration Lag", f"{proc_df['declaration_lag_days'].median():.1f} Days" if proc_df['declaration_lag_days'].notna().any() else "N/A")

    # Render Reports via Tabs
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "Directorate & Category Analysis", 
        "Grade Analysis", 
        "Compliance Metrics", 
        "Risk Rankings", 
        "Group Events & High-Freq Pairings",
        "Data Quality Logs"
    ])
    
    with t1:
        st.subheader("Value Summary by Directorate Matrix")
        st.dataframe(reports.get('Value_by_Directorate_YR'), use_container_width=True)
        st.subheader("Value Summary by Classification Categories")
        st.dataframe(reports.get('Value_by_Category'), use_container_width=True)

    with t2:
        if 'Value_by_Grade_YR' in reports:
            st.subheader("Value Breakdown Across Corporate Grades")
            st.dataframe(reports.get('Value_by_Grade_YR'), use_container_width=True)
        else:
            st.info("Grade Analysis was omitted or data layer mapping was skipped.")

    with t3:
        st.subheader("Calculated Directorate Operational Compliance Indicators")
        st.dataframe(reports.get('Compliance_by_Directorate'), use_container_width=True)

    with t4:
        ind_data = reports.get('Individual_Compliance_Data', {})
        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("Top 10 Log Entries with Value Parsing Issues")
            if 'top_val_err' in ind_data:
                st.dataframe(ind_data['top_val_err'], use_container_width=True)
            else:
                st.write("No value errors found.")
        with col_r:
            st.subheader("Top 10 Highest Declaration Lags (Days)")
            st.dataframe(ind_data.get('top_lag'), use_container_width=True)

    with t5:
        st.subheader("Potential Group Events Matrix (Matches on Date & Supplying Org)")
        st.dataframe(reports.get('Potential_Group_Events'), use_container_width=True)
        st.subheader("High Frequency Recipient-Donor Pairing Profiles (Top 50)")
        st.dataframe(reports.get('High_Freq_Pairings'), use_container_width=True)

    with t6:
        st.subheader("Isolated Broken Data Elements Registry Log")
        if 'Data_Quality_Log' in reports:
            st.dataframe(reports.get('Data_Quality_Log'), use_container_width=True)
        else:
            st.success("Clean Data! Zero tracking issue warning violations were triggered.")

    # Excel Exporter Engine Generation Block
    st.divider()
    st.subheader("📥 Export Final Clean Data Reports")
    
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        for sheet_name, data in reports.items():
            if isinstance(data, dict): # Handle nested dicts like Individual_Compliance_Data
                for sub_key, sub_df in data.items():
                    sub_df.to_excel(writer, sheet_name=sub_key[:31], index=False)
            else:
                data.to_excel(writer, sheet_name=sheet_name[:31])
    
    st.download_button(
        label="Download Analytical Reports (.xlsx)",
        data=buffer.getvalue(),
        file_name="robust_hospitality_analysis_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
