import streamlit as st
import pandas as pd
from io import BytesIO
import zipfile
import os
import shutil
from datetime import datetime
import regex as re
import numpy as np

def load_and_preprocess_tcd(tcd_filepath, keyword_mapping_df):
    # Load the original TCD without any preprocessing
    original_df = pd.read_excel(tcd_filepath)
    
    # Validate the original DataFrame to preserve row indices
    required_columns = ["Labels", "Action", "Expected Results", "Description", "link issue Test", "Planned Execution", "Summary", "Issue ID"]
    col_to_letter_col = list(original_df.columns)
    error_list = []
    keyword_set = set(keyword_mapping_df["TCD Keywords"].str.strip().str.lower())
    
    # Map column names to Excel column letters
    col_to_letter = {col: chr(65 + i) for i, col in enumerate(col_to_letter_col)}  # B=Labels, C=Action, D=Expected Results
    
    # Get user-defined label components from session state
    variants = st.session_state.get("variants", "").split(",") if st.session_state.get("variants", "") else []
    variants = [v.strip() for v in variants if v.strip()]
    type_of_testing = st.session_state.get("type_of_testing", "").split(",") if st.session_state.get("type_of_testing", "") else []
    type_of_testing = [t.strip() for t in type_of_testing if t.strip()]
    features = st.session_state.get("features", "").split(",") if st.session_state.get("features", "") else []
    features = [f.strip() for f in features if f.strip()]
    sub_features = st.session_state.get("sub_features", "").split(",") if st.session_state.get("sub_features", "") else []
    sub_features = [s.strip() for s in sub_features if s.strip()]
    functionalities = st.session_state.get("functionalities", "").split(",") if st.session_state.get("functionalities", "") else []
    functionalities = [f.strip() for f in functionalities if f.strip()]
    
    # Validate Issue ID sequence
    issue_ids = original_df.get("Issue ID", pd.Series()).astype(str).str.strip()
    expected_sequence = list(range(1, len(original_df) + 1))
    for idx, issue_id in enumerate(issue_ids):
        excel_row = idx + 2
        try:
            issue_id_int = int(issue_id)
            if issue_id_int != expected_sequence[idx]:
                error_list.append({
                    "Row": excel_row,
                    "Column": "Issue ID",
                    "Cell": f"{col_to_letter.get('Issue ID', 'Unknown')}{excel_row}",
                    "Value": issue_id,
                    "Error": "Non-sequential Issue ID",
                    "Issue Details": f"Issue ID must be sequential starting from 1 (expected {expected_sequence[idx]}, found {issue_id})"
                })
        except (ValueError, TypeError):
            error_list.append({
                "Row": excel_row,
                "Column": "Issue ID",
                "Cell": f"{col_to_letter.get('Issue ID', 'Unknown')}{excel_row}",
                "Value": issue_id,
                "Error": "Invalid Issue ID",
                "Issue Details": "Issue ID must be a numeric value"
            })
    
    for idx, row in original_df.iterrows():
        # Excel row = DataFrame index + 2 (header row in Excel)
        excel_row = idx + 2
        errors = validate_tcd_row(row, keyword_set, excel_row, col_to_letter, variants, type_of_testing, features, sub_features, functionalities)
        error_list.extend(errors)
    
    error_df = pd.DataFrame(error_list, columns=["Row", "Column", "Cell", "Value", "Error", "Issue Details"])
    
    # Now preprocess the DataFrame for script generation
    df = original_df.copy()
    
    # Filter rows where "Planned Execution" is "Planned Automation" or contains "Manual" (case-insensitive)
    df = df[df["Planned Execution"].str.lower().isin(["planned automation", "manual"])]
    
    # If no rows match the filter, return an empty DataFrame
    if df.empty:
        df = pd.DataFrame(columns=required_columns)
    else:
        # Keep only the required columns
        df = df[required_columns]
        
        # Filter out completely empty rows
        df = df.dropna(how='all')
        
        # Filter out rows where only 'Labels' has a value
        df = df.dropna(subset=['Action', 'Expected Results', 'Description', 'Summary', 'Issue ID'], how='all')
        
        # Reset index after filtering
        df = df.reset_index(drop=True)
        
        # Apply preprocessing
        df["Test_Case_Type"] = df["Labels"].apply(lambda x: x.split("_")[-1].strip().lower().replace(" ", "") if isinstance(x, str) else "")
        df["Sub_Feature"] = df["Labels"].apply(lambda x: "_".join(x.split("_")[3:-1]).strip() if isinstance(x, str) else "")
        df["feature"] = df["Labels"].apply(lambda x: x.split("_")[2].strip().lower().replace(" ", "") if isinstance(x, str) else "")
        
        def sanitize_filename(name):
            if not isinstance(name, str):
                return ""
            invalid_chars = r'[<>:"/\\|?*]+'
            sanitized = re.sub(invalid_chars, '_', name.strip().lower())
            sanitized = re.sub(r'[_\s]+', '_', sanitized)
            return sanitized.strip('_')
        
        df["Normalized_Feature"] = df["Sub_Feature"].apply(sanitize_filename)
        df["link issue Test"] = df["link issue Test"].astype(str).str.replace(r"[\n\r;,\s]+", "_", regex=True).str.strip("_")
        # Ensure Description is a non-empty string
        df["Description"] = df["Description"].fillna("Unnamed Test Case").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        df["Summary"] = df["Summary"].fillna("Unnamed Test Case").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        df["Issue ID"] = df["Issue ID"].astype(str).str.strip()
    
    return df, error_df

def validate_tcd_row(row, keyword_set, row_num, col_to_letter, variants, type_of_testing, features, sub_features, functionalities):
    errors = []
    
    # Validate Labels
    labels = row.get("Labels", "")
    if isinstance(labels, str) and labels.strip():
        pattern = re.compile(
            r"^(" + "|".join(variants) + r")_(" + "|".join(type_of_testing) + r")_(" +
            "|".join(features) + r")_([A-Za-z0-9_]+)_(" + "|".join(functionalities) + ")$",
            re.IGNORECASE
        )
        if not pattern.match(labels):
            errors.append({
                "Row": row_num,
                "Column": "Labels",
                "Cell": f"{col_to_letter['Labels']}{row_num}",
                "Value": labels,
                "Error": "Invalid label format",
                "Issue Details": f"Expected format: VARIANT_TYPEOFTESTING_FEATURE_SUBFEATURE_FUNCTIONALITY, with VARIANT in {variants}, TYPEOFTESTING in {type_of_testing}, FEATURE in {features}, and FUNCTIONALITY in {functionalities}"
            })
        if re.search(r'[<>:"/\\|?*]', labels):
            errors.append({
                "Row": row_num,
                "Column": "Labels",
                "Cell": f"{col_to_letter['Labels']}{row_num}",
                "Value": labels,
                "Error": "Invalid characters in label",
                "Issue Details": "Labels must not contain special characters except underscores"
            })
    elif not labels:
        errors.append({
            "Row": row_num,
            "Column": "Labels",
            "Cell": f"{col_to_letter['Labels']}{row_num}",
            "Value": "",
            "Error": "Empty label",
            "Issue Details": "Labels column must not be empty"
        })
    
    # Validate Action
    action = row.get("Action", "")
    if isinstance(action, str) and action.strip():
        lines = action.split("\n")
        if not any("Steps:" in line for line in lines):
            errors.append({
                "Row": row_num,
                "Column": "Action",
                "Cell": f"{col_to_letter['Action']}{row_num}",
                "Value": action,
                "Error": "Missing Steps: header",
                "Issue Details": "Action must start with 'Steps:' followed by numbered steps"
            })
        else:
            extracting = False
            step_numbers = []
            for line in lines:
                if "Steps:" in line:
                    extracting = True
                    continue
                if extracting and line.strip():
                    # Match steps with optional suffix (i, a)
                    step_match = re.match(r"^(\d+)([ia])?\.\s*(.+)", line)
                    if not step_match:
                        errors.append({
                            "Row": row_num,
                            "Column": "Action",
                            "Cell": f"{col_to_letter['Action']}{row_num}",
                            "Value": line.strip(),
                            "Error": "Invalid step format",
                            "Issue Details": "Each step must start with a number and optional suffix [i,a] (e.g., '1i. text')"
                        })
                    else:
                        step_num = int(step_match.group(1))
                        suffix = step_match.group(2) or ""
                        clean_line = step_match.group(3).strip()
                        step_numbers.append(step_num)
                        
                        if suffix == "o":
                            errors.append({
                                "Row": row_num,
                                "Column": "Action",
                                "Cell": f"{col_to_letter['Action']}{row_num}",
                                "Value": line.strip(),
                                "Error": "Invalid suffix",
                                "Issue Details": "Action steps must only use suffixes 'i' or 'a', found 'o'"
                            })
                        
                        if clean_line:
                            if ":" in clean_line:
                                keyword, value = clean_line.split(":", 1)
                                keyword_clean = keyword.strip().lower()
                                value_clean = value.strip()
                                if not value_clean:
                                    errors.append({
                                        "Row": row_num,
                                        "Column": "Action",
                                        "Cell": f"{col_to_letter['Action']}{row_num}",
                                        "Value": clean_line,
                                        "Error": "Missing value after colon",
                                        "Issue Details": "Steps with a colon must have a value (e.g., 'keyword: value')"
                                    })
                            else:
                                keyword_clean = clean_line.strip().lower()
                                value_clean = ""
                            if keyword_clean not in keyword_set:
                                errors.append({
                                    "Row": row_num,
                                    "Column": "Action",
                                    "Cell": f"{col_to_letter['Action']}{row_num}",
                                    "Value": clean_line,
                                    "Error": "Unmapped keyword",
                                    "Issue Details": f"Keyword '{keyword_clean}' not found in keyword mapping"
                                })
            
            # Validate step number sequence
            if step_numbers:
                expected_sequence = list(range(1, len(step_numbers) + 1))
                if step_numbers != expected_sequence:
                    errors.append({
                        "Row": row_num,
                        "Column": "Action",
                        "Cell": f"{col_to_letter['Action']}{row_num}",
                        "Value": ", ".join(map(str, step_numbers)),
                        "Error": "Incorrect step sequence",
                        "Issue Details": f"Step numbers must be sequential starting from 1 (e.g., 1., 2., 3.). Found: {step_numbers}"
                    })
    elif not action:
        errors.append({
            "Row": row_num,
            "Column": "Action",
            "Cell": f"{col_to_letter['Action']}{row_num}",
            "Value": "",
            "Error": "Empty Action",
            "Issue Details": "Action column must not be empty unless intentionally blank"
        })
    
    # Validate Expected Results
    expected = row.get("Expected Results", "")
    if isinstance(expected, str) and expected.strip():
        lines = expected.split("\n")
        for line in lines:
            if line.strip():
                # Match steps with optional suffix (o, a)
                step_match = re.match(r"^(\d+)([oa])?\.\s*(.+)", line)
                if not step_match:
                    errors.append({
                        "Row": row_num,
                        "Column": "Expected Results",
                        "Cell": f"{col_to_letter['Expected Results']}{row_num}",
                        "Value": line.strip(),
                        "Error": "Invalid step format",
                        "Issue Details": "Each expected result must start with a number and optional suffix [o,a] (e.g., '1o. text')"
                    })
                else:
                    step_num = int(step_match.group(1))
                    suffix = step_match.group(2) or ""
                    clean_value = step_match.group(3).strip()
                    
                    if suffix == "i":
                        errors.append({
                            "Row": row_num,
                            "Column": "Expected Results",
                            "Cell": f"{col_to_letter['Expected Results']}{row_num}",
                            "Value": line.strip(),
                            "Error": "Invalid suffix",
                            "Issue Details": "Expected Results steps must only use suffixes 'o' or 'a', found 'i'"
                        })
                    
                    if clean_value:
                        if ":" in clean_value:
                            keyword, value = clean_value.split(":", 1)
                            keyword_clean = keyword.strip().lower()
                            value_clean = value.strip()
                            if not value_clean:
                                errors.append({
                                    "Row": row_num,
                                    "Column": "Expected Results",
                                    "Cell": f"{col_to_letter['Expected Results']}{row_num}",
                                    "Value": clean_value,
                                    "Error": "Missing value after colon",
                                    "Issue Details": "Expected results with a colon must have a value (e.g., 'keyword: value')"
                                })
                        else:
                            keyword_clean = clean_value.strip().lower()
                            value_clean = ""
                        if keyword_clean not in keyword_set:
                            errors.append({
                                "Row": row_num,
                                "Column": "Expected Results",
                                "Cell": f"{col_to_letter['Expected Results']}{row_num}",
                                "Value": clean_value,
                                "Error": "Unmapped keyword",
                                "Issue Details": f"Keyword '{keyword_clean}' not found in keyword mapping"
                            })
    elif not expected:
        errors.append({
            "Row": row_num,
            "Column": "Expected Results",
            "Cell": f"{col_to_letter['Expected Results']}{row_num}",
            "Value": "",
            "Error": "Empty Expected Results",
            "Issue Details": "Expected Results column must not be empty unless no verification is required"
        })
    
    return errors

def extract_steps(row, keyword_map):
    action_steps = []
    expected_steps = {}
    action_lines = []
    expected_results_str = ""
    
    if isinstance(row.get("Action", ""), float) and np.isnan(row.get("Action", "")):
        action_lines = []
    else:
        action_lines = str(row.get("Action", "")).split("\n")
    
    extracting = False
    for line in action_lines:
        if "Steps:" in line:
            extracting = True
            continue
        if extracting and line.strip():
            # Match steps with optional suffix (i, a)
            step_match = re.match(r"^(\d+)([ia])?\.\s*(.+)", line)
            if step_match:
                step_num = int(step_match.group(1))
                suffix = step_match.group(2) or ""
                clean_line = step_match.group(3).strip()
                action_steps.append((step_num, suffix, clean_line))
    
    expected_results = row.get("Expected Results", "")
    if isinstance(expected_results, float) and np.isnan(expected_results):
        expected_results_str = ""
    else:
        expected_results_str = str(expected_results)
    
    expected_lines = expected_results_str.split("\n")
    for line in expected_lines:
        if line.strip():
            # Match steps with optional suffix (o, a)
            step_match = re.match(r"^(\d+)([oa])?\.\s*(.+)", line)
            if step_match:
                step_num = int(step_match.group(1))
                suffix = step_match.group(2) or ""
                clean_value = step_match.group(3).strip()
                expected_steps[step_num] = (suffix, clean_value)
    
    final_steps = []
    used_expected = set()
    for step_num, suffix, action in action_steps:
        if suffix == "i":
            # Apply keyword mapping to the action text
            if ":" in action:
                keyword, value = action.split(":", 1)
                keyword_clean = keyword.strip().lower()
                value_clean = value.strip()
                replacement_keyword = keyword_map.get(keyword_clean, keyword_clean).strip()
                action_mapped = f"{replacement_keyword}: {value_clean}" if value_clean else replacement_keyword
            else:
                keyword_clean = action.strip().lower()
                action_mapped = keyword_map.get(keyword_clean, keyword_clean).strip()
            final_steps.append(f"show_dialog    input    {action_mapped}    20")
        elif suffix == "a" or not suffix:
            # Apply keyword mapping
            if ":" in action:
                keyword, value = action.split(":", 1)
                keyword_clean = keyword.strip().lower()
                value_clean = value.strip()
                replacement_keyword = keyword_map.get(keyword_clean, keyword_clean).strip()
                final_steps.append(f"{replacement_keyword}    {value_clean}" if value_clean else f"{replacement_keyword}")
            else:
                keyword_clean = action.strip().lower()
                replacement_keyword = keyword_map.get(keyword_clean, keyword_clean).strip()
                final_steps.append(replacement_keyword)
        
        if step_num in expected_steps:
            exp_suffix, exp_value = expected_steps[step_num]
            if exp_suffix == "o":
                # Apply keyword mapping to the expected value
                if ":" in exp_value:
                    keyword, value = exp_value.split(":", 1)
                    keyword_clean = keyword.strip().lower()
                    value_clean = value.strip()
                    replacement_keyword = keyword_map.get(keyword_clean, keyword_clean).strip()
                    exp_value_mapped = f"{replacement_keyword}: {value_clean}" if value_clean else replacement_keyword
                else:
                    keyword_clean = exp_value.strip().lower()
                    exp_value_mapped = keyword_map.get(keyword_clean, keyword_clean).strip()
                final_steps.append(f"show_dialog    output    {exp_value_mapped}    20")
            elif exp_suffix == "a" or not exp_suffix:
                # Apply keyword mapping
                if ":" in exp_value:
                    keyword, value = exp_value.split(":", 1)
                    keyword_clean = keyword.strip().lower()
                    value_clean = value.strip()
                    replacement_keyword = keyword_map.get(keyword_clean, keyword_clean).strip()
                    final_steps.append(f"{replacement_keyword}    {value_clean}" if value_clean else f"{replacement_keyword}")
                else:
                    keyword_clean = exp_value.strip().lower()
                    replacement_keyword = keyword_map.get(keyword_clean, keyword_clean).strip()
                    final_steps.append(replacement_keyword)
            used_expected.add(step_num)
    
    for key in sorted(expected_steps.keys()):
        if key not in used_expected:
            exp_suffix, exp_value = expected_steps[key]
            if exp_suffix == "o":
                # Apply keyword mapping
                if ":" in exp_value:
                    keyword, value = exp_value.split(":", 1)
                    keyword_clean = keyword.strip().lower()
                    value_clean = value.strip()
                    replacement_keyword = keyword_map.get(keyword_clean, keyword_clean).strip()
                    exp_value_mapped = f"{replacement_keyword}: {value_clean}" if value_clean else replacement_keyword
                else:
                    keyword_clean = exp_value.strip().lower()
                    exp_value_mapped = keyword_map.get(keyword_clean, keyword_clean).strip()
                final_steps.append(f"show_dialog    output    {exp_value_mapped}    20")
            elif exp_suffix == "a" or not exp_suffix:
                # Apply keyword mapping
                if ":" in exp_value:
                    keyword, value = exp_value.split(":", 1)
                    keyword_clean = keyword.strip().lower()
                    value_clean = value.strip()
                    replacement_keyword = keyword_map.get(keyword_clean, keyword_clean).strip()
                    final_steps.append(f"{replacement_keyword}    {value_clean}" if value_clean else f"{replacement_keyword}")
                else:
                    keyword_clean = exp_value.strip().lower()
                    replacement_keyword = keyword_map.get(keyword_clean, keyword_clean).strip()
                    final_steps.append(replacement_keyword)
    
    return final_steps

def generate_separate_robot_files(df, keyword_mapping_df=None, header_file_path=None, output_dir="generated_test_scripts"):
    keyword_map = {}
    if keyword_mapping_df is not None:
        keyword_map = dict(zip(
            keyword_mapping_df["TCD Keywords"].str.strip().str.lower(),
            keyword_mapping_df["Python Func Name"].str.strip()
        ))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if header_file_path:
        with open(header_file_path, "r", encoding="utf-8") as header_file:
            header_content = header_file.read().rstrip() + "\n\n"
    else:
        raise ValueError("Header file path must be provided.")
    
    def sanitize_filename(name):
        if not isinstance(name, str):
            return ""
        invalid_chars = r'[<>:"/\\|?*]+'
        sanitized = re.sub(invalid_chars, '_', name.strip())
        sanitized = re.sub(r'[_\s]+', '_', sanitized)
        return sanitized.strip('_')
    
    # Get user-defined functionalities and their prefixes from session state
    functionalities = st.session_state.get("functionalities", "").split(",") if st.session_state.get("functionalities", "") else []
    functionalities = [f.strip().lower() for f in functionalities if f.strip()]
    functionality_prefixes = st.session_state.get("functionality_prefixes", "").split(",") if st.session_state.get("functionality_prefixes", "") else []
    functionality_prefixes = [p.strip() for p in functionality_prefixes if p.strip()]
    
    # Create a mapping of functionality to prefix
    test_case_type_prefix = {}
    for func, prefix in zip(functionalities, functionality_prefixes):
        test_case_type_prefix[func] = prefix
    
    # Define the order of Test_Case_Type based on user input
    test_case_type_order = {func: i for i, func in enumerate(functionalities, 1)}
    
    # Generate separate .robot files as before
    grouped = df.groupby("Normalized_Feature")
    
    for feature_norm, feature_group in grouped:
        precondition_row = feature_group[feature_group["Test_Case_Type"] == "precondition"]
        other_tests = feature_group[feature_group["Test_Case_Type"] != "precondition"]
        
        precondition = precondition_row.iloc[0] if not precondition_row.empty else None
        precondition_steps = extract_steps(precondition, keyword_map) if precondition is not None else []
        precondition_issues = precondition.get("link issue Test", "").strip() if precondition is not None else ""
        # Safely handle precondition Description
        precondition_sum = precondition.get("Summary", "Precondition") if precondition is not None else "Precondition"
        precondition_sum = re.sub(r"\s+", " ", str(precondition_sum).strip())
        
        precondition_desc = precondition.get("Summary", "Precondition") if precondition is not None else "Precondition"
        precondition_desc = re.sub(r"\s+", " ", str(precondition_desc).strip())
        
        feature_dir = os.path.join(output_dir, feature_norm)
        os.makedirs(feature_dir, exist_ok=True)
        
        # Group by Test_Case_Type and sort according to the defined order
        category_groups = other_tests.groupby("Test_Case_Type")
        # Convert to list of (category, group) and sort
        sorted_groups = sorted(
            category_groups,
            key=lambda x: test_case_type_order.get(x[0].lower(), float('inf'))
        )
        
        for category, group_df in sorted_groups:
            sub_feature_tag = sanitize_filename(group_df.iloc[0]["Sub_Feature"].strip())
            feature_tag = sanitize_filename(group_df.iloc[0]["feature"].strip())
            
            # Use the new nomenclature: PREFIX_<FEATURE_TAG>_<TEST_CASE_TYPE>.robot
            category_lower = category.lower()
            prefix = test_case_type_prefix.get(category_lower, "TC0")  # Default prefix for unexpected types
            full_filename = f"{prefix}_{category.upper()}_{sub_feature_tag.upper()}_{feature_tag.upper()}.robot"       
            file_path = os.path.join(feature_dir, full_filename)
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(header_content)
                f.write("*** Test Cases ***\n\n")
                tc_index = 1
                
                if precondition is not None:
                    f.write(f"TC{tc_index:03d}: [SYS5] {precondition_sum}\n")
                    f.write(f"    [Documentation]    REQ_{precondition_issues}\n")
                    f.write(f"    [Tags]    {feature_tag}\n")
                    f.write(f"    [Description]    {precondition_desc}\n")
                    f.write(f"    [Feature]    {sub_feature_tag}\n")
                    f.write(f"    [Feature_group]   {sub_feature_tag}_precondition\n\n")
                    
                    for step in precondition_steps:
                        f.write(f"    {step}\n")
                    
                    # Append Set Normal Condition as the last step
                    f.write(f"    Set Normal Condition\n")
                    f.write(f"    Sleep    20\n")
                    f.write("\n")
                    tc_index += 1
                
                for _, row in group_df.iterrows():
                    # Safely handle Description, converting to string and handling NaN
                    description = row.get("Description", "Unnamed Test Case")
                    Summary = row.get("Summary", "Unnamed Test Case")
                    Summary_f = re.sub(r"\s+", " ", str(Summary).strip())
                    test_name = re.sub(r"\s+", " ", str(description).strip())
                    linked_issues = row.get("link issue Test", "").strip()
                    full_group = f"{feature_tag}_{category.lower()}"
                    f.write(f"TC{tc_index:03d}: [SYS5] {Summary_f}\n")
                    f.write(f"    [Documentation]   REQ_{linked_issues}\n")
                    f.write(f"    [Tags]    {feature_tag}\n")
                    f.write(f"    [Description]    {test_name}\n")
                    f.write(f"    [Feature]    {sub_feature_tag}\n")
                    f.write(f"    [Feature_group]    {full_group}\n\n")
                    tc_index += 1
                    
                    steps = extract_steps(row, keyword_map)
                    for step in steps:
                        f.write(f"    {step}\n")
                    
                    # Append Set Normal Condition
                    f.write(f"    Set Normal Condition\n")
                    f.write(f"    Sleep    20\n")
                    f.write("\n")
    
        # Generate manual .robot files for this feature
        manual_df = feature_group[feature_group["Planned Execution"].str.lower() == "manual"]
        if not manual_df.empty:
            manual_sub_feature_grouped = manual_df.groupby("Sub_Feature")
            for sub_feature, sub_feature_group in manual_sub_feature_grouped:
                sub_feature_tag = sanitize_filename(sub_feature.strip())
                manual_file_path = os.path.join(feature_dir, f"MANUAL_{sub_feature_tag}.robot")
                with open(manual_file_path, "w", encoding="utf-8") as f:
                    f.write(header_content)
                    f.write("*** Test Cases ***\n\n")
                    tc_index = 1
                    
                    # Handle precondition for this sub-feature
                    precondition_row = sub_feature_group[sub_feature_group["Test_Case_Type"] == "precondition"]
                    other_tests = sub_feature_group[sub_feature_group["Test_Case_Type"] != "precondition"]
                    
                    precondition = precondition_row.iloc[0] if not precondition_row.empty else None
                    precondition_steps = extract_steps(precondition, keyword_map) if precondition is not None else []
                    precondition_issues = precondition.get("link issue Test", "").strip() if precondition is not None else ""
                    precondition_sum = precondition.get("Summary", "Precondition") if precondition is not None else "Precondition"
                    precondition_sum = re.sub(r"\s+", " ", str(precondition_sum).strip())
                    precondition_desc = precondition.get("Summary", "Precondition") if precondition is not None else "Precondition"
                    precondition_desc = re.sub(r"\s+", " ", str(precondition_desc).strip())
                    feature_tag = sanitize_filename(sub_feature_group.iloc[0]["feature"].strip())
                    
                    if precondition is not None:
                        f.write(f"TC{tc_index:03d}: [SYS5] {precondition_sum}\n")
                        f.write(f"    [Documentation]    REQ_{precondition_issues}\n")
                        f.write(f"    [Tags]    {feature_tag}\n")
                        f.write(f"    [Description]    {precondition_desc}\n")
                        f.write(f"    [Feature]    {sub_feature_tag}\n")
                        f.write(f"    [Feature_group]   {sub_feature_tag}_precondition\n\n")
                        
                        for step in precondition_steps:
                            f.write(f"    {step}\n")
                        
                        f.write(f"    Set Normal Condition\n")
                        f.write(f"    Sleep    20\n")
                        f.write("\n")
                        tc_index += 1
                    
                    for _, row in other_tests.iterrows():
                        description = row.get("Description", "Unnamed Test Case")
                        Summary = row.get("Summary", "Unnamed Test Case")
                        Summary_f = re.sub(r"\s+", " ", str(Summary).strip())
                        test_name = re.sub(r"\s+", " ", str(description).strip())
                        linked_issues = row.get("link issue Test", "").strip()
                        category = row["Test_Case_Type"].lower()
                        full_group = f"{feature_tag}_{category}"
                        
                        f.write(f"TC{tc_index:03d}: [SYS5] {Summary_f}\n")
                        f.write(f"    [Documentation]   REQ_{linked_issues}\n")
                        f.write(f"    [Tags]    {feature_tag}\n")
                        f.write(f"    [Description]    {test_name}\n")
                        f.write(f"    [Feature]    {sub_feature_tag}\n")
                        f.write(f"    [Feature_group]    {full_group}\n\n")
                        tc_index += 1
                        
                        steps = extract_steps(row, keyword_map)
                        for step in steps:
                            f.write(f"    {step}\n")
                        
                        f.write(f"    Set Normal Condition\n")
                        f.write(f"    Sleep    20\n")
                        f.write("\n")
    
    return output_dir
