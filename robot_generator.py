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
    required_columns = ["Issue ID", "Labels", "Action", "Expected Results", "Description", "link issue Test", "Planned Execution", "Summary"]
    col_to_letter_col = list(original_df.columns)
    error_list = []
    keyword_set = set(keyword_mapping_df["TCD Keywords"].str.strip().str.lower())
    
    # Map column names to Excel column letters
    col_to_letter = {col: chr(65 + i) for i, col in enumerate(col_to_letter_col)}
    
    # Get user-defined label components from session state
    variants = st.session_state.get("variants", "").split(",") if st.session_state.get("variants", "") else []
    variants = [v.strip() for v in variants if v.strip()] or [r"[A-Za-z0-9]+"]
    type_of_testing = st.session_state.get("type_of_testing", "").split(",") if st.session_state.get("type_of_testing", "") else []
    type_of_testing = [t.strip() for t in type_of_testing if t.strip()] or [r"[A-Za-z0-9]+"]
    features = st.session_state.get("features", "").split(",") if st.session_state.get("features", "") else []
    features = [f.strip() for f in features if f.strip()] or [r"[A-Za-z0-9]+"]
    sub_features = st.session_state.get("sub_features", "").split(",") if st.session_state.get("sub_features", "") else []
    sub_features = [s.strip() for s in sub_features if s.strip()] or [r"[A-Za-z0-9_]+"]
    functionalities = st.session_state.get("functionalities", "").split(",") if st.session_state.get("functionalities", "") else []
    functionalities = [f.strip() for f in functionalities if f.strip()] or [r"[A-Za-z0-9]+"]
    
    # Check for duplicate Issue IDs
    issue_ids = original_df["Issue ID"].astype(str).str.strip()
    issue_id_counts = issue_ids.value_counts()
    duplicates = issue_id_counts[issue_id_counts > 1].index.tolist()
    if "" in duplicates:
        duplicates.remove("")  # Empty strings handled separately
    
    for idx, row in original_df.iterrows():
        excel_row = idx + 2  # Excel row = DataFrame index + 2 (header row)
        errors = validate_tcd_row(row, keyword_set, excel_row, col_to_letter, variants, type_of_testing, features, sub_features, functionalities, duplicates)
        error_list.extend(errors)
    
    # Check Issue ID sequence (unique and sequential starting from 1)
    try:
        issue_ids_numeric = pd.to_numeric(issue_ids, errors='coerce')
        seen_ids = set()
        for idx, issue_id in enumerate(issue_ids_numeric):
            excel_row = idx + 2
            if pd.isna(issue_id):
                continue  # Handled in validate_tcd_row
            if issue_id in seen_ids:
                continue  # Duplicates handled in validate_tcd_row
            seen_ids.add(issue_id)
            expected_id = len(seen_ids)
            if issue_id != expected_id:
                error_list.append({
                    "Row": excel_row,
                    "Column": "Issue ID",
                    "Cell": f"{col_to_letter['Issue ID']}{excel_row}",
                    "Value": str(issue_id),
                    "Error": "Incorrect Issue ID sequence",
                    "Issue Details": f"Issue ID should be sequential starting from 1 (e.g., 1, 2, 3...). Expected {expected_id}, found {issue_id}"
                })
    except Exception as e:
        error_list.append({
            "Row": "N/A",
            "Column": "Issue ID",
            "Cell": "N/A",
            "Value": "N/A",
            "Error": "Sequence validation error",
            "Issue Details": f"Failed to validate Issue ID sequence: {str(e)}"
        })
    
    error_df = pd.DataFrame(error_list, columns=["Row", "Column", "Cell", "Value", "Error", "Issue Details"])
    
    # Preprocess DataFrame for script generation
    df = original_df.copy()
    
    # Filter rows where "Planned Execution" is "Planned Automation" (case-insensitive)
    df = df[df["Planned Execution"].str.lower() == "planned automation"]
    
    # If no rows match, return empty DataFrame
    if df.empty:
        df = pd.DataFrame(columns=required_columns)
    else:
        # Keep only required columns
        df = df[required_columns]
        
        # Filter out completely empty rows
        df = df.dropna(how='all')
        
        # Filter out rows where only 'Labels' has a value
        df = df.dropna(subset=['Action', 'Expected Results', 'Description', 'Summary'], how='all')
        
        # Reset index
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
        df["Description"] = df["Description"].fillna("Unnamed Test Case").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        df["Summary"] = df["Summary"].fillna("Unnamed Test Case").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    
    return df, error_df

def validate_tcd_row(row, keyword_set, row_num, col_to_letter, variants, type_of_testing, features, sub_features, functionalities, duplicates):
    errors = []
    
    # Validate Issue ID
    issue_id = row.get("Issue ID", "")
    issue_id_str = str(issue_id).strip()
    if not issue_id_str:
        errors.append({
            "Row": row_num,
            "Column": "Issue ID",
            "Cell": f"{col_to_letter.get('Issue ID', 'A')}{row_num}",
            "Value": "",
            "Error": "Empty Issue ID",
            "Issue Details": "Issue ID column must not be empty"
        })
    else:
        try:
            issue_id_num = float(issue_id_str)
            if not issue_id_num.is_integer() or issue_id_num <= 0:
                errors.append({
                    "Row": row_num,
                    "Column": "Issue ID",
                    "Cell": f"{col_to_letter.get('Issue ID', 'A')}{row_num}",
                    "Value": issue_id_str,
                    "Error": "Invalid Issue ID",
                    "Issue Details": "Issue ID must be a positive integer"
                })
        except (ValueError, TypeError):
            errors.append({
                "Row": row_num,
                "Column": "Issue ID",
                "Cell": f"{col_to_letter.get('Issue ID', 'A')}{row_num}",
                "Value": issue_id_str,
                "Error": "Invalid Issue ID",
                "Issue Details": "Issue ID must be a numeric value"
            })
        if issue_id_str in duplicates:
            errors.append({
                "Row": row_num,
                "Column": "Issue ID",
                "Cell": f"{col_to_letter.get('Issue ID', 'A')}{row_num}",
                "Value": issue_id_str,
                "Error": "Duplicate Issue ID",
                "Issue Details": f"Issue ID '{issue_id_str}' is repeated in the TCD"
            })
    
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
                "Cell": f"{col_to_letter.get('Labels', 'B')}{row_num}",
                "Value": labels,
                "Error": "Invalid label format",
                "Issue Details": f"Expected format: VARIANT_TYPEOFTESTING_FEATURE_SUBFEATURE_FUNCTIONALITY"
            })
        if re.search(r'[<>:"/\\|?*]', labels):
            errors.append({
                "Row": row_num,
                "Column": "Labels",
                "Cell": f"{col_to_letter.get('Labels', 'B')}{row_num}",
                "Value": labels,
                "Error": "Invalid characters in label",
                "Issue Details": "Labels must not contain special characters except underscores"
            })
    elif not labels:
        errors.append({
            "Row": row_num,
            "Column": "Labels",
            "Cell": f"{col_to_letter.get('Labels', 'B')}{row_num}",
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
                "Cell": f"{col_to_letter.get('Action', 'C')}{row_num}",
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
                    if not re.match(r"^\d+\.\s*", line):
                        errors.append({
                            "Row": row_num,
                            "Column": "Action",
                            "Cell": f"{col_to_letter.get('Action', 'C')}{row_num}",
                            "Value": line.strip(),
                            "Error": "Missing step number",
                            "Issue Details": "Each step must start with a number (e.g., '1.')"
                        })
                    else:
                        step_num = int(re.match(r"^(\d+)\.\s*", line).group(1))
                        step_numbers.append(step_num)
                        clean_line = re.sub(r"^\d+\.\s*", "", line.strip())
                        if ":" in clean_line:
                            keyword, value = clean_line.split(":", 1)
                            keyword_clean = keyword.strip().lower()
                            value_clean = value.strip()
                            if not value_clean:
                                errors.append({
                                    "Row": row_num,
                                    "Column": "Action",
                                    "Cell": f"{col_to_letter.get('Action', 'C')}{row_num}",
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
                                "Cell": f"{col_to_letter.get('Action', 'C')}{row_num}",
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
                        "Cell": f"{col_to_letter.get('Action', 'C')}{row_num}",
                        "Value": ", ".join(map(str, step_numbers)),
                        "Error": "Incorrect step sequence",
                        "Issue Details": f"Step numbers must be sequential starting from 1 (e.g., 1., 2., 3.). Found: {step_numbers}"
                    })
    elif not action:
        errors.append({
            "Row": row_num,
            "Column": "Action",
            "Cell": f"{col_to_letter.get('Action', 'C')}{row_num}",
            "Value": "",
            "Error": "Empty Action",
            "Issue Details": "Action column must not be empty unless intentionally blank"
        })
    
    # Validate Expected Results
    expected = row.get("Expected Results", "")
    if isinstance(expected, str) and expected.strip():
        lines = expected.split("\n")
        for line in lines:
            if line.strip() and re.match(r"^\d+\.", line):
                parts = line.split(".", 1)
                try:
                    clean_value = parts[1].strip()
                    if ":" in clean_value:
                        keyword, value = clean_value.split(":", 1)
                        keyword_clean = keyword.strip().lower()
                        value_clean = value.strip()
                        if not value_clean:
                            errors.append({
                                "Row": row_num,
                                "Column": "Expected Results",
                                "Cell": f"{col_to_letter.get('Expected Results', 'D')}{row_num}",
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
                            "Cell": f"{col_to_letter.get('Expected Results', 'D')}{row_num}",
                            "Value": clean_value,
                            "Error": "Unmapped keyword",
                            "Issue Details": f"Keyword '{keyword_clean}' not found in keyword mapping"
                        })
                except ValueError:
                    errors.append({
                        "Row": row_num,
                        "Column": "Expected Results",
                        "Cell": f"{col_to_letter.get('Expected Results', 'D')}{row_num}",
                        "Value": line.strip(),
                        "Error": "Invalid step format",
                        "Issue Details": "Expected Results must have numbered steps (e.g., '1. Verifyscreen: VALUE')"
                    })
            elif line.strip():
                errors.append({
                    "Row": row_num,
                    "Column": "Expected Results",
                    "Cell": f"{col_to_letter.get('Expected Results', 'D')}{row_num}",
                    "Value": line.strip(),
                    "Error": "Missing step number",
                    "Issue Details": "Each expected result must start with a number (e.g., '1.')"
                })
    elif not expected:
        errors.append({
            "Row": row_num,
            "Column": "Expected Results",
            "Cell": f"{col_to_letter.get('Expected Results', 'D')}{row_num}",
            "Value": "",
            "Error": "Empty Expected Results",
            "Issue Details": "Expected Results column must not be empty unless no verification is required"
        })
    
    return errors



def extract_steps(row):
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
            clean_line = re.sub(r"^\d+\.\s*", "", line.strip())
            action_steps.append(clean_line)
    
    expected_results = row.get("Expected Results", "")
    if isinstance(expected_results, float) and np.isnan(expected_results):
        expected_results_str = ""
    else:
        expected_results_str = str(expected_results)
    
    expected_lines = expected_results_str.split("\n")
    for line in expected_lines:
        if line.strip() and re.match(r"^\d+\.", line):
            parts = line.split(".", 1)
            try:
                step_num = int(parts[0].strip())
                clean_value = parts[1].strip()
                expected_steps[step_num] = clean_value
            except ValueError:
                continue
    
    final_steps = []
    used_expected = set()
    for i, action in enumerate(action_steps, start=1):
        final_steps.append(action)
        if i in expected_steps:
            final_steps.append(expected_steps[i])
            used_expected.add(i)
    
    for key in sorted(expected_steps.keys()):
        if key not in used_expected:
            final_steps.append(expected_steps[key])
    
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
    
    grouped = df.groupby("Normalized_Feature")
    
    for feature_norm, feature_group in grouped:
        precondition_row = feature_group[feature_group["Test_Case_Type"] == "precondition"]
        other_tests = feature_group[feature_group["Test_Case_Type"] != "precondition"]
        
        precondition = precondition_row.iloc[0] if not precondition_row.empty else None
        precondition_steps = extract_steps(precondition) if precondition is not None else []
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
                    f.write(f"TC{tc_index:03d}: [SYS5] {precondition_sum}\n")
                    f.write(f"    [Feature]    {sub_feature_tag}\n")
                    f.write(f"    [Feature_group]   {sub_feature_tag}_precondition\n\n")
                    
                    for step in precondition_steps:
                        clean_step = re.sub(r"^\d+\.\s*", "", step.strip())
                        if ":" in clean_step:
                            keyword, value = clean_step.split(":", 1)
                            keyword_clean = keyword.strip().lower()
                            value_clean = value.strip()
                        else:
                            keyword_clean = clean_step.strip().lower()
                            value_clean = ""
                        
                        replacement_keyword = keyword_map.get(keyword_clean, keyword_clean).strip()
                        
                        if "do" in replacement_keyword.lower() or not value_clean:
                            f.write(f"    {replacement_keyword}\n")
                        else:
                            f.write(f"    {replacement_keyword}    {value_clean}\n")
                    
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
                    f.write(f"    [Documentation]   REQ_{linked_issues}\n")
                    f.write(f"    [Tags]    {feature_tag}\n")
                    f.write(f"    [Description]    {test_name}\n")
                    f.write(f"    [Feature]    {feature_tag}\n")
                    f.write(f"    [Feature_group]    {full_group}\n\n")
                    tc_index += 1
                    
                    steps = extract_steps(row)
                    for step in steps:
                        clean_step = re.sub(r"^\d+\.\s*", "", step.strip())
                        if ":" in clean_step:
                            keyword, value = clean_step.split(":", 1)
                            keyword_clean = keyword.strip().lower()
                            value_clean = value.strip()
                        else:
                            keyword_clean = clean_step.strip().lower()
                            value_clean = ""
                        
                        replacement_keyword = keyword_map.get(keyword_clean, keyword_clean).strip()
                        
                        if "do" in replacement_keyword.lower() or not value_clean:
                            f.write(f"    {replacement_keyword}\n")
                        else:
                            f.write(f"    {replacement_keyword}    {value_clean}\n")
                    
                    # Append Set Normal Condition as the last step
                    f.write(f"    Set Normal Condition\n")
                    f.write(f"    Sleep    20\n")
                    
                    f.write("\n")
    
    return output_dir
