import streamlit as st
import pandas as pd
from io import BytesIO
import zipfile
import os
import regex as re
import numpy as np

def load_and_preprocess_tcd(tcd_filepath, keyword_mapping_df):
    # Load the original TCD without any preprocessing
    original_df = pd.read_excel(tcd_filepath)
    
    # Validate the original DataFrame to preserve row indices
    required_columns = ["Labels", "Action", "Expected Results", "Description", "link issue Test"]
    col_to_letter_col = list(original_df.columns)
    error_list = []
    keyword_set = set(keyword_mapping_df["TCD Keywords"].str.strip().str.lower())
    
    # Map column names to Excel column letters
    col_to_letter = {col: chr(65 + i) for i, col in enumerate(col_to_letter_col)}  # B=Labels, C=Action, D=Expected Results
    
    for idx, row in original_df.iterrows():
        # Excel row = DataFrame index + 2 (header row in Excel)
        excel_row = idx + 2
        errors = validate_tcd_row(row, keyword_set, excel_row, col_to_letter)
        error_list.extend(errors)
    
    error_df = pd.DataFrame(error_list, columns=["Row", "Column", "Cell", "Value", "Error", "Issue Details"])
    
    # Now preprocess the DataFrame for script generation
    df = original_df[required_columns].copy()
    
    # Filter out completely empty rows
    df = df.dropna(how='all')
    
    # Filter out rows where only 'Labels' has a value
    df = df.dropna(subset=['Action', 'Expected Results', 'Description'], how='all')
    
    # Reset index after filtering
    df = df.reset_index(drop=True)
    
    # Apply preprocessing to ensure all columns are strings and handle NaN
    df["Labels"] = df["Labels"].fillna("").astype(str).str.strip()
    df["Action"] = df["Action"].fillna("").astype(str).str.strip()
    df["Expected Results"] = df["Expected Results"].fillna("").astype(str).str.strip()
    df["Description"] = df["Description"].fillna("Unnamed Test Case").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["link issue Test"] = df["link issue Test"].fillna("").astype(str).str.replace(r"[\n\r;,\s]+", "_", regex=True).str.strip("_")
    
    # Derive additional columns
    df["Test_Case_Type"] = df["Labels"].apply(lambda x: x.split("_")[-1].strip().lower().replace(" ", "") if x and "_" in x else "")
    df["Sub_Feature"] = df["Labels"].apply(lambda x: "_".join(x.split("_")[2:-1]).strip() if x and len(x.split("_")) > 3 else "")
    
    def sanitize_filename(name):
        if not isinstance(name, str) or not name:
            return "unknown_feature"
        invalid_chars = r'[<>:"/\\|?*]+'
        sanitized = re.sub(invalid_chars, '_', name.strip().lower())
        sanitized = re.sub(r'[_\s]+', '_', sanitized)
        return sanitized.strip('_') or "unknown_feature"
    
    df["Normalized_Feature"] = df["Sub_Feature"].apply(sanitize_filename)
    
    return df, error_df

def validate_tcd_row(row, keyword_set, row_num, col_to_letter):
    errors = []
    valid_test_types = {
        "Precondition", "LogicalCombination", "FailureMode", "PowerMode", "Configuration", "VoltageMode"
    }
    
    # Helper function to check if value is valid string
    def is_valid_string(value):
        return isinstance(value, str) and value.strip()
    
    # Validate Labels
    labels = row.get("Labels", "")
    if pd.isna(labels):
        errors.append({
            "Row": row_num,
            "Column": "Labels",
            "Cell": f"{col_to_letter.get('Labels', 'B')}{row_num}",
            "Value": "NaN",
            "Error": "Missing Labels",
            "Issue Details": "Labels column must not be empty or NaN"
        })
    elif not is_valid_string(labels):
        errors.append({
            "Row": row_num,
            "Column": "Labels",
            "Cell": f"{col_to_letter.get('Labels', 'B')}{row_num}",
            "Value": str(labels),
            "Error": "Invalid Labels",
            "Issue Details": f"Labels must be a non-empty string, got {type(labels).__name__}"
        })
    else:
        pattern = re.compile(
            r"^[W616\]_FV_((Alert|TT|Chime)_)?[A-Z_]+_(" +
            "|".join(valid_test_types) + ")$",
            re.IGNORECASE
        )
        if not pattern.match(labels):
            errors.append({
                "Row": row_num,
                "Column": "Labels",
                "Cell": f"{col_to_letter.get('Labels', 'B')}{row_num}",
                "Value": labels,
                "Error": "Invalid label format",
                "Issue Details": f"Expected format: PROJECT[ID]_FV_[TYPEOFFEATURE]_FEATURENAME_TESTCASETYPE, with TYPEOFFEATURE in {{Alert, TT, Chime}} (optional) and TESTCASETYPE in {valid_test_types}"
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
    
    # Validate Action
    action = row.get("Action", "")
    if pd.isna(action):
        errors.append({
            "Row": row_num,
            "Column": "Action",
            "Cell": f"{col_to_letter.get('Action', 'C')}{row_num}",
            "Value": "NaN",
            "Error": "Missing Action",
            "Issue Details": "Action column must not be empty or NaN unless intentionally blank"
        })
    elif not is_valid_string(action):
        errors.append({
            "Row": row_num,
            "Column": "Action",
            "Cell": f"{col_to_letter.get('Action', 'C')}{row_num}",
            "Value": str(action),
            "Error": "Invalid Action",
            "Issue Details": f"Action must be a non-empty string, got {type(action).__name__}"
        })
    else:
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
                        if keyword_clean and keyword_clean not in keyword_set:
                            errors.append({
                                "Row": row_num,
                                "Column": "Action",
                                "Cell": f"{col_to_letter.get('Action', 'C')}{row_num}",
                                "Value": clean_line,
                                "Error": "Unmapped keyword",
                                "Issue Details": f"Keyword '{keyword_clean}' not found in keyword mapping"
                            })
    
    # Validate Expected Results
    expected = row.get("Expected Results", "")
    if pd.isna(expected):
        errors.append({
            "Row": row_num,
            "Column": "Expected Results",
            "Cell": f"{col_to_letter.get('Expected Results', 'D')}{row_num}",
            "Value": "NaN",
            "Error": "Missing Expected Results",
            "Issue Details": "Expected Results column must not be empty or NaN unless no verification is required"
        })
    elif not is_valid_string(expected):
        errors.append({
            "Row": row_num,
            "Column": "Expected Results",
            "Cell": f"{col_to_letter.get('Expected Results', 'D')}{row_num}",
            "Value": str(expected),
            "Error": "Invalid Expected Results",
            "Issue Details": f"Expected Results must be a non-empty string, got {type(expected).__name__}"
        })
    else:
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
                    if keyword_clean and keyword_clean not in keyword_set:
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
    
    # Validate Description
    description = row.get("Description", "")
    if pd.isna(description):
        errors.append({
            "Row": row_num,
            "Column": "Description",
            "Cell": f"{col_to_letter.get('Description', 'E')}{row_num}",
            "Value": "NaN",
            "Error": "Missing Description",
            "Issue Details": "Description column must not be empty or NaN"
        })
    elif not is_valid_string(description):
        errors.append({
            "Row": row_num,
            "Column": "Description",
            "Cell": f"{col_to_letter.get('Description', 'E')}{row_num}",
            "Value": str(description),
            "Error": "Invalid Description",
            "Issue Details": f"Description must be a non-empty string, got {type(description).__name__}"
        })
    
    # Validate link issue Test
    link_issue = row.get("link issue Test", "")
    if pd.isna(link_issue):
        errors.append({
            "Row": row_num,
            "Column": "link issue Test",
            "Cell": f"{col_to_letter.get('link issue Test', 'F')}{row_num}",
            "Value": "NaN",
            "Error": "Missing link issue Test",
            "Issue Details": "link issue Test column must not be empty or NaN"
        })
    elif not is_valid_string(link_issue):
        errors.append({
            "Row": row_num,
            "Column": "link issue Test",
            "Cell": f"{col_to_letter.get('link issue Test', 'F')}{row_num}",
            "Value": str(link_issue),
            "Error": "Invalid link issue Test",
            "Issue Details": f"link issue Test must be a non-empty string, got {type(link_issue).__name__}"
        })
    
    return errors

def extract_steps(row):
    action_steps = []
    expected_steps = {}
    
    # Safely handle Action
    action = row.get("Action", "")
    if pd.isna(action) or not isinstance(action, str):
        action_lines = []
    else:
        action_lines = action.split("\n")
    
    extracting = False
    for line in action_lines:
        if "Steps:" in line:
            extracting = True
            continue
        if extracting and line.strip():
            clean_line = re.sub(r"^\d+\.\s*", "", line.strip())
            action_steps.append(clean_line)
    
    # Safely handle Expected Results
    expected_results = row.get("Expected Results", "")
    if pd.isna(expected_results) or not isinstance(expected_results, str):
        expected_lines = []
    else:
        expected_lines = expected_results.split("\n")
    
    for line in expected_lines:
        if line.strip() and re.match(r"^\d+\.", line):
            parts = line.split(".", 1)
            try:
                step_num = int(parts[0].strip())
                clean_value = parts[1].strip()
                expected_steps[step_num] = clean_value
            except (ValueError, IndexError):
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
        if not isinstance(name, str) or not name:
            return "unknown_feature"
        invalid_chars = r'[<>:"/\\|?*]+'
        sanitized = re.sub(invalid_chars, '_', name.strip())
        sanitized = re.sub(r'[_\s]+', '_', sanitized)
        return sanitized.strip('_') or "unknown_feature"
    
    grouped = df.groupby("Normalized_Feature")
    
    for feature_norm, feature_group in grouped:
        precondition_row = feature_group[feature_group["Test_Case_Type"] == "precondition"]
        other_tests = feature_group[feature_group["Test_Case_Type"] != "precondition"]
        
        precondition = precondition_row.iloc[0] if not precondition_row.empty else None
        precondition_steps = extract_steps(precondition) if precondition is not None else []
        precondition_issues = str(precondition.get("link issue Test", "")).strip() if precondition is not None else ""
        precondition_desc = str(precondition.get("Description", "Precondition")).strip() if precondition is not None else "Precondition"
        precondition_desc = re.sub(r"\s+", " ", precondition_desc) if precondition_desc else "Precondition"
        
        feature_dir = os.path.join(output_dir, feature_norm)
        os.makedirs(feature_dir, exist_ok=True)
        
        category_groups = other_tests.groupby("Test_Case_Type")
        
        for category, group_df in category_groups:
            feature_tag = sanitize_filename(str(group_df.iloc[0]["Sub_Feature"]).strip())
            full_filename = f"{feature_tag.upper()}_{category.upper()}.robot"
            file_path = os.path.join(feature_dir, full_filename)
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(header_content)
                f.write("*** Test Cases ***\n\n")
                tc_index = 1
                
                if precondition is not None:
                    f.write(f"TC{tc_index:03d}: [SYS5] {precondition_desc}\n")
                    f.write(f"    [Documentation]    REQ_{precondition_issues}\n")
                    f.write(f"    [Tags]    {feature_tag}\n")
                    f.write(f"    [Description]    {precondition_desc}\n")
                    f.write(f"    [Feature]    {feature_tag}\n")
                    f.write(f"    [Feature_group]   {feature_tag}_precondition\n\n")
                    
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
                    
                    f.write("\n")
                    tc_index += 1
                
                for _, row in group_df.iterrows():
                    description = str(row.get("Description", "Unnamed Test Case")).strip()
                    test_name = re.sub(r"\s+", " ", description) if description else "Unnamed Test Case"
                    linked_issues = str(row.get("link issue Test", "")).strip()
                    full_group = f"{feature_tag}_{category.lower()}"
                    f.write(f"TC{tc_index:03d}: [SYS5] {test_name}\n")
                    f.write(f"    [Documentation]    REQ_{linked_issues}\n")
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
                    
                    f.write("\n")
    
    return output_dir
