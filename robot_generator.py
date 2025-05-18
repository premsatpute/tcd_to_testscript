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
    
    # Apply preprocessing
    df["Test_Case_Type"] = df["Labels"].apply(lambda x: x.split("_")[-1].strip().lower().replace(" ", "") if isinstance(x, str) else "")
    df["Sub_Feature"] = df["Labels"].apply(lambda x: "_".join(x.split("_")[2:-1]).strip() if isinstance(x, str) else "")
    
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
    
    return df, error_df

def validate_tcd_row(row, keyword_set, row_num, col_to_letter):
    errors = []
    valid_test_types = {
        "Precondition", "LogicalCombination", "FailureMode", "PowerMode", "Configuration", "VoltageMode"
    }
    
    # Validate Labels
    labels = row.get("Labels", "")
    if isinstance(labels, str) and labels.strip():
        pattern = re.compile(
            r"^[W616\]_FV_((Alert|TT|Chime)_)?[A-Z_]+_(" +
            "|".join(valid_test_types) + ")$",
            re.IGNORECASE
        )
        if not pattern.match(labels):
            errors.append({
                "Row": row_num,
                "Column": "Labels",
                "Cell": f"{col_to_letter['Labels']}{row_num}",
                "Value": labels,
                "Error": "Invalid label format",
                "Issue Details": f"Expected format: PROJECT[ID]_FV_[TYPEOFFEATURE]_FEATURENAME_TESTCASETYPE, with TYPEOFFEATURE in {{Alert, TT, Chime}} (optional) and TESTCASETYPE in {valid_test_types}"
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
                    if not re.match(r"^\d+\.\s*", line):
                        errors.append({
                            "Row": row_num,
                            "Column": "Action",
                            "Cell": f"{col_to_letter['Action']}{row_num}",
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
                except ValueError:
                    errors.append({
                        "Row": row_num,
                        "Column": "Expected Results",
                        "Cell": f"{col_to_letter['Expected Results']}{row_num}",
                        "Value": line.strip(),
                        "Error": "Invalid step format",
                        "Issue Details": "Expected Results must have numbered steps (e.g., '1. Verifyscreen: VALUE')"
                    })
            elif line.strip():
                errors.append({
                    "Row": row_num,
                    "Column": "Expected Results",
                    "Cell": f"{col_to_letter['Expected Results']}{row_num}",
                    "Value": line.strip(),
                    "Error": "Missing step number",
                    "Issue Details": "Each expected result must start with a number (e.g., '1.')"
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
    
    # Define the desired order of Test_Case_Type
    test_case_type_order = {
        "logicalcombination": 1,
        "failuremode": 2,
        "powermode": 3,
        "voltagemode": 4,
        "configuration": 5
    }
    
    # Define filename prefixes for each Test_Case_Type
    test_case_type_prefix = {
        "logicalcombination": "TC00101",
        "failuremode": "TC00201",
        "powermode": "TC00301",
        "voltagemode": "TC00401",
        "configuration": "TC00501"
    }
    
    grouped = df.groupby("Normalized_Feature")
    
    for feature_norm, feature_group in grouped:
        precondition_row = feature_group[feature_group["Test_Case_Type"] == "precondition"]
        other_tests = feature_group[feature_group["Test_Case_Type"] != "precondition"]
        
        precondition = precondition_row.iloc[0] if not precondition_row.empty else None
        precondition_steps = extract_steps(precondition) if precondition is not None else []
        precondition_issues = precondition.get("link issue Test", "").strip() if precondition is not None else ""
        # Safely handle precondition Description
        precondition_desc = precondition.get("Description", "Precondition") if precondition is not None else "Precondition"
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
            feature_tag = sanitize_filename(group_df.iloc[0]["Sub_Feature"].strip())
            # Use the new nomenclature: TC<identifier>_<FEATURE_TAG>_<TEST_CASE_TYPE>.robot
            category_lower = category.lower()
            prefix = test_case_type_prefix.get(category_lower, "TC00000")  # Default prefix for unexpected types
            full_filename = f"{prefix}_{feature_tag.upper()}_{category.upper()}.robot"
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
                    
                    # Append Set Normal Condition as the last step
                    f.write(f"    Set Normal Condition\n")
                    
                    f.write("\n")
                    tc_index += 1
                
                for _, row in group_df.iterrows():
                    # Safely handle Description, converting to string and handling NaN
                    description = row.get("Description", "Unnamed Test Case")
                    test_name = re.sub(r"\s+", " ", str(description).strip())
                    linked_issues = row.get("link issue Test", "").strip()
                    full_group = f"{feature_tag}_{category.lower()}"
                    f.write(f"TC{tc_index:03d}: [SYS5] {test_name}\n")
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
                    
                    f.write("\n")
    
    return output_dir
