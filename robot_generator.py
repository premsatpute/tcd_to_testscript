
import os
import pandas as pd
import re


def load_and_preprocess_tcd(tcd_filepath):
    df = pd.read_excel(tcd_filepath)
    df = df[["Labels", "Action", "Expected Results", "Description", "link issue Test"]].dropna()
    df["Test_Case_Type"] = df["Labels"].apply(lambda x: x.split("_")[-1].strip().lower().replace(" ", "") if isinstance(x, str) else "")
    df["Sub_Feature"] = df["Labels"].apply(lambda x: "_".join(x.split("_")[2:-1]).strip() if isinstance(x, str) else "")
    df["Normalized_Feature"] = df["Sub_Feature"].apply(lambda x: re.sub(r"[_\s]+", "_", x.strip().lower()) if isinstance(x, str) else "")
    df["link issue Test"] = df["link issue Test"].astype(str).str.replace(r"[\n\r;]+", "_", regex=True).str.strip("_")
    df["Description"] = df["Description"].str.replace(r"\s+", " ", regex=True).str.strip()  # Normalize spacing
    return df

def extract_steps(row):
    action_steps = []
    expected_steps = {}
    action_lines = row["Action"].split("\n")
    extracting = False
    for line in action_lines:
        if "Steps:" in line:
            extracting = True
            continue
        if extracting and line.strip():
            clean_line = re.sub(r"^\d+\.\s*", "", line.strip())
            action_steps.append(clean_line)
    expected_lines = row["Expected Results"].split("\n")
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
    import os
    import re

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

    # Get precondition test case (only one expected)
    precondition_row = df[df["Test_Case_Type"] == "precondition"]
    precondition = precondition_row.iloc[0] if not precondition_row.empty else None
    precondition_steps = extract_steps(precondition) if precondition is not None else []
    precondition_issues = precondition.get("link issue Test", "").strip() if precondition is not None else ""
    precondition_desc = re.sub(r"\s+", " ", precondition.get("Description", "Precondition").strip()) if precondition is not None else ""

    # Remove precondition from the main dataframe
    df = df[df["Test_Case_Type"] != "precondition"]

    grouped = df.groupby(["Normalized_Feature", "Test_Case_Type"])

    for (feature_norm, category), group_df in grouped:
        feature_tag = group_df.iloc[0]["Sub_Feature"].strip().replace(" ", "_")
        full_filename = f"{feature_tag.upper()}_{category.upper()}.robot"
        file_path = os.path.join(output_dir, full_filename)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(header_content)
            f.write("*** Test Cases ***\n\n")
            tc_index = 1

            # Add precondition test case as TC001
            if precondition is not None:
                f.write(f"TC{tc_index:03d}: [SYS5] {precondition_desc}\n")
                f.write(f"    [Documentation]    REQ_{precondition_issues}\n")
                f.write(f"    [Tags]    {feature_tag}\n")
                f.write(f"    [Description]    {precondition_desc}\n")
                f.write(f"    [Feature]    {feature_tag}\n")
                f.write(f"    [Feature_group]   {feature_tag}_precondition \n\n")

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

                    if "do" in replacement_keyword.lower():
                        f.write(f"    {replacement_keyword}\n")
                    elif value_clean:
                        f.write(f"    {replacement_keyword}    {value_clean}\n")
                    else:
                        f.write(f"    {replacement_keyword}\n")

                f.write("\n")
                tc_index += 1

            # Add category-specific test cases
            for _, row in group_df.iterrows():
                test_name = re.sub(r"\s+", " ", row.get("Description", "Unnamed Test Case").strip())
                linked_issues = row.get("link issue Test", "").strip()
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

                    if "do" in replacement_keyword.lower():
                        f.write(f"    {replacement_keyword}\n")
                    elif value_clean:
                        f.write(f"    {replacement_keyword}    {value_clean}\n")
                    else:
                        f.write(f"    {replacement_keyword}\n")

                f.write("\n")

    return output_dir
