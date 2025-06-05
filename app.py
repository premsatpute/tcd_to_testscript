import streamlit as st
import pandas as pd
from io import BytesIO
import zipfile
import os
import shutil
from datetime import datetime
from robot_generator import (
    load_and_preprocess_tcd,
    generate_separate_robot_files
)

# ---------- Page Config ----------
st.set_page_config(page_title="TCD to Robot Generator", layout="wide")

# Initialize session state
if "keyword_df" not in st.session_state:
    st.session_state.keyword_df = None
if "original_keyword_df" not in st.session_state:
    st.session_state.original_keyword_df = None
if "show_mapping" not in st.session_state:
    st.session_state.show_mapping = False
if "original_keyword_rows" not in st.session_state:
    st.session_state.original_keyword_rows = 0
if "proceed_with_errors" not in st.session_state:
    st.session_state.proceed_with_errors = False
if "use_updated_keywords" not in st.session_state:
    st.session_state.use_updated_keywords = None
if "variants" not in st.session_state:
    st.session_state.variants = ""
if "type_of_testing" not in st.session_state:
    st.session_state.type_of_testing = ""
if "features" not in st.session_state:
    st.session_state.features = ""
if "sub_features" not in st.session_state:
    st.session_state.sub_features = ""
if "functionalities" not in st.session_state:
    st.session_state.functionalities = ""
if "functionality_prefixes" not in st.session_state:
    st.session_state.functionality_prefixes = ""

# ---------- Branding ----------
st.image("assets/visteon_logo.png", width=150)
st.title("TCD → .robot Scripts Generator [Prototype]")
st.markdown("""
This app converts your **TCD Excel sheet** into multiple structured `.robot` test scripts categorized by feature and type.

### 📘 Instructions:
1. Enter the label format components (Variant, TypeOfTesting, Feature, SubFeature, Functionality).
2. Upload your **TCD Excel sheet**.
3. Upload the **Keyword Mapping Excel sheet** (maps TCD keywords to Python functions).
4. Upload the **Header Template (.txt)** (contains imports and global settings).
5. View or modify the keyword mapping using the options below.
6. Validation errors will be displayed if found. You can choose to proceed with script generation.
7. Download the error report and generated scripts as a zip!

---
""")

# ---------- Label Format Inputs ----------
st.subheader("Define Label Format")
st.markdown("Enter the components of your label format (comma-separated for multiple values).")
col1, col2, col3 = st.columns(3)
with col1:
    st.session_state.variants = st.text_input("Variant (e.g., W616, MSIL, Nissan etc )", value=st.session_state.variants)
    st.session_state.type_of_testing = st.text_input("Type of Testing (e.g., FV, FT etc)", value=st.session_state.type_of_testing)
with col2:
    st.session_state.features = st.text_input("Feature (e.g., Alert, Warning, Chime,TT etc)", value=st.session_state.features)
    st.session_state.sub_features = st.text_input("Sub-Feature", value=st.session_state.sub_features)
with col3:
    st.session_state.functionalities = st.text_input("Functionality (e.g., precondition, logicalcombination)", value=st.session_state.functionalities)
    st.session_state.functionality_prefixes = st.text_input("Functionality Prefixes (e.g., TC0101, TC0201)", value=st.session_state.functionality_prefixes)

# Validate functionality and prefixes alignment
if st.session_state.functionalities and st.session_state.functionality_prefixes:
    funcs = [f.strip() for f in st.session_state.functionalities.split(",") if f.strip()]
    prefixes = [p.strip() for p in st.session_state.functionality_prefixes.split(",") if p.strip()]
    if len(funcs) != len(prefixes):
        st.error("The number of Functionality entries must match the number of Functionality Prefixes.")

# ---------- Upload Files ----------
tcd_file = st.file_uploader("📂 Upload TCD Excel Sheet", type=["xlsx"])
keyword_file = st.file_uploader("🧠 Upload Keyword Mapping Excel Sheet", type=["xlsx"])

# ---------- Keyword Mapping Management ----------
if keyword_file:
    try:
        st.session_state.keyword_df = pd.read_excel(keyword_file)
        st.session_state.original_keyword_df = st.session_state.keyword_df.copy()  # Store original keyword mapping
        st.session_state.original_keyword_rows = st.session_state.keyword_df.shape[0]
        if "TCD Keywords" not in st.session_state.keyword_df.columns or "Python Func Name" not in st.session_state.keyword_df.columns:
            st.error("Keyword Mapping Excel sheet must contain 'TCD Keywords' and 'Python Func Name' columns.")
            st.session_state.keyword_df = None
            st.session_state.original_keyword_df = None
            st.session_state.original_keyword_rows = 0
        elif st.session_state.keyword_df.empty:
            st.error("Keyword Mapping Excel sheet is empty.")
            st.session_state.keyword_df = None
            st.session_state.original_keyword_df = None
            st.session_state.original_keyword_rows = 0
    except Exception as e:
        st.error(f"Failed to read Keyword Mapping Excel sheet: {str(e)}")
        st.session_state.keyword_df = None
        st.session_state.original_keyword_df = None
        st.session_state.original_keyword_rows = 0

if st.session_state.keyword_df is not None:
    st.subheader("Keyword Mapping")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.markdown('<div class="heading-box">View Keyword Mapping</div>', unsafe_allow_html=True)
        st.markdown('<div class="column-content">', unsafe_allow_html=True)
        if st.button("View Keyword Mapping"):
            st.session_state.show_mapping = not st.session_state.show_mapping
        if st.session_state.show_mapping:
            st.dataframe(st.session_state.keyword_df)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="heading-box">Add New Keyword</div>', unsafe_allow_html=True)
        st.markdown('<div class="column-content">', unsafe_allow_html=True)
        
        # Initialize a list to store new keywords
        if "new_keywords" not in st.session_state:
            st.session_state.new_keywords = []
        
        # Form for adding a single keyword
        with st.form(key="add_keyword_form"):
            tcd_keyword = st.text_input("TCD Keyword")
            python_func = st.text_input("Python Function Name")
            submit_button = st.form_submit_button("Add Keyword")
            
            if submit_button:
                if tcd_keyword and python_func:
                    if tcd_keyword.lower() in st.session_state.keyword_df["TCD Keywords"].str.lower().values:
                        st.error(f"Keyword '{tcd_keyword}' already exists in the mapping.")
                    else:
                        # Add to temporary list
                        st.session_state.new_keywords.append({
                            "TCD Keywords": tcd_keyword,
                            "Python Func Name": python_func
                        })
                        st.success(f"Added keyword: {tcd_keyword} → {python_func}")
                else:
                    st.error("Please provide both TCD Keyword and Python Function Name.")
        
        # Ask if user wants to add more keywords
        if st.session_state.new_keywords:
            add_more = st.radio("Would you like to add another keyword?", ("Yes", "No"), key="add_more_keywords")
            if add_more == "No":
                # Add all new keywords to the dataframe
                if st.session_state.new_keywords:
                    new_mapping = pd.DataFrame(st.session_state.new_keywords)
                    st.session_state.keyword_df = pd.concat(
                        [st.session_state.keyword_df, new_mapping], ignore_index=True
                    )
                    st.session_state.original_keyword_rows = st.session_state.keyword_df.shape[0]
                    st.session_state.new_keywords = []  # Clear the temporary list
                    st.success("All new keywords have been added to the mapping.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="heading-box">Save New Keywords to File?</div>', unsafe_allow_html=True)
        st.markdown('<div class="column-content">', unsafe_allow_html=True)
        save_mapping = st.radio("Save permanently?", ("Yes", "No"), key="save_mapping")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Always provide option to download current keyword mapping
    st.markdown("**Current Keyword Mapping**")
    buffer = BytesIO()
    st.session_state.keyword_df.to_excel(buffer, index=False)
    buffer.seek(0)
    st.download_button(
        label="📥 Download Keyword Mapping",
        data=buffer,
        file_name="keyword_mapping.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    # Handle new keywords
    if st.session_state.keyword_df.shape[0] > st.session_state.original_keyword_rows:
        st.markdown("**New Keywords Added**")
        if save_mapping == "Yes":
            try:
                st.session_state.keyword_df.to_excel("keyword_mapping.xlsx", index=False)
                st.success("Keyword mapping saved permanently to keyword_mapping.xlsx")
            except Exception as e:
                st.error(f"Failed to save keyword mapping: {str(e)}")
        else:
            st.info("New keywords will be used only for this session.")
        
        # Ask user whether to use updated or original keyword mapping for script generation
        st.markdown("**Choose Keyword Mapping for Script Generation**")
        st.session_state.use_updated_keywords = st.radio(
            "Use updated keywords for script generation?",
            ("Yes", "No"),
            key="use_updated_keywords_radio"
        )

header_file = st.file_uploader("📄 Upload Header Template (.txt)", type=["txt"])

# ---------- Validate and Process Files ----------
def validate_uploaded_files(tcd_file, keyword_file, header_file, keyword_df):
    if not tcd_file:
        st.error("Please upload a TCD Excel sheet.")
        return False
    if not keyword_file:
        st.error("Please upload a Keyword Mapping Excel sheet.")
        return False
    if not header_file:
        st.error("Please upload a Header Template (.txt).")
        return False
    if keyword_df is None:
        st.error("Keyword mapping is invalid or not loaded.")
        return False
    if not st.session_state.variants or not st.session_state.type_of_testing or not st.session_state.features  or not st.session_state.functionalities:
        st.error("Please provide all label format components (Variant, TypeOfTesting, Feature, SubFeature, Functionality).")
        return False
    if not st.session_state.functionality_prefixes:
        st.error("Please provide Functionality Prefixes for naming .robot files.")
        return False
    try:
        df_tcd = pd.read_excel(tcd_file)
        required_columns = ["Labels", "Action", "Expected Results", "Description", "link issue Test"]
        missing_columns = [col for col in required_columns if col not in df_tcd.columns]
        if missing_columns:
            st.error(f"TCD Excel sheet is missing required columns: {', '.join(missing_columns)}")
            return False
    except Exception as e:
        st.error(f"Failed to read TCD Excel sheet: {str(e)}")
        return False
    return True

if tcd_file and keyword_file and header_file:
    if validate_uploaded_files(tcd_file, keyword_file, header_file, st.session_state.keyword_df):
        st.success("✅ Files uploaded and validated successfully!")
        
        # Save TCD file temporarily
        tcd_temp_path = "temp_tcd.xlsx"
        with open(tcd_temp_path, "wb") as f:
            f.write(tcd_file.getvalue())
        
        # Read files into memory
        df_tcd, error_df = load_and_preprocess_tcd(tcd_temp_path, st.session_state.keyword_df)
        header_path = "temp_header.txt"
        with open(header_path, "w", encoding="utf-8") as f:
            f.write(header_file.getvalue().decode("utf-8"))
        
        # Handle validation errors
        if not error_df.empty:
            with st.expander("Validation Errors", expanded=True):
                st.warning("The following errors were found in the TCD sheet. Please review and correct them.")
                st.dataframe(error_df)
                
                # Generate error report Excel
                error_buffer = BytesIO()
                error_df.to_excel(error_buffer, index=False)
                error_buffer.seek(0)
                st.download_button(
                    label="📥 Download Error Report",
                    data=error_buffer,
                    file_name="Error_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                # Pop-up for proceeding with errors
                st.warning(f"{len(error_df)} errors found in the TCD sheet. Do you still want to generate the scripts?")
                st.session_state.proceed_with_errors = st.checkbox("Proceed with script generation", value=False)
        
        # Proceed with script generation if no errors or user confirms
        if error_df.empty or st.session_state.proceed_with_errors:
            # Determine which keyword mapping to use
            keyword_mapping_to_use = st.session_state.keyword_df
            if st.session_state.use_updated_keywords == "No" and st.session_state.original_keyword_df is not None:
                keyword_mapping_to_use = st.session_state.original_keyword_df
                st.info("Using original keyword mapping for script generation.")
            else:
                st.info("Using updated keyword mapping for script generation.")
            
            # Create a temporary output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"robot_scripts_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate robot files
            generate_separate_robot_files(
                df=df_tcd,
                keyword_mapping_df=keyword_mapping_to_use,
                header_file_path=header_path,
                output_dir=output_dir
            )
            
            # Zip the folder
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(output_dir):
                    for file in files:
                        filepath = os.path.join(root, file)
                        arcname = os.path.relpath(filepath, output_dir)
                        zipf.write(filepath, arcname=arcname)
            zip_buffer.seek(0)
            
            # Show preview of one script
            st.markdown("### 🧪 Preview of One Generated Script")
            robot_files = []
            for root, _, files in os.walk(output_dir):
                for file in files:
                    if file.endswith(".robot"):
                        robot_files.append(os.path.join(root, file))
            if robot_files:
                try:
                    sample_path = robot_files[0]
                    with open(sample_path, "r", encoding="utf-8") as preview_file:
                        st.code(preview_file.read(), language="robotframework")
                except (PermissionError, FileNotFoundError) as e:
                    st.error(f"Failed to preview script: {str(e)}")
            else:
                st.warning("No .robot files found for preview.")
            
            # Download zip
            st.download_button(
                label="📥 Download All .robot Scripts as ZIP",
                data=zip_buffer,
                file_name="Robot_Test_Scripts.zip",
                mime="application/zip"
            )
            
            # Clean up temp folder
            shutil.rmtree(output_dir)
        
        # Clean up temp files
        if os.path.exists(tcd_temp_path):
            os.remove(tcd_temp_path)
        if os.path.exists(header_path):
            os.remove(header_path)
else:
    st.info("⬆️ Please upload all three required files and define label format components to proceed.")
