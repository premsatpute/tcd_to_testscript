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

# ---------- Page Config (Must be the first Streamlit command) ----------
st.set_page_config(page_title="TCD to Robot Generator", layout="wide")

# Initialize session state for keyword mapping and preview toggle
if "keyword_df" not in st.session_state:
    st.session_state.keyword_df = None
if "show_mapping" not in st.session_state:
    st.session_state.show_mapping = False
if "original_keyword_rows" not in st.session_state:
    st.session_state.original_keyword_rows = 0

# Custom CSS for styling
st.markdown("""
<style>
    .heading-box {
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 8px 15px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 10px;
        display: inline-block;
        width: 100%;
        box-sizing: border-box;
    }
    .stButton > button {
        width: 100%;
        margin-top: 10px;
    }
    .stTextInput > div > div > input {
        width: 100%;
    }
    .stRadio > div {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-top: 10px;
    }
    .stRadio > div > label {
        margin: 5px 0;
    }
    .column-content {
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Branding ----------
st.image("assets/visteon_logo.png", width=150)
st.title("TCD →  .robot Scripts Generator [Prototype]")
st.markdown("""
This app converts your **TCD Excel sheet** into multiple structured `.robot` test scripts categorized by feature and type.

### 📘 Instructions:
1. Upload your **TCD Excel sheet**.
2. Upload the **Keyword Mapping Excel sheet** (maps TCD keywords to Python functions).
3. Upload the **Header Template (.txt)** (contains imports and global settings).
4. View or modify the keyword mapping using the options below.
5. Click to generate scripts & download all in a zip!

---
""")

# ---------- Upload Files ----------
tcd_file = st.file_uploader("📂 Upload TCD Excel Sheet", type=["xlsx"])
keyword_file = st.file_uploader("🧠 Upload Keyword Mapping Excel Sheet", type=["xlsx"])
header_file = st.file_uploader("📄 Upload Header Template (.txt)", type=["txt"])

# ---------- Keyword Mapping Management ----------
if keyword_file:
    try:
        st.session_state.keyword_df = pd.read_excel(keyword_file)
        st.session_state.original_keyword_rows = st.session_state.keyword_df.shape[0]
        if "TCD Keywords" not in st.session_state.keyword_df.columns or "Python Func Name" not in st.session_state.keyword_df.columns:
            st.error("Keyword Mapping Excel sheet must contain 'TCD Keywords' and 'Python Func Name' columns.")
            st.session_state.keyword_df = None
            st.session_state.original_keyword_rows = 0
        elif st.session_state.keyword_df.empty:
            st.error("Keyword Mapping Excel sheet is empty.")
            st.session_state.keyword_df = None
            st.session_state.original_keyword_rows = 0
    except Exception as e:
        st.error(f"Failed to read Keyword Mapping Excel sheet: {str(e)}")
        st.session_state.keyword_df = None
        st.session_state.original_keyword_rows = 0

if st.session_state.keyword_df is not None:
    st.subheader("Keyword Mapping")
    
    # Create side-by-side columns
    col1, col2, col3 = st.columns([2, 2, 1])
    
    # Column 1: View Keyword Mapping
    with col1:
        st.markdown('<div class="heading-box">View Keyword Mapping</div>', unsafe_allow_html=True)
        st.markdown('<div class="column-content">', unsafe_allow_html=True)
        if st.button("View Keyword Mapping"):
            st.session_state.show_mapping = not st.session_state.show_mapping
        if st.session_state.show_mapping:
            st.dataframe(st.session_state.keyword_df)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Column 2: Add New Keyword
    with col2:
        st.markdown('<div class="heading-box">Add New Keyword</div>', unsafe_allow_html=True)
        st.markdown('<div class="column-content">', unsafe_allow_html=True)
        tcd_keyword = st.text_input("TCD Keyword", key="tcd_keyword")
        python_func = st.text_input("Python Function Name", key="python_func")
        if st.button("Add Keyword"):
            if tcd_keyword and python_func:
                # Validate for duplicates
                if tcd_keyword.lower() in st.session_state.keyword_df["TCD Keywords"].str.lower().values:
                    st.error(f"Keyword '{tcd_keyword}' already exists in the mapping.")
                else:
                    new_mapping = pd.DataFrame({
                        "TCD Keywords": [tcd_keyword],
                        "Python Func Name": [python_func]
                    })
                    st.session_state.keyword_df = pd.concat(
                        [st.session_state.keyword_df, new_mapping], ignore_index=True
                    )
                    st.success(f"Added keyword: {tcd_keyword} → {python_func}")
            else:
                st.error("Please provide both TCD Keyword and Python Function Name.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Column 3: Save to File Option
    with col3:
        st.markdown('<div class="heading-box">Save New Keywords to File?</div>', unsafe_allow_html=True)
        st.markdown('<div class="column-content">', unsafe_allow_html=True)
        save_mapping = st.radio("Save permanently?", ("Yes", "No"), key="save_mapping")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display updated mapping and download option after adding a keyword
    if st.session_state.keyword_df.shape[0] > st.session_state.original_keyword_rows:
        st.markdown("**Updated Keyword Mapping**")
        st.dataframe(st.session_state.keyword_df)
        
        # Save permanently if selected
        if save_mapping == "Yes":
            try:
                st.session_state.keyword_df.to_excel("keyword_mapping.xlsx", index=False)
                st.success("Keyword mapping saved permanently to keyword_mapping.xlsx")
            except Exception as e:
                st.error(f"Failed to save keyword mapping: {str(e)}")
        else:
            st.info("New keywords will be used only for this session.")
        
        # Download updated mapping
        buffer = BytesIO()
        st.session_state.keyword_df.to_excel(buffer, index=False)
        buffer.seek(0)
        st.download_button(
            label="📥 Download Updated Keyword Mapping",
            data=buffer,
            file_name="keyword_mapping.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ---------- Validate Uploaded Files ----------
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

# ---------- Process Files ----------
if tcd_file and keyword_file and header_file:
    if validate_uploaded_files(tcd_file, keyword_file, header_file, st.session_state.keyword_df):
        st.success("✅ Files uploaded and validated successfully!")
        
        # Read files into memory
        df_tcd = load_and_preprocess_tcd(tcd_file)
        keyword_df = st.session_state.keyword_df  # Use current keyword mapping
        header_path = "temp_header.txt"
        
        with open(header_path, "w", encoding="utf-8") as f:
            f.write(header_file.getvalue().decode("utf-8"))
        
        # Create a temporary output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"robot_scripts_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate robot files
        generate_separate_robot_files(
            df=df_tcd,
            keyword_mapping_df=keyword_df,
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
        
        # Show preview of one of the scripts
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
else:
    st.info("⬆️ Please upload all three required files to proceed.")
