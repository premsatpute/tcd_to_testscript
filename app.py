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

# ---------- Branding ----------
st.image("assets/visteon_logo.png", width=150)
st.title("TCD →  .robot Scripts Generator [Prototype]")
st.markdown("""
This app converts your **TCD Excel sheet** into multiple structured `.robot` test scripts categorized by feature and type.

### 📘 Instructions:
1. Upload your **TCD Excel sheet**.
2. Upload the **Keyword Mapping sheet** (maps TCD keywords to Python functions).
3. Upload the **Header Template (.txt)** (contains imports and global settings).
4. Click to generate scripts & download all in a zip!

---
""")

# ---------- Upload Files ----------
tcd_file = st.file_uploader("📂 Upload TCD Excel Sheet", type=["xlsx"])
keyword_file = st.file_uploader("🧠 Upload Keyword Mapping Excel Sheet", type=["xlsx"])
header_file = st.file_uploader("📄 Upload Header Template (.txt)", type=["txt"])

# ---------- Process Files ----------
if tcd_file and keyword_file and header_file:
    st.success("✅ All files uploaded successfully!")

    # Read files into memory
    df_tcd = load_and_preprocess_tcd(tcd_file)
    keyword_df = pd.read_excel(keyword_file)
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
    sample_files = os.listdir(output_dir)
    if sample_files:
        st.markdown("### 🧪 Preview of One Generated Script")
        sample_path = os.path.join(output_dir, sample_files[0])
        with open(sample_path, "r", encoding="utf-8") as preview_file:
            st.code(preview_file.read(), language="robotframework")

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
