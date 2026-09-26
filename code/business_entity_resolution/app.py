"""
Interactive Web Dashboard for Distributed Business Entity Resolution.
Allows picking Laptop Shards, monitoring progress, and 1-Click Merging.
"""

import streamlit as st
import subprocess
import os
import sys
from pathlib import Path
import time
import pandas as pd

# Page setup
st.set_page_config(
    page_title="Entity Resolution Distributed Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⚡ Distributed Entity Resolution Hub</div>', unsafe_allow_html=True)
st.caption("3-Machine Parallel Inference & 1-Click Submission Merger")

tabs = st.tabs(["🚀 Run Shard on This Laptop", "🧩 Merge All Shards", "📊 Submission Preview"])

# ----------------- TAB 1: RUN SHARD -----------------
with tabs[0]:
    st.subheader("Configure & Run Shard on this Machine")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        total_shards = st.selectbox("Total Laptops / Machines", [2, 3, 4, 5], index=1)
    with col2:
        shard_options = [f"Laptop {i+1} (Part {i+1} of {total_shards})" for i in range(total_shards)]
        selected_shard_str = st.selectbox("Assign this Laptop to:", shard_options)
        shard_id = shard_options.index(selected_shard_str)
    with col3:
        chunk_size = st.select_slider("Streaming Chunk Size", options=[50000, 100000, 200000], value=100000)

    colA, colB = st.columns(2)
    with colA:
        test_dir_input = st.text_input("Test Data Folder", value=r"D:\test_data")
    with colB:
        output_dir_input = st.text_input("Output Destination Folder", value=r"D:\output")

    st.info(f"💡 **Target Workload:** This laptop will process **Shard {shard_id + 1}/{total_shards}** (approx {1732544 // total_shards:,} entities) and stream results directly to `{output_dir_input}` without filling your C: drive.")

    if st.button("▶ Start Sharded Inference", type="primary", use_container_width=True):
        st.write("---")
        st.write(f"Executing: `python src/shard_infer.py --num-shards {total_shards} --shard-id {shard_id}`")
        
        script_path = Path(__file__).parent / "src" / "shard_infer.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--test-dir", test_dir_input,
            "--output-dir", output_dir_input,
            "--num-shards", str(total_shards),
            "--shard-id", str(shard_id),
            "--chunk-size", str(chunk_size)
        ]

        with st.spinner(f"Running Shard {shard_id + 1} across all CPU cores..."):
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            log_box = st.empty()
            full_log = ""
            for line in iter(process.stdout.readline, ""):
                full_log += line
                log_box.code(full_log[-2000:], language="bash")
            process.wait()

            if process.returncode == 0:
                st.success(f"✅ Shard {shard_id + 1} of {total_shards} completed successfully! Files saved in {output_dir_input}")
            else:
                st.error("❌ Process exited with an error. Review log above.")

# ----------------- TAB 2: MERGE SHARDS -----------------
with tabs[1]:
    st.subheader("1-Click Shard Merger")
    st.write("Once all 3 laptops have completed their shards, copy all shard TSV files into one common folder (e.g., `D:\\output`) and merge them here.")

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        merge_out_dir = st.text_input("Folder containing Shard TSVs", value=r"D:\output", key="m_out")
    with m_col2:
        merge_shards_count = st.number_input("Number of Shards", min_value=2, max_value=10, value=3)

    out_p = Path(merge_out_dir)
    shard_cand_files = list(out_p.glob("*_candidate_pairs.tsv"))
    shard_match_files = list(out_p.glob("*_matching_results.tsv"))

    st.write(f"Detected **{len(shard_cand_files)}** candidate files and **{len(shard_match_files)}** matching files in `{merge_out_dir}`.")

    if st.button("🧩 Merge Shards & Validate Submission", type="primary", use_container_width=True):
        merge_script = Path(__file__).parent / "merge_shards.py"
        cmd = [
            sys.executable,
            str(merge_script),
            "--output-dir", merge_out_dir,
            "--num-shards", str(merge_shards_count),
            "--test-dir", r"D:\test_data"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        st.code(res.stdout, language="bash")
        if res.returncode == 0:
            st.success("🎉 Final `candidate_pairs.tsv` and `matching_results.tsv` successfully generated and 100% PRD validated!")
        else:
            st.error("Merge error occurred. Check file names.")

# ----------------- TAB 3: PREVIEW -----------------
with tabs[2]:
    st.subheader("Final Output Inspection")
    cand_f = Path(r"D:\output\candidate_pairs.tsv")
    match_f = Path(r"D:\output\matching_results.tsv")

    if cand_f.exists() and match_f.exists():
        st.success("Found merged submission files in `D:\\output`!")
        c1, c2 = st.columns(2)
        with c1:
            st.write("`candidate_pairs.tsv` (Top 10):")
            df_c = pd.read_csv(cand_f, sep="\t", nrows=10)
            st.dataframe(df_c, use_container_width=True)
        with c2:
            st.write("`matching_results.tsv` (Top 10):")
            df_m = pd.read_csv(match_f, sep="\t", nrows=10)
            st.dataframe(df_m, use_container_width=True)
    else:
        st.info("Run Shards & Merge to preview submission files here.")
