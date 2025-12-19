import streamlit as st
import pandas as pd
import numpy as np
import os

# =============================================
# CONFIG
# =============================================

GITHUB_BASE = "https://raw.githubusercontent.com/gnaneshidex/TDS/main/Tool_Dashboard_Streamlit/data/"


DATA_DIR = "data"
st.set_page_config(page_title="Tool Compatibility Optimizer", layout="wide")

st.title("Akron Brass Company Tool Compatibility Optimizer")


# =============================================
# CACHING – SPEED BOOST
# =============================================
@st.cache_data(show_spinner=False)
def load_excel(path):
    """Loads Tool_Matrix and Binary sheets quickly with caching."""
    df_matrix = pd.read_excel(path, sheet_name="Tool_Matrix", index_col=0)
    try:
        df_binary = pd.read_excel(path, sheet_name="Binary", index_col=0).fillna(0).astype(int)
    except:
        df_binary = None
    return df_matrix, df_binary


@st.cache_data(show_spinner=False)
def optimize_sequence(shared_df, start_item):
    """Greedy optimizer with caching."""
    remaining = list(shared_df.index)
    seq = [start_item]
    remaining.remove(start_item)

    while remaining:
        last = seq[-1]
        next_item = shared_df.loc[last, remaining].idxmax()
        seq.append(next_item)
        remaining.remove(next_item)

    return seq


# =============================================
# DIAGONAL HIGHLIGHTING
# =============================================
def highlight_diagonal(df):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    rng = np.arange(len(df))
    styles.values[rng, rng] = (
        "background-color:#b3e5fc; color:black; font-weight:bold;"
    )
    return styles


# =============================================
# WORK CENTER SELECTION
# =============================================
if not os.path.exists(DATA_DIR):
    st.error(f"Folder '{DATA_DIR}' not found.")
    st.stop()

work_centers = sorted([
    f for f in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, f)) and f.startswith("WC")
])

if len(work_centers) == 0:
    st.error("No WC folders found inside the data directory.")
    st.stop()

selected_wc = st.selectbox("Select Work Center", work_centers)

# Load workbook
path = os.path.join(DATA_DIR, selected_wc, "Tool_Matrix.xlsx")
if not os.path.exists(path):
    st.error(f"Tool_Matrix.xlsx missing in {selected_wc}")
    st.stop()

df, df_binary = load_excel(path)

st.success(f"Loaded matrix: {df.shape[0]} × {df.shape[1]}")


# =============================================
# ITEM SELECTION (EMPTY ON LOAD)
# =============================================
all_items = df.index.tolist()

selected_items = st.multiselect(
    "Select items to include",
    all_items,
    default=[]  # 🔥 Empty by default
)

# Show start item selector only when items selected
if selected_items:
    start_item = st.selectbox("Select starting item", selected_items)
else:
    start_item = None


# =============================================
# OPTIMIZATION BUTTON
# =============================================
if st.button("⚙️ Optimize"):

    if not selected_items:
        st.error("Please select at least one item.")
        st.stop()

    if start_item is None:
        st.error("Please select a starting item.")
        st.stop()

    # Subset shared matrix
    valid = [i for i in selected_items if i in df.index]
    df_sub = df.loc[valid, valid]

    # Compute optimized sequence
    seq = optimize_sequence(df_sub, start_item)

    st.subheader("🧩 Optimized Sequence")
    st.write(" → ".join(seq))

    # =============================================
    # TOOL MATRIX FOR SEQUENCE
    # =============================================
    st.subheader("📘 Tool Matrix for Sequence")

    seq_matrix = df_sub.loc[seq, seq]
    styled_matrix = seq_matrix.style.apply(highlight_diagonal, axis=None)

    st.dataframe(styled_matrix, use_container_width=True)

    # =============================================
    # SUMMARY TABLE (BINARY) – optional
    # =============================================
    if df_binary is not None:

        bin_sub = df_binary.loc[seq]
        total_bin = bin_sub.sum(axis=1)

        saved = [0]
        added = [int(total_bin.loc[seq[0]])]

        for i in range(1, len(seq)):
            prev = bin_sub.loc[seq[i-1]].values
            curr = bin_sub.loc[seq[i]].values

            shared = int(np.sum((prev == 1) & (curr == 1)))
            saved.append(shared)
            added.append(int(total_bin.loc[seq[i]] - shared))

        summary_table = pd.DataFrame({
            "Item": seq,
            "Total Tools Required": total_bin.loc[seq].values,
            "Tool Setup Saved": saved,
            "Tool Changes Added": added
        })

        summary_table.loc["SUM"] = [
            "—",
            summary_table["Total Tools Required"].sum(),
            summary_table["Tool Setup Saved"].sum(),
            summary_table["Tool Changes Added"].sum()
        ]

        st.subheader("📘 Summary Table")
        st.dataframe(summary_table, use_container_width=True)

    # =============================================
    # EXPORT TO EXCEL
    # =============================================
    out_path = os.path.join(DATA_DIR, selected_wc, "Optimization_Results.xlsx")

    with pd.ExcelWriter(out_path) as writer:
        pd.DataFrame({"Optimized_Sequence": seq}).to_excel(
            writer, sheet_name="Sequence", index=False
        )
        seq_matrix.to_excel(writer, sheet_name="Tool_Matrix", index=True)

        if df_binary is not None:
            summary_table.to_excel(writer, sheet_name="Summary_Table", index=False)

    with open(out_path, "rb") as f:
        st.download_button(
            "💾 Download Results Excel",
            f,
            file_name=f"{selected_wc}_Optimization_Results.xlsx"
        )

