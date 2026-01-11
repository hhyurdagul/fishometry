import streamlit as st
import polars as pl
import os

# Constants
CSV_PATH = "/students/Hasan/fishometry/data/data-outside/raw.csv"
IMG_DIR = "/students/Hasan/fishometry/data/data-outside/raw"
ITEMS_PER_PAGE = 20

def main():
    st.set_page_config(page_title="Raw Data Explorer", layout="wide")
    st.title("Raw Data Explorer")

    # 1. Load Data
    if not os.path.exists(CSV_PATH):
        st.error(f"CSV file not found: {CSV_PATH}")
        return

    try:
        df = pl.read_csv(CSV_PATH)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return

    # 2. Sort Data (Ascending by length)
    if "length" in df.columns:
        df = df.sort("length", descending=False)
    else:
        st.warning("Column 'length' not found in CSV. Showing unsorted.")

    # 3. Sidebar stats and controls
    st.sidebar.header("Controls")
    total_items = df.height
    st.sidebar.text(f"Total Items: {total_items}")

    # Pagination
    if "page" not in st.session_state:
        st.session_state.page = 0

    total_pages = max(1, (total_items + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
    
    col_prev, col_page, col_next = st.columns([1, 2, 1])
    
    with col_prev:
        if st.button("Previous"):
            st.session_state.page = max(0, st.session_state.page - 1)
    
    with col_next:
        if st.button("Next"):
            st.session_state.page = min(total_pages - 1, st.session_state.page + 1)
            
    with col_page:
        st.write(f"Page {st.session_state.page + 1} of {total_pages}")

    # Slice data for current page
    start_idx = st.session_state.page * ITEMS_PER_PAGE
    end_idx = start_idx + ITEMS_PER_PAGE
    current_df = df.slice(start_idx, ITEMS_PER_PAGE)

    # 4. Display List
    st.divider()
    
    for row in current_df.iter_rows(named=True):
        img_name = row.get("name", "Unknown")
        length = row.get("length", "N/A")
        fish_type = row.get("fish_type", "N/A")
        
        img_path = os.path.join(IMG_DIR, img_name)
        
        # Container for each item
        with st.container():
            c1, c2, c3 = st.columns([1, 2, 1])
            
            with c1:
                if os.path.exists(img_path):
                    st.image(img_path, width=250)
                else:
                    st.warning(f"Image not found: {img_name}")
            
            with c2:
                st.subheader(f"{fish_type}")
                st.write(f"**Filename:** {img_name}")
                st.write(f"**Length:** {length} cm")
            
            with c3:
                st.write("Actions")
                if os.path.exists(img_path):
                    # Use a unique key for the button
                    if st.button("🗑️ Delete File", key=f"del_{img_name}"):
                        try:
                            os.remove(img_path)
                            st.toast(f"Deleted {img_name}", icon="✅")
                            # Rerun to update the UI (image will disappear or show as not found)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to delete: {e}")
                else:
                    st.info("File already deleted")
            
            st.divider()

if __name__ == "__main__":
    main()
