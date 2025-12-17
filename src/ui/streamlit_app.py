import streamlit as st
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Import engine components with error handling
try:
    from src.engine.retriever import Retriever
    from src.engine.llm_handler import LLMHandler
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False


@st.cache_resource
def load_engine():
    """
    Load and cache the search engine components.

    Returns:
        tuple: (Retriever, LLMHandler) or (None, None) if engine not available
    """
    if HAS_ENGINE:
        return Retriever(), LLMHandler()
    return None, None


# Initialize engine components
retriever, llm = load_engine()

# Configure the Streamlit page
st.set_page_config(page_title="SHL Assessment Finder")
st.title("SHL ASSESSMENT FINDER")
st.write("")

# Input section - Search query and filters
_, middle_space, _ = st.columns([1, 3, 1])

with middle_space:
    job_query = st.text_input("Enter Job Role or Skills",
                              placeholder="e.g. Java Developer with Teamwork")
    st.success(f"QUERY: {job_query if job_query else 'Waiting for input...'}")
    st.write("")

# Filter options
col1, col2 = st.columns(2)

with col1:
    test_type = st.selectbox("Filter by Type",
                             ['All Types', 'Knowledge & Skills', 'Personality & Behavior',
                              'Ability & Aptitude', 'Simulations'])
    st.success(f"Type: {test_type}")
    st.write("")

with col2:
    limit = st.number_input("Max Results", min_value=1, max_value=10, value=10, step=1)
    st.success(f"Limit: {limit}")
    st.write("")

# Information about the tool
st.info("""
**Note:** This tool uses Hybrid Search (Vector + Keywords) to find the best matching SHL assessments for your job description.
""")
st.write("")

# Initialize session state for search results
if "search_pressed" not in st.session_state:
    st.session_state["search_pressed"] = False

# Search button and metrics layout
left, middle_space, right = st.columns([1.5, 1, 1])

# Search button and processing
with middle_space:
    if st.button("SEARCH RECOMMENDATIONS"):
        if not job_query:
            st.error("Please enter a Job Role first.")
        else:
            st.session_state["search_pressed"] = True

            with st.spinner("Analyzing database..."):
                if retriever:
                    # Get initial search results with high recall
                    results = retriever.search(job_query, n_results=200)

                    # Apply type filter if selected
                    if test_type != 'All Types':
                        results = [r for r in results if test_type in str(r.get('test_type', ''))]

                    # Apply LLM reranking if available
                    if llm:
                        try:
                            results = llm.rerank(job_query, results)
                        except Exception:
                            pass  # Fallback to vector results if LLM fails

                    # Limit results to user-specified count
                    st.session_state["results"] = results[:limit]
                else:
                    st.session_state["results"] = []

            # Display metrics about search results
            count = len(st.session_state["results"])
            with left:
                st.metric(label="Assessments Found", value=f"{count}")
            with right:
                st.metric(label="Status", value="Complete")

# Results display section
st.write("")
st.markdown("---")

# Show results if search has been performed
if st.session_state["search_pressed"]:
    results = st.session_state.get("results", [])

    if not results:
        st.warning("No assessments found matching your criteria.")
    else:
        st.subheader(f"Top {len(results)} Recommendations")

        # Display each result as an expandable card
        for i, res in enumerate(results):
            t_type = res.get('test_type', 'General')
            duration = res.get('duration', 'N/A')
            url = res.get('url', '#')

            with st.expander(f"{i + 1}. {res['name']}  (Duration: {duration} mins)"):
                col_a, col_b = st.columns([3, 1])

                with col_a:
                    st.markdown(f"**Type:** `{t_type}`")
                    st.markdown(f"**Description:** {res.get('description', '')}")

                with col_b:
                    st.link_button("🔗 View Assessment", url)
