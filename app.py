import streamlit as st
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from rag_meme_analyzer import MemeTrendRAG
from utils.visualizer import plot_virality_trends  # Explicit import

# Load environment variables
load_dotenv()

st.title("🦆 MemeTrendRAG: Meme-Powered Data Analytics Oracle")

# Sidebar for inputs
st.sidebar.header("Meme Data")
st.sidebar.write("No X API? Use sample data or add your own memes below!")

# Manual meme input
with st.sidebar.form(key="add_meme_form"):
    meme_text = st.text_input("Meme Text (e.g., 'Pandas is slow #DataScience')")
    meme_desc = st.text_input("Image Description (e.g., 'Sad panda cartoon')")
    likes = st.number_input("Likes", min_value=0, value=50)
    retweets = st.number_input("Retweets", min_value=0, value=10)
    submit_meme = st.form_submit_button("Add Meme")
    if submit_meme and meme_text and meme_desc:
        new_meme = {
            "text": meme_text,
            "image_desc": meme_desc,
            "metadata": {"likes": likes, "retweets": retweets, "date": str(datetime.now())}
        }
        try:
            with open("examples/sample_memes.json", "r") as f:
                memes = json.load(f)
        except FileNotFoundError:
            memes = []
        memes.append(new_meme)
        with open("examples/sample_memes.json", "w") as f:
            json.dump(memes, f)
        st.sidebar.success("Meme added!")

# Load and visualize sample data
try:
    with open("examples/sample_memes.json", "r") as f:
        memes = json.load(f)
    if memes:
        st.sidebar.success(f"Loaded {len(memes)} memes from sample data.")
        plot_virality_trends(memes)
        if os.path.exists("data/virality_plot.png"):
            st.image("data/virality_plot.png", caption="Meme Virality Trends")
        else:
            st.warning("Virality plot not found. Try adding more memes.")
    else:
        st.sidebar.warning("Sample data is empty. Add a meme above!")
except FileNotFoundError:
    st.sidebar.error("No sample data found. Add a meme to start!")
except Exception as e:
    st.sidebar.error(f"Error loading data: {str(e)}")

# Main RAG Query
query = st.text_input("Ask about trends:", "Predict next big data tool from memes?")
if st.button("Generate Insights"):
    try:
        rag = MemeTrendRAG()
        with open("examples/sample_memes.json", "r") as f:
            memes = json.load(f)
        if not memes:
            st.error("No memes available. Add one in the sidebar!")
        else:
            rag.load_or_create_index(memes)
            insight = rag.analyze_trends(query)
            st.markdown("### 🎭 Augmented Meme Insights")
            st.write(insight)
    except FileNotFoundError:
        st.error("No meme data found. Add a meme in the sidebar!")
    except Exception as e:
        st.error(f"Error generating insights: {str(e)}")