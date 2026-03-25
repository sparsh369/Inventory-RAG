import streamlit as st
import pandas as pd
import sqlite3
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from openai import OpenAI
import uuid

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Inventory SQL + RAG", layout="wide")

# Load OpenAI key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Qdrant setup (in-memory for simplicity)
qdrant = QdrantClient(":memory:")
COLLECTION_NAME = "inventory"

if COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

# SQLite setup
conn = sqlite3.connect("inventory.db", check_same_thread=False)

# ---------------- FUNCTIONS ----------------

def load_data(file):
    df = pd.read_excel(file)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def store_sql(df):
    df.to_sql("inventory", conn, if_exists="replace", index=False)


def store_qdrant(df):
    points = []
    for i, row in df.iterrows():
        text = " ".join([f"{col} {row[col]}" for col in df.columns])
        vector = embed_model.encode(text).tolist()
        payload = row.to_dict()

        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=payload
        ))

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


def classify_query(query):
    prompt = f"""
    Classify the query into one of two types:
    1. SQL (structured, aggregation, filtering)
    2. RAG (semantic/general)

    Query: {query}

    Answer only SQL or RAG.
    """

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content.strip()


def generate_sql(query):
    prompt = f"""
    Convert this question into SQL query.
    Table name: inventory

    Question: {query}
    Only return SQL.
    """

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content.strip()


def run_sql(query):
    try:
        result = pd.read_sql(query, conn)
        return result
    except Exception as e:
        return str(e)


def rag_search(query):
    vector = embed_model.encode(query).tolist()

    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=5
    )

    context = "\n".join([str(hit.payload) for hit in hits])

    prompt = f"""
    You are an inventory assistant.
    Use ONLY the context.

    Context:
    {context}

    Question: {query}
    """

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content

# ---------------- UI ----------------

st.title("📦 Inventory SQL + RAG Dashboard")

file = st.file_uploader("Upload Inventory Excel", type=["xlsx"])

if file:
    df = load_data(file)
    st.subheader("Preview")
    st.dataframe(df.head())

    if st.button("Process Data"):
        store_sql(df)
        store_qdrant(df)
        st.success("Data stored in SQL + Vector DB")

# Query section
st.subheader("Ask Questions")
query = st.text_input("Enter your question")

if st.button("Run Query") and query:
    qtype = classify_query(query)

    st.write(f"Detected Type: {qtype}")

    if "SQL" in qtype:
        sql_query = generate_sql(query)
        st.code(sql_query, language="sql")

        result = run_sql(sql_query)
        st.dataframe(result)

    else:
        answer = rag_search(query)
        st.write(answer)

# Sidebar tips
st.sidebar.title("Tips")
st.sidebar.write("""
Try questions like:
- Which items are low in stock?
- Show all items in warehouse A
- What is total quantity?
- Tell me about item X
""")
