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

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ✅ Cache model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_model()

# Qdrant setup
qdrant = QdrantClient(":memory:")
COLLECTION_NAME = "inventory"

if COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

# SQLite
conn = sqlite3.connect("inventory.db", check_same_thread=False)

# ---------------- FUNCTIONS ----------------

def load_data(file):
    df = pd.read_excel(file)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def store_sql(df):
    df.to_sql("inventory", conn, if_exists="replace", index=False)


# ✅ FIXED FAST VERSION (WITH PROGRESS BAR)
def store_qdrant(df):
    texts = []
    payloads = []

    for _, row in df.iterrows():
        text = " ".join([f"{col} {row[col]}" for col in df.columns])
        texts.append(text)
        payloads.append(row.to_dict())

    batch_size = 32
    all_vectors = []

    progress = st.progress(0)

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        vectors = embed_model.encode(batch)
        all_vectors.extend(vectors)

        progress.progress(min((i + batch_size) / len(texts), 1.0))

    points = []
    for i in range(len(all_vectors)):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=all_vectors[i].tolist(),
            payload=payloads[i]
        ))

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


# ❌ REMOVED CACHE (IMPORTANT FIX)
def process_data(df):
    store_sql(df)
    store_qdrant(df)


def classify_query(query):
    prompt = f"""
    Classify query as SQL or RAG.

    SQL = numbers, filters, aggregation
    RAG = explanation, meaning

    Query: {query}
    Answer only SQL or RAG.
    """

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content.strip()


def generate_sql(query, columns):
    prompt = f"""
    You are a SQL expert.

    Table: inventory
    Columns: {columns}

    Rules:
    - Use ONLY these columns
    - Do NOT invent columns
    - Replace 'stock' with 'shelf_stock'
    - Do NOT use 'threshold'

    Question: {query}

    Return ONLY SQL.
    """

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content.strip()


def run_sql(query):
    try:
        df = pd.read_sql(query, conn)
        return df, None
    except Exception as e:
        return None, str(e)


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
        with st.spinner("Processing data... ⏳"):
            process_data(df)
            st.session_state["processed"] = True
        st.success("✅ Data processed successfully!")

# ---------------- QUERY ----------------

st.subheader("Ask Questions")

query = st.text_input("Enter your question")

if "processed" not in st.session_state:
    st.warning("Please click 'Process Data' first")
    st.stop()

if st.button("Run Query") and query:
    qtype = classify_query(query)

    st.write(f"Detected Type: {qtype}")

    if "SQL" in qtype:
        sql_query = generate_sql(query, df.columns.tolist())
        st.code(sql_query, language="sql")

        result, error = run_sql(sql_query)

        if error:
            st.error(error)
        else:
            st.dataframe(result)

    else:
        answer = rag_search(query)
        st.write(answer)

# ---------------- SIDEBAR ----------------

st.sidebar.title("Tips")
st.sidebar.write("""
Try:
- items where shelf_stock < safety_stock
- top 10 items by demand
- show items in plant 2001
- tell me about material X
""")
