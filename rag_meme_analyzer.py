import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import json
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MemeTrendRAG:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        self.vectorstore = None
        self.ds_glossary = [
            "Pandas: Python lib for data manipulation, often memed for slow groupbys.",
            "RAG: Retrieval-Augmented Generation, boosting LLM accuracy with external data.",
            "Trend Score: Virality metric = (likes + retweets) / age_in_days"
        ]

    def load_or_create_index(self, memes_data):
        if not memes_data:
            logger.error("No meme data provided for indexing.")
            raise ValueError("No meme data provided for indexing.")
        texts = [f"{m['text']} | Desc: {m['image_desc']}" for m in memes_data]
        logger.debug(f"Indexing {len(texts)} memes and {len(self.ds_glossary)} glossary items.")
        self.vectorstore = FAISS.from_texts(texts + self.ds_glossary, self.embeddings)
        self.vectorstore.save_local("data/meme_index")

    def analyze_trends(self, query="What's buzzing in data analytics memes?"):
        logger.debug(f"Received query: {query}")
        if not self.vectorstore:
            logger.debug("Loading vectorstore from data/meme_index")
            self.vectorstore = FAISS.load_local("data/meme_index", self.embeddings, allow_dangerous_deserialization=True)

        prompt_template = """
        You are a witty data oracle. Using the provided meme contexts and analytics knowledge, generate a fun, insightful report on the query: {query}.
        Include:
        1) Top 3 trends with virality scores.
        2) Predictive forecast (e.g., "This meme wave predicts SQL's comeback").
        3) A generated counter-meme idea.
        Retrieved Contexts: {context}
        """
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["query", "context"])

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

        try:
            logger.debug(f"Invoking QA chain with query: {query}")
            result = qa_chain.invoke({"query": query})
            logger.debug(f"QA chain result: {result}")
            return result["result"]
        except Exception as e:
            logger.error(f"Error in RAG chain: {str(e)}")
            raise Exception(f"Error in RAG chain: {str(e)}")

if __name__ == "__main__":
    rag = MemeTrendRAG()
    with open("examples/sample_memes.json", "r") as f:
        memes = json.load(f)
    rag.load_or_create_index(memes)
    insight = rag.analyze_trends("What's trending in data analytics memes?")
    print(insight)