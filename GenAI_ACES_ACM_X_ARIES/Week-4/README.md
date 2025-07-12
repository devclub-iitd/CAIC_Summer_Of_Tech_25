## **Gen AI Problem Statement Part \- II : Extension with Context Management**

---

## **Context Management of Ongoing Interview**

To **maintain context across user interactions** (the evolving structure of hypotheses, questions, user diagrams, and case flow), you will implement **context management** in the bot using:

### **1️⃣ State & Memory Management**

Use **stores and savers** in **LangChain / LangGraph**:

* **`SQLiteStore` or `SQLiteSaver`** (inbuilt in LangChain) for **lightweight, persistent, and local context storage**.

* Optionally, **VectorStore-backed retrieval for RAG (Retrieval-Augmented Generation)** with:

  * **Quadrant DB** (open-source, lightweight) or

  * Free cloud vector DB tiers (**MongoDB Atlas Vector Search**, **Milvus**, **Chroma**, **Pinecone free tier**).

These will store:

* Current **case state** (company, industry, geography, hypothesis tree, user’s current question)

* **Feedback and mistake logs** for personalized improvement tracking

* **Adaptive hinting** based on user’s last interaction

---

### **2️⃣ Chunk Input Data for RAG**

To handle **ingestion of your curated intake sources and case compendiums**:

* **Chunk the documents (intake sources) into semantically meaningful segments** (e.g., by case type, industry context, frameworks).

* **Insert into your RAG system (Quadrant DB / Mongo Vector Search / Milvus)** for **retrieval when users ask questions within a case**.

This enables:

* Retrieval of **framework reminders** (e.g., MECE structures, profitability trees) when users get stuck.

* Adaptive **reference to company/industry context** without hallucination.

* Handling follow-up questions with **consistency and minimal drift**.

---

### **3️⃣ Tool Documentation for Integration**

For documentation, you can provide these references to your team:

✅ **LangChain Stores/Savers:**

* [LangChain SQLiteStore Docs](https://python.langchain.com/docs/modules/data_connection/storage/stores/)

* [LangGraph Official Docs](https://www.langgraph.org/)

✅ **Vector DB Options:**

* [Quadrant DB GitHub](https://github.com/tryquadrant/tryquadrant) (lightweight local vector DB)

* [MongoDB Atlas Vector Search](https://www.mongodb.com/products/platform/atlas-vector-search)

* [Milvus Docs](https://milvus.io/docs/)

* [Chroma Docs](https://docs.trychroma.com/)

* [Pinecone Free Tier](https://www.pinecone.io/)

✅ **Chunking for RAG:**

* [LangChain Document Loaders & Chunking](https://python.langchain.com/docs/modules/data_connection/document_loaders/)

* [OpenAI Cookbook \- RAG with Chunking](https://cookbook.openai.com/examples/retrieval_augmented_generation)

---

## **Final Goal**

🚀 Implement a **functional conversational bot** capable of answering case prep queries with **context-awareness and minimal hallucination**, maintaining **case state and adaptive progression**, backed by:

✅ **Unique case generation**  
 ✅ **Simulation of interviewer dialogue**  
 ✅ **Adaptive context management via LangChain/Vector DB**  
 ✅ **Constructive, personalized feedback**  
 ✅ **Documentation-backed reproducibility**

---

## **Submission Details**

You need to prepare a **README file 📄** explaining how to run the project.

The README should include:

* ⚙️ **Steps to set up the project**

* ▶️ **How to run the project**

* 📝 **Any additional instructions or dependencies**

Additionally, include some screenshots of the chatbot in the final zip file.

**Submission Date:** 🗓️ *18th July 2025*

