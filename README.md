# The Aboutness Project (TAP)

## 💡 Aboutness: A Semantic Search & Discovery Tool for Academic Research

**The Aboutness Project (TAP)** is an ambitious initiative to revolutionize how researchers discover and understand academic literature. Moving beyond traditional keyword matching, TAP aims to identify the **true conceptual "aboutness"** of research papers, powered by cutting-edge Artificial Intelligence and vector database technology.

This project is inspired by decades-long research into the fundamental problem of "aboutness" in information retrieval, particularly the insights gained from early work at IBM in the 1960s. Where previous efforts relied on complex and often brittle syntactic analysis, TAP leverages the power of modern semantic embeddings to unlock a deeper, more intuitive understanding of research topics.

### 🚀 Vision

Our vision for TAP is to create an intelligent assistant that helps researchers:

* **Efficiently explore interdisciplinary connections:** Discover papers that bridge different fields, even if they use varying terminology.
* **Uncover hidden insights:** Find foundational or cutting-edge research that might be missed by conventional search methods.
* **Accelerate literature reviews:** Quickly grasp the core concepts of relevant papers and clusters of work.
* **Facilitate collaboration:** Identify researchers working on conceptually similar problems.

### ✨ Core Concept: Semantic Aboutness

TAP's innovation lies in its approach to "aboutness":

1.  **Vector Embeddings:** Instead of just looking for matching words, we transform academic papers (primarily their abstracts) into high-dimensional numerical vectors using advanced AI models. Papers that are conceptually similar will have vectors that are "close" to each other in this abstract space.
2.  **Vector Database:** These semantic vectors are then stored and indexed in a specialized vector database, optimized for lightning-fast **similarity search**.
3.  **Intelligent Retrieval-Augmented Generation (RAG):** When a user queries TAP, their input is also converted into a vector. The system then finds the most semantically similar papers. An integrated Large Language Model (LLM) then generates concise "Aboutness Snippets" and "Why Relevant?" explanations for each result, providing immediate contextual understanding.
4.  **Citation Graph Integration (Future):** A key differentiator will be the integration of citation network data (who cites whom) with semantic similarity. Papers that are both semantically close AND share strong citation links are highly likely to be truly "about" the same core concepts.

### 🛣️ Project Roadmap (Future Work)

This project is currently in its very early prototyping phase. Here's a glimpse of what's next:

* **Frontend Development:** Enhance the existing UI/UX prototype to include more filtering options, a visual "Aboutness Map," and richer display of paper metadata.
* **Backend Implementation:**
    * Set up a robust backend (e.g., using Python/FastAPI or Node.js/Express) hosted on Render.
    * Integrate a Vector Database (e.g., Qdrant, Chroma, or PostgreSQL with `pgvector`) for storing paper embeddings.
    * Implement an LLM integration for generating "Aboutness Snippets" and "Why Relevant?" explanations.
* **Data Acquisition & Ingestion:**
    * Strategize and implement methods for acquiring academic paper data (e.g., via Semantic Scholar API, arXiv API, or other academic datasets).
    * Develop robust pipelines for text extraction, chunking, and embedding generation.
* **Citation Graph Integration:** Design and implement algorithms to combine vector similarity with citation network analysis for enhanced "aboutness" ranking and discovery.
* **Evaluation Framework:** Establish a rigorous evaluation methodology using modern metrics (Recall, Precision, MAP, NDCG, RAGAS metrics like Faithfulness and Contextual Recall) to objectively measure the system's performance.

### 🌐 Live Prototype

You can view the current frontend prototype here:
**[eohatdan.github.io/TAP/](https://eohatdan.github.io/TAP/)**

* **Try it out:** Enter sample queries like "AI for climate change adaptation policy" or "ocean acidification on coral reefs" to see simulated results.
* **Explore Features:** Click "Why Relevant?", "Find More Like This", and "Add to Reading List" to see the basic interactions.

### 🤝 Contribution & Collaboration

This project is led by a pioneering vision to solve a fundamental challenge in information science. While currently a personal project, we welcome intellectual contributions, discussions, and insights from anyone passionate about semantic search, AI, and academic discovery.

### 📧 Contact

For inquiries or to discuss collaboration, please reach out via email to: daninreno@gmail.com
