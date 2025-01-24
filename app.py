import os
import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader, PdfWriter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import BM25Retriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from sentence_transformers import CrossEncoder
from langchain_core.retrievers import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from typing import List, Optional
from pydantic import BaseModel, Field
import io
import base64
import json
from datetime import datetime
from dotenv import load_dotenv

def save_chunks_to_json_per_file(documents: List[Document], output_directory: str = "chunk_outputs"):
    """Save chunks and their metadata to separate JSON files for each file."""
    import os
    os.makedirs(output_directory, exist_ok=True)

    files = {}
    for doc in documents:
        filename = doc.metadata['filename']
        if filename not in files:
            files[filename] = []
        files[filename].append({
            "chunk_content": doc.page_content,
            "metadata": doc.metadata
        })

    for filename, chunks in files.items():
        # Remove invalid characters from the filename (like slashes)
        sanitized_filename = filename.replace("/", "_").replace("\\", "_")
        output_path = os.path.join(output_directory, f"{sanitized_filename}_chunks.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"Chunks for {filename} saved to {output_path}")


# Cross-Encoder Initialization
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Define Reranker Class
class CrossEncoderReranker:
    def __init__(self, cross_encoder_model, k=5):
        self.cross_encoder = cross_encoder_model
        self.k = k

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        if not documents:
            return []

        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.cross_encoder.predict(pairs)

        # Add rerank scores to document metadata
        for doc, score in zip(documents, scores):
            doc.metadata['rerank_score'] = float(score)

        doc_score_pairs = list(zip(documents, scores))
        reranked_docs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

        # Return top k
        return [doc for doc, _ in reranked_docs[:self.k]]

class HybridRetrieverWithReranking(BaseRetriever, BaseModel):
    bm25_retriever: BM25Retriever = Field(...)
    vector_retriever: Optional[BaseRetriever] = Field(default=None)  # Allow None
    reranker: CrossEncoderReranker = Field(...)
    k: int = Field(default=5)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        # Get documents from BM25 retriever
        bm25_docs = self.bm25_retriever.get_relevant_documents(query)

        # Get documents from vector retriever, if provided
        vector_docs = []
        if self.vector_retriever is not None:
            vector_docs = self.vector_retriever.get_relevant_documents(query)

        # Combine and deduplicate documents
        seen_contents = set()
        combined_docs = []
        for doc in bm25_docs + vector_docs:
            if doc.page_content not in seen_contents:
                combined_docs.append(doc)
                seen_contents.add(doc.page_content)

        # Rerank documents
        reranked_docs = self.reranker.rerank(query, combined_docs)

        return reranked_docs

# Configure API Key
def configure_api():
    load_dotenv()  # Load environment variables from the .env file
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("API key not found. Please set it in the .env file.")
    genai.configure(api_key=api_key)

# Extract PDF Text
def get_pdf_text(pdf_files: List[io.BytesIO]) -> str:
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
    return text

# Split Text into Chunks
def get_chunks(text: str, pdf_files: List[io.BytesIO]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    documents = []
    for pdf_index, pdf in enumerate(pdf_files):
        reader = PdfReader(pdf)
        pdf_name = pdf.name
        for page_number, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            chunked_text = splitter.split_text(page_text)
            for chunk_index, chunk in enumerate(chunked_text):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            'page_number': page_number + 1,
                            'document_id': pdf_index,
                            'filename': pdf_name,
                            'chunk_index': chunk_index,
                            'chunk_size': len(chunk),
                            'total_chunks_in_page': len(chunked_text)
                        }
                    )
                )

    # Save chunks to JSON file
    json_path = save_chunks_to_json_per_file(documents)
    print(f"Chunks saved to {json_path}")

    return documents

def create_hybrid_retriever(text_chunks: List[Document]):
    """
    Create a hybrid retriever combining BM25 and CrossEncoder reranking without ChromaDB.
    """
    bm25_retriever = BM25Retriever.from_documents(text_chunks)
    bm25_retriever.k = 10

    reranker = CrossEncoderReranker(cross_encoder, k=5)

    hybrid_retriever = HybridRetrieverWithReranking(
        bm25_retriever=bm25_retriever,
        vector_retriever=None,  # No vector-based retriever
        reranker=reranker,
        k=5
    )
    return hybrid_retriever, None

# Create Conversation Chain
def create_conversation_chain(retriever: BaseRetriever):
    llm = ChatGoogleGenerativeAI(
        model='models/gemini-1.5-pro-latest',
        temperature=0.7
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={'prompt': None}
    )
    return conversation_chain

def extract_page_from_pdf(pdf_file: io.BytesIO, page_number: int) -> Optional[io.BytesIO]:
    """Extract a single page from PDF and return it as a new PDF file."""
    try:
        reader = PdfReader(pdf_file)
        writer = PdfWriter()
        page_idx = page_number - 1

        if 0 <= page_idx < len(reader.pages):
            writer.add_page(reader.pages[page_idx])

            output_buffer = io.BytesIO()
            writer.write(output_buffer)
            output_buffer.seek(0)

            return output_buffer
    except Exception as e:
        st.error(f"Error extracting PDF page: {e}")
    return None

def show_pdf_page(pdf_bytes: io.BytesIO):
    """Display PDF using embedded HTML."""
    base64_pdf = base64.b64encode(pdf_bytes.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def display_chunks(chunks: List[Document], pdf_files: List[io.BytesIO], k: int = 5):
    """Display top k relevant chunks with chunk text on the left and PDF preview on the right."""
    st.subheader("Top Relevant Chunks")

    # Filter chunks with positive scores and sort them by score
    relevant_chunks = sorted(
        (chunk for chunk in chunks if chunk.metadata.get('rerank_score', 0) > 0),
        key=lambda x: x.metadata.get('rerank_score', 0),
        reverse=True
    )

    if not relevant_chunks:
        st.warning("No relevant chunks to display.")
        return

    for i, chunk in enumerate(relevant_chunks[:k], 1):
        score = chunk.metadata.get('rerank_score', 0)
        st.markdown(f"### Chunk {i} (Score: {score:.4f})")

        col1, col2 = st.columns([1.5, 1])  # Left column wider for chunk text
        with col1:
            st.text_area("Text Content", chunk.page_content, height=400, key=f"text_{i}")
            st.json({
                "Filename": chunk.metadata.get('filename', 'Unknown'),
                "Page Number": chunk.metadata.get('page_number', 'Unknown'),
                "Rerank Score": f"{score:.4f}"
            })

        with col2:
            if 'page_number' in chunk.metadata and 'document_id' in chunk.metadata:
                doc_id, page_num = chunk.metadata['document_id'], chunk.metadata['page_number']
                if doc_id < len(pdf_files):
                    pdf_file = pdf_files[doc_id]
                    pdf_file.seek(0)
                    page_pdf = extract_page_from_pdf(io.BytesIO(pdf_file.read()), page_num)
                    if page_pdf:
                        st.markdown("PDF Preview:")
                        show_pdf_page(page_pdf)  # Reuse modular function
                        st.download_button(
                            label=f"📥 Download Page {page_num}",
                            data=page_pdf,
                            file_name=f"page_{page_num}_{chunk.metadata.get('filename', 'doc')}.pdf",
                            mime="application/pdf",
                            key=f"download_{i}_{doc_id}_{page_num}"  # Unique key for each button
                        )
        st.markdown("---")

def handle_user_input(user_question: str, pdf_files: List[io.BytesIO]):
    if st.session_state.conversation is not None:
        response = st.session_state.conversation({'question': user_question})
        st.write(f"🤖 **Assistant:** {response['answer']}")
        st.write("---")
        if 'source_documents' in response:
            display_chunks(response['source_documents'], pdf_files, k=st.session_state.get('k_chunks', 5))

def main():
    configure_api()
    st.set_page_config(page_title='Document Chat', layout='wide')
    st.header("💬 Chat with Your Documents")

    # Initialize session states
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'k_chunks' not in st.session_state:
        st.session_state.k_chunks = 5
    if 'pdf_files' not in st.session_state:
        st.session_state.pdf_files = []
    if 'chunks_info' not in st.session_state:
        st.session_state.chunks_info = None

    # Sidebar for document management
    with st.sidebar:
        st.subheader("Document Management")

        # File upload section for initial files
        pdf_files = st.file_uploader("Upload PDFs", type=['pdf'], accept_multiple_files=True)
        if pdf_files:
            st.session_state.pdf_files = pdf_files

        # Document processing section
        if st.button("Process Documents"):
            with st.spinner("Processing..."):
                try:
                    if not pdf_files:
                        st.warning("Please upload PDF files first!")
                        return

                    # Process documents
                    raw_text = get_pdf_text(pdf_files)
                    text_chunks = get_chunks(raw_text, pdf_files)

                    # Create or update vectorstore
                    hybrid_retriever, vectorstore = create_hybrid_retriever(
                        text_chunks
                    )
                    st.session_state.vectorstore = vectorstore
                    st.session_state.conversation = create_conversation_chain(hybrid_retriever)

                    # Update chunk statistics
                    if 'chunks_info' not in st.session_state or st.session_state.chunks_info is None:
                        st.session_state.chunks_info = {'total_chunks': 0, 'chunks_per_doc': {}, 'avg_chunk_size': 0}

                    for chunk in text_chunks:
                        doc_id = chunk.metadata['filename']
                        if doc_id not in st.session_state.chunks_info['chunks_per_doc']:
                            st.session_state.chunks_info['chunks_per_doc'][doc_id] = 0
                        st.session_state.chunks_info['chunks_per_doc'][doc_id] += 1

                    st.session_state.chunks_info['total_chunks'] += len(text_chunks)
                    total_chunk_sizes = sum(len(chunk.page_content) for chunk in text_chunks)
                    st.session_state.chunks_info['avg_chunk_size'] = total_chunk_sizes / st.session_state.chunks_info['total_chunks']

                    st.success("Documents processed successfully!")
                except Exception as e:
                    st.error(f"Error processing documents: {e}")

        # # Chunk display settings
        # st.session_state.k_chunks = st.slider(
        #     "Chunks to display",
        #     min_value=1,
        #     max_value=10,
        #     value=st.session_state.k_chunks
        # )

        # Display chunking statistics if available
        if st.session_state.chunks_info:
            st.subheader("📊 Chunking Statistics")
            st.write(f"Total chunks: {st.session_state.chunks_info['total_chunks']}")
            st.write(f"Average chunk size: {st.session_state.chunks_info['avg_chunk_size']:.0f} characters")
            st.write("Chunks per document:")
            for doc, count in st.session_state.chunks_info['chunks_per_doc'].items():
                st.write(f"- {doc}: {count} chunks")

    # Main chat interface
    if st.session_state.conversation is None:
        st.info("👈 Upload documents and process them to start chatting!")
    else:
        # Chat interface
        user_question = st.text_input("Ask a question about your documents")
        if user_question:
            with st.spinner("Generating response..."):
                handle_user_input(user_question, st.session_state.pdf_files)

        # Chat control buttons
        if st.button("Clear Chat History"):
            st.session_state.conversation.memory.clear()
            st.success("Chat history cleared!")

        if st.button("Clear All Data"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("All data cleared! Please refresh the page.")

    # Footer
    st.markdown("---")
    st.markdown("💡 *Tip: You can upload multiple PDF documents and chat with them all at once!*")

if __name__ == "__main__":
    main()
