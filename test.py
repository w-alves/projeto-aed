import os
import re
import json
import fitz  # PyMuPDF
import glob
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from openai import OpenAI
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

@dataclass
class DocumentMetadata:
    """Metadata for a document."""
    filename: str
    title: str
    author: str = ""
    date: str = ""
    num_pages: int = 0
    file_path: str = ""

@dataclass
class DocumentSection:
    """Section of a document with hierarchical structure."""
    title: str
    level: int
    content: str
    page_number: int
    section_id: str

@dataclass
class Document:
    """Normalized document structure with metadata and content."""
    metadata: DocumentMetadata
    sections: List[DocumentSection]
    
    def to_dict(self):
        return {
            "metadata": asdict(self.metadata),
            "sections": [asdict(section) for section in self.sections]
        }

class PDFProcessor:
    """Process PDF files to extract structured content."""
    
    def __init__(self):
        """Initialize the PDF processor."""
        pass
        
    def extract_metadata(self, pdf_document, file_path: str) -> DocumentMetadata:
        """Extract metadata from a PDF document."""
        metadata = pdf_document.metadata
        
        # Extract title from metadata or first page if not available
        title = metadata.get("title", "")
        if not title:
            # Try to extract title from first page
            if pdf_document.page_count > 0:
                first_page_text = pdf_document[0].get_text()
                lines = first_page_text.strip().split('\n')
                if lines:
                    title = lines[0].strip()
        
        # If still no title, use filename
        if not title:
            title = os.path.basename(file_path)
            title = os.path.splitext(title)[0]  # Remove extension
        
        return DocumentMetadata(
            filename=os.path.basename(file_path),
            title=title,
            author=metadata.get("author", ""),
            date=metadata.get("creationDate", ""),
            num_pages=pdf_document.page_count,
            file_path=file_path
        )
    
    def _is_heading(self, text: str) -> bool:
        """Heuristically determine if text is a heading."""
        # Simple heuristics for headers
        if not text.strip():
            return False
        
        # Check if less than 100 characters and ends without period
        if len(text) < 100 and not text.strip().endswith('.'):
            return True
            
        # Check for common heading patterns
        heading_patterns = [
            r'^Chapter \d+',
            r'^Section \d+',
            r'^\d+\.\d+',
            r'^\d+\.\d+\.\d+',
            r'^[A-Z][A-Z\s]+$',  # All caps heading
        ]
        
        for pattern in heading_patterns:
            if re.match(pattern, text.strip()):
                return True
                
        return False
    
    def _determine_heading_level(self, text: str) -> int:
        """Determine the heading level based on the text."""
        # Basic implementation - more sophisticated logic could be added
        if re.match(r'^Chapter \d+', text):
            return 1
        elif re.match(r'^Section \d+', text):
            return 2
        elif re.match(r'^\d+\.\d+\.\d+', text):
            return 3
        elif re.match(r'^\d+\.\d+', text):
            return 2
        elif re.match(r'^\d+\.', text):
            return 1
        return 2  # Default heading level
    
    def _extract_sections(self, pdf_document) -> List[DocumentSection]:
        """Extract sections with hierarchical structure from PDF."""
        sections = []
        current_section = None
        current_content = []
        section_id_counter = 0
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            blocks = page.get_text("blocks")
            
            for block in blocks:
                text = block[4].strip()
                if not text:
                    continue
                    
                if self._is_heading(text):
                    # Save previous section if exists
                    if current_section:
                        section_content = "\n".join(current_content)
                        sections.append(DocumentSection(
                            title=current_section,
                            level=current_level,
                            content=section_content,
                            page_number=current_page,
                            section_id=f"section_{section_id_counter}"
                        ))
                        section_id_counter += 1
                        
                    # Start new section
                    current_section = text
                    current_level = self._determine_heading_level(text)
                    current_content = []
                    current_page = page_num + 1
                else:
                    # Add to current section content
                    current_content.append(text)
                    
        # Add the last section
        if current_section:
            section_content = "\n".join(current_content)
            sections.append(DocumentSection(
                title=current_section,
                level=current_level,
                content=section_content,
                page_number=current_page,
                section_id=f"section_{section_id_counter}"
            ))
            
        # If no sections were identified, create a default one
        if not sections:
            all_text = "\n".join(page.get_text() for page in pdf_document)
            sections.append(DocumentSection(
                title="Document Content",
                level=0,
                content=all_text,
                page_number=1,
                section_id="section_0"
            ))
            
        return sections
    
    def process_pdf(self, file_path: str) -> Optional[Document]:
        """Process a PDF file and return a structured document."""
        try:
            pdf_document = fitz.open(file_path)
            metadata = self.extract_metadata(pdf_document, file_path)
            sections = self._extract_sections(pdf_document)
            return Document(metadata=metadata, sections=sections)
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            return None


class DocumentIndexer:
    """Index documents for quick search."""
    
    def __init__(self, documents_dir: str, index_file: str = "document_index.json"):
        """Initialize indexer with documents directory and index file path."""
        self.documents_dir = documents_dir
        self.index_file = index_file
        self.processor = PDFProcessor()
        self.document_index = {}
        self.client = OpenAI()
        
    def index_documents(self, force_reindex: bool = False) -> Dict:
        """Index all PDF documents in the directory."""
        if not force_reindex and os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r') as f:
                    self.document_index = json.load(f)
                logger.info(f"Loaded existing index with {len(self.document_index)} documents.")
                return self.document_index
            except Exception as e:
                logger.error(f"Error loading index file: {str(e)}")
                
        # Create new index
        self.document_index = {}
        pdf_files = glob.glob(os.path.join(self.documents_dir, "**/*.pdf"), recursive=True)
        
        logger.info(f"Indexing {len(pdf_files)} PDF documents...")
        for file_path in tqdm(pdf_files):
            try:
                document = self.processor.process_pdf(file_path)
                if document:
                    # Store document
                    doc_id = os.path.relpath(file_path, self.documents_dir)
                    self.document_index[doc_id] = document.to_dict()
            except Exception as e:
                logger.error(f"Error indexing {file_path}: {str(e)}")
                
        # Save index to file
        with open(self.index_file, 'w') as f:
            json.dump(self.document_index, f)
            
        logger.info(f"Indexed {len(self.document_index)} documents successfully.")
        return self.document_index

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI API."""
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    def create_embeddings_index(self, force_rebuild: bool = False) -> Dict:
        """Create embeddings for document sections."""
        embeddings_file = "document_embeddings.json"
        
        if not force_rebuild and os.path.exists(embeddings_file):
            try:
                with open(embeddings_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading embeddings: {str(e)}")
        
        if not self.document_index:
            self.index_documents()
            
        embeddings_index = {}
        
        for doc_id, doc_data in tqdm(self.document_index.items(), desc="Creating embeddings"):
            try:
                # Create embedding for document title
                title_key = f"{doc_id}:title"
                title_text = doc_data["metadata"]["title"]
                embeddings_index[title_key] = {
                    "embedding": self.get_embedding(title_text),
                    "text": title_text,
                    "doc_id": doc_id,
                    "type": "title"
                }
                
                # Create embeddings for each section
                for section in doc_data["sections"]:
                    section_key = f"{doc_id}:{section['section_id']}"
                    # Create embedding for section title
                    section_title_key = f"{section_key}:title"
                    embeddings_index[section_title_key] = {
                        "embedding": self.get_embedding(section["title"]),
                        "text": section["title"],
                        "doc_id": doc_id,
                        "section_id": section["section_id"],
                        "type": "section_title"
                    }
                    
                    # Split section content into chunks for embedding
                    content = section["content"]
                    sentences = sent_tokenize(content)
                    
                    # Create chunks of approximately 1000 characters
                    chunks = []
                    current_chunk = []
                    current_length = 0
                    
                    for sentence in sentences:
                        if current_length + len(sentence) > 1000:
                            chunks.append(" ".join(current_chunk))
                            current_chunk = [sentence]
                            current_length = len(sentence)
                        else:
                            current_chunk.append(sentence)
                            current_length += len(sentence)
                            
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                    
                    # Create embeddings for chunks
                    for i, chunk in enumerate(chunks):
                        chunk_key = f"{section_key}:chunk{i}"
                        embeddings_index[chunk_key] = {
                            "embedding": self.get_embedding(chunk),
                            "text": chunk,
                            "doc_id": doc_id,
                            "section_id": section["section_id"],
                            "type": "content",
                            "chunk_id": i
                        }
            except Exception as e:
                logger.error(f"Error creating embeddings for {doc_id}: {str(e)}")
                
        # Save embeddings
        with open(embeddings_file, 'w') as f:
            json.dump(embeddings_index, f)
            
        logger.info(f"Created embeddings for {len(embeddings_index)} items.")
        return embeddings_index


class DocumentSearchEngine:
    """Search engine for finding documents and content."""
    
    def __init__(self, documents_dir: str):
        """Initialize the search engine."""
        self.documents_dir = documents_dir
        self.indexer = DocumentIndexer(documents_dir)
        self.document_index = {}
        self.embeddings_index = {}
        self.client = OpenAI()
        
    def initialize(self, force_reindex: bool = False):
        """Initialize search engine by loading or creating indexes."""
        self.document_index = self.indexer.index_documents(force_reindex)
        self.embeddings_index = self.indexer.create_embeddings_index(force_reindex)
        logger.info("Search engine initialized successfully.")
        
    def search_by_filename(self, query: str) -> List[Dict]:
        """Search documents by filename or title."""
        results = []
        query_lower = query.lower()
        
        for doc_id, doc_data in self.document_index.items():
            filename = os.path.basename(doc_id).lower()
            title = doc_data["metadata"]["title"].lower()
            
            # Calculate simple relevance score based on string matching
            filename_score = 1.0 if query_lower in filename else 0.0
            title_score = 1.0 if query_lower in title else 0.0
            
            if filename_score > 0 or title_score > 0:
                results.append({
                    "doc_id": doc_id,
                    "title": doc_data["metadata"]["title"],
                    "filename": os.path.basename(doc_id),
                    "score": max(filename_score, title_score),
                    "metadata": doc_data["metadata"]
                })
                
        # Sort by relevance score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
        
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search documents by semantic similarity."""
        query_embedding = self.indexer.get_embedding(query)
        
        results = []
        for key, data in self.embeddings_index.items():
            embedding = data["embedding"]
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            
            results.append({
                "key": key,
                "text": data["text"],
                "doc_id": data["doc_id"],
                "type": data["type"],
                "section_id": data.get("section_id"),
                "score": float(similarity)
            })
            
        # Sort by similarity and return top results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
        
    def get_document_section(self, doc_id: str, section_id: str) -> Dict:
        """Get a specific section from a document."""
        if doc_id in self.document_index:
            doc_data = self.document_index[doc_id]
            for section in doc_data["sections"]:
                if section["section_id"] == section_id:
                    return {
                        "title": section["title"],
                        "content": section["content"],
                        "page_number": section["page_number"],
                        "doc_id": doc_id,
                        "doc_title": doc_data["metadata"]["title"]
                    }
        return None
        
    def get_document_context(self, doc_id: str, section_id: str = None) -> Dict:
        """Get document and optionally section context."""
        if doc_id in self.document_index:
            doc_data = self.document_index[doc_id]
            result = {
                "metadata": doc_data["metadata"],
                "sections": []
            }
            
            if section_id:
                # Get specific section and adjacent sections for context
                section_indices = {s["section_id"]: i for i, s in enumerate(doc_data["sections"])}
                if section_id in section_indices:
                    idx = section_indices[section_id]
                    
                    # Get the target section and possibly adjacent sections
                    start_idx = max(0, idx - 1)
                    end_idx = min(len(doc_data["sections"]), idx + 2)
                    result["sections"] = doc_data["sections"][start_idx:end_idx]
                    result["focus_section_id"] = section_id
            else:
                # Return all sections if no specific section requested
                result["sections"] = doc_data["sections"]
                
            return result
        return None


class SearchAgent:
    """Agent for searching PDF documents using a multi-step approach."""
    
    def __init__(self, documents_dir: str):
        """Initialize the search agent."""
        self.search_engine = DocumentSearchEngine(documents_dir)
        self.search_engine.initialize()
        self.client = OpenAI()
        
    def _evaluate_search_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """Use LLM to evaluate search results relevance to query."""
        if not results:
            return []
            
        # Prepare context for the LLM
        context = f"Query: {query}\n\nSearch Results:\n"
        for i, result in enumerate(results[:5]):  # Limit to top 5 for LLM evaluation
            result_text = result.get("text", "")
            # Truncate text if too long
            if len(result_text) > 200:
                result_text = result_text[:200] + "..."
                
            context += f"Result {i+1}:\n"
            context += f"Document: {result['doc_id']}\n"
            context += f"Type: {result['type']}\n"
            context += f"Text: {result_text}\n\n"
            
        # Ask LLM to evaluate
        prompt = f"""
        {context}
        
        Please evaluate the relevance of each search result to the query on a scale of 0-10.
        Return your evaluation as a JSON array with scores, where each item has:
        - result_index: the index of the result (1-based)
        - relevance_score: 0-10 score of relevance
        - reasoning: brief explanation of the score
        
        JSON format only (no other text):
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            evaluation = json.loads(response.choices[0].message.content)
            
            # Add LLM scores to results
            scored_results = []
            if "evaluations" in evaluation:
                scores = {item["result_index"]: item["relevance_score"] for item in evaluation["evaluations"]}
                
                for i, result in enumerate(results[:5]):
                    idx = i + 1  # 1-based index
                    if idx in scores:
                        result["llm_score"] = scores[idx]
                        result["score"] = (result["score"] + scores[idx]/10) / 2  # Average normalized scores
                    scored_results.append(result)
                    
                # Add remaining results
                scored_results.extend(results[5:])
            else:
                scored_results = results
                
            # Re-sort based on combined scores
            scored_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            return scored_results
            
        except Exception as e:
            logger.error(f"Error in LLM evaluation: {str(e)}")
            return results
    
    def search(self, query: str, max_iterations: int = 3) -> Dict:
        """Search for documents and content related to query."""
        logger.info(f"Starting search for: {query}")
        
        # Step 1: Search by filename/title first
        filename_results = self.search_engine.search_by_filename(query)
        
        if filename_results:
            logger.info(f"Found {len(filename_results)} results by filename/title.")
            best_match = filename_results[0]
            
            # If we have a strong filename match, get the document content
            if best_match["score"] >= 0.8:
                doc_context = self.search_engine.get_document_context(best_match["doc_id"])
                
                # If the document is found, do a semantic search within it
                if doc_context:
                    logger.info(f"Found document by name: {best_match['title']}")
                    return {
                        "found_by": "filename",
                        "document": best_match,
                        "context": doc_context,
                        "query": query
                    }
        
        # Step 2: Semantic search across all documents
        logger.info("Performing semantic search across all documents.")
        semantic_results = self.search_engine.semantic_search(query, top_k=10)
        
        # Use LLM to evaluate results
        scored_results = self._evaluate_search_results(query, semantic_results)
        
        if scored_results:
            best_match = scored_results[0]
            
            # If we have a good semantic match
            if best_match["score"] >= 0.7:
                doc_id = best_match["doc_id"]
                section_id = best_match.get("section_id")
                
                doc_context = self.search_engine.get_document_context(doc_id, section_id)
                
                if doc_context:
                    logger.info(f"Found relevant content in document: {doc_id}")
                    return {
                        "found_by": "semantic",
                        "document": {
                            "doc_id": doc_id,
                            "title": doc_context["metadata"]["title"],
                            "score": best_match["score"]
                        },
                        "context": doc_context,
                        "query": query
                    }
        
        # Step 3: Progressive search through documents
        logger.info("Starting progressive document search.")
        all_examined = set()
        
        for iteration in range(max_iterations):
            logger.info(f"Search iteration {iteration+1}/{max_iterations}")
            
            # Skip documents we've already examined
            current_results = [r for r in semantic_results if r["doc_id"] not in all_examined]
            
            if not current_results:
                break
                
            # Take the best candidate
            best_candidate = current_results[0]
            doc_id = best_candidate["doc_id"]
            all_examined.add(doc_id)
            
            # Get document context
            doc_context = self.search_engine.get_document_context(doc_id)
            
            if not doc_context:
                continue
                
            # Evaluate if this document answers the query
            doc_text = doc_context["metadata"]["title"] + "\n"
            doc_text += "\n".join(section["title"] + "\n" + section["content"][:500] 
                                for section in doc_context["sections"][:3])
            
            # Truncate if too long
            if len(doc_text) > 4000:
                doc_text = doc_text[:4000] + "..."
                
            prompt = f"""
            Query: {query}
            
            Document excerpt: 
            {doc_text}
            
            Does this document contain information relevant to the query?
            Rate relevance from 0-10 and explain why.
            
            Answer in JSON format:
            {{
                "relevance_score": [0-10],
                "explanation": "Your explanation",
                "contains_answer": true/false
            }}
            """
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                
                evaluation = json.loads(response.choices[0].message.content)
                
                if "relevance_score" in evaluation and evaluation["relevance_score"] >= 7:
                    # If document is relevant, do a more focused search within it
                    section_results = []
                    
                    for section in doc_context["sections"]:
                        section_text = section["title"] + "\n" + section["content"]
                        # Calculate simple relevance using term frequency
                        query_terms = query.lower().split()
                        text_lower = section_text.lower()
                        
                        term_matches = sum(1 for term in query_terms if term in text_lower)
                        section_score = term_matches / len(query_terms) if query_terms else 0
                        
                        section_results.append({
                            "section_id": section["section_id"],
                            "title": section["title"],
                            "score": section_score,
                            "page_number": section["page_number"]
                        })
                    
                    # Sort sections by relevance
                    section_results.sort(key=lambda x: x["score"], reverse=True)
                    
                    if section_results:
                        best_section = section_results[0]
                        
                        # Get context with focus on best section
                        focused_context = self.search_engine.get_document_context(
                            doc_id, best_section["section_id"]
                        )
                        
                        logger.info(f"Found relevant content in document: {doc_id}, "
                                   f"section: {best_section['title']}")
                        
                        return {
                            "found_by": "progressive",
                            "document": {
                                "doc_id": doc_id,
                                "title": doc_context["metadata"]["title"],
                                "score": evaluation["relevance_score"] / 10
                            },
                            "section": best_section,
                            "context": focused_context,
                            "query": query
                        }
            except Exception as e:
                logger.error(f"Error evaluating document: {str(e)}")
        
        # If we get here, we didn't find a good match
        logger.info("No relevant documents found.")
        return {
            "found_by": "none",
            "query": query,
            "message": "No relevant documents found after exhaustive search."
        }
    
    def format_search_results(self, search_result: Dict) -> str:
        """Format search results for human-readable output."""
        if search_result["found_by"] == "none":
            return "No relevant documents found for your query."
            
        output = []
        output.append(f"Search results for: {search_result['query']}\n")
        
        if "document" in search_result:
            doc = search_result["document"]
            output.append(f"📄 Document: {doc['title']}")
            output.append(f"   File: {doc['doc_id']}")
            output.append(f"   Relevance: {doc['score']:.2f}")
            output.append("")
        
        if "context" in search_result and "sections" in search_result["context"]:
            focus_section_id = search_result["context"].get("focus_section_id")
            
            for section in search_result["context"]["sections"]:
                is_focus = section["section_id"] == focus_section_id
                
                if is_focus:
                    output.append(f"🔍 Section: {section['title']} (Page {section['page_number']})")
                    
                    # Add section content, truncate if too long
                    content = section["content"]
                    if len(content) > 800:
                        content = content[:800] + "...(truncated)"
                    
                    output.append(f"\n{content}\n")
                else:
                    output.append(f"📌 Related Section: {section['title']} (Page {section['page_number']})")
        
        return "\n".join(output)


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PDF Document Search Engine")
    parser.add_argument("--dir", required=True, help="Directory containing PDF documents")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--reindex", action="store_true", help="Force reindexing of documents")
    
    args = parser.parse_args()
    
    agent = SearchAgent(args.dir)
    
    if args.query:
        results = agent.search(args.query)
        print(agent.format_search_results(results))
    else:
        # Interactive mode
        print("PDF Search Engine initialized. Enter your queries (or 'quit' to exit):")
        while True:
            query = input("\nQuery: ")
            if query.lower() in ("quit", "exit"):
                break
                
            results = agent.search(query)
            print("\n" + agent.format_search_results(results))
