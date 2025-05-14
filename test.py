import os
import re
import json
import fitz  # PyMuPDF
import glob
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict, field
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

# Available metatags
AVAILABLE_METATAGS = [
    "FINANCIAL_METRICS", 
    "PERFORMANCE_DRIVERS", 
    "GUIDANCE_AND_OUTLOOK",
    "SEGMENT_ANALYSIS", 
    "MARGIN_ANALYSIS", 
    "CORPORATE_ACTIONS",
    "MANAGEMENT_COMMENTARY", 
    "RISKS_AND_HEADWINDS", 
    "OPPORTUNITIES",
    "VALUATION_AND_MARKET_REACTION", 
    "MACRO_AND_MARKET_COLOR",
    "TECHNICAL_INDICATORS", 
    "ANALYST_CONSENSUS", 
    "CAPITAL_ALLOCATION"
]

@dataclass
class DocumentMetadata:
    """Metadata for a document."""
    filename: str
    title: str
    author: str = ""
    date: str = ""
    year: str = ""
    company: str = ""
    num_pages: int = 0
    file_path: str = ""
    metatags: List[str] = field(default_factory=list)

@dataclass
class DocumentChunk:
    """A chunk of document text."""
    content: str
    page_number: int
    chunk_id: str
    metatags: List[str] = field(default_factory=list)

@dataclass
class Document:
    """Normalized document structure with metadata and content."""
    metadata: DocumentMetadata
    chunks: List[DocumentChunk]
    
    def to_dict(self):
        return {
            "metadata": asdict(self.metadata),
            "chunks": [asdict(chunk) for chunk in self.chunks]
        }

class PDFProcessor:
    """Process PDF files to extract structured content."""
    
    def __init__(self, client: OpenAI):
        """Initialize the PDF processor."""
        self.client = client
        
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
        
        # Extract year from metadata creation date or file name
        year = ""
        date = metadata.get("creationDate", "")
        if date and len(date) >= 4:
            # Try to extract year from metadata date string
            year_match = re.search(r'(\d{4})', date)
            if year_match:
                year = year_match.group(1)
        
        # If year not found in metadata, try to extract from filename or title
        if not year:
            # Look for a year pattern in filename or title
            year_pattern = re.compile(r'(19|20)\d{2}')
            filename = os.path.basename(file_path)
            year_match = year_pattern.search(filename) or year_pattern.search(title)
            if year_match:
                year = year_match.group(0)
        
        # Try to extract company name from title or filename
        company = ""
        
        return DocumentMetadata(
            filename=os.path.basename(file_path),
            title=title,
            author=metadata.get("author", ""),
            date=metadata.get("creationDate", ""),
            year=year,
            company=company,
            num_pages=pdf_document.page_count,
            file_path=file_path,
            metatags=[]
        )
    
    def extract_company_and_metatags(self, text: str, metadata: DocumentMetadata) -> Tuple[str, List[str]]:
        """Extract company name and metatags using LLM."""
        # Extract first 2000 characters for analysis
        sample_text = text[:2000] if len(text) > 2000 else text
        
        prompt = f"""
        Please analyze this document text to extract:
        1. Company name (if any)
        2. Relevant topic tags from the following list:
        {', '.join(AVAILABLE_METATAGS)}
        
        Document title: {metadata.title}
        Document year: {metadata.year if metadata.year else 'Unknown'}
        
        Sample text from document:
        ```
        {sample_text}
        ```
        
        Return your analysis in JSON format only:
        {{
            "company": "Company name or empty string if none found",
            "metatags": ["TAG1", "TAG2", ...]
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            analysis = json.loads(response.choices[0].message.content)
            return analysis.get("company", ""), analysis.get("metatags", [])
        except Exception as e:
            logger.error(f"Error in LLM analysis: {str(e)}")
            return "", []
    
    def extract_paragraph_metatags(self, paragraph: str, doc_tags: List[str]) -> List[str]:
        """Extract metatags for a specific paragraph using LLM."""
        # Truncate paragraph if too long
        sample_text = paragraph[:1000] if len(paragraph) > 1000 else paragraph
        
        prompt = f"""
        Please analyze this paragraph to determine which of the following tags apply:
        {', '.join(AVAILABLE_METATAGS)}
        
        Document already has these tags: {', '.join(doc_tags)}
        
        Paragraph:
        ```
        {sample_text}
        ```
        
        Return only a JSON array of applicable tags:
        ["TAG1", "TAG2", ...]
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            if isinstance(result, list):
                return result
            elif "tags" in result:
                return result.get("tags", [])
            else:
                return []
        except Exception as e:
            logger.error(f"Error in paragraph LLM analysis: {str(e)}")
            return []
    
    def extract_paragraphs(self, text: str) -> List[str]:
        """Extract paragraphs from text."""
        # Split by double newlines to identify paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Filter out empty paragraphs
        return [p.strip() for p in paragraphs if p.strip()]
    
    def process_pdf(self, file_path: str) -> Optional[Document]:
        """Process a PDF file and return a structured document with paragraph chunks."""
        try:
            pdf_document = fitz.open(file_path)
            metadata = self.extract_metadata(pdf_document, file_path)
            
            # Extract full text from PDF
            full_text = ""
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                full_text += page.get_text() + "\n\n"
            
            # Extract company and document-level metatags
            company, doc_metatags = self.extract_company_and_metatags(full_text, metadata)
            
            # Update metadata with company and metatags
            metadata.company = company
            metadata.metatags = doc_metatags
            
            logger.info(f"Extracted metadata for {file_path}: Company={company}, Tags={doc_metatags}")
            
            # Process document by pages and extract paragraphs
            chunks = []
            chunk_id = 0
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                page_text = page.get_text()
                
                # Extract paragraphs from page
                paragraphs = self.extract_paragraphs(page_text)
                
                # Process each paragraph
                for paragraph in paragraphs:
                    # Skip very short paragraphs (likely noise)
                    if len(paragraph.split()) < 5:
                        continue
                        
                    # Get paragraph-specific tags using LLM
                    # Note: In production, you might want to batch these calls or process async
                    # to avoid too many API calls
                    paragraph_tags = []  # Placeholder - would make LLM call here
                    
                    # In production, uncomment below to get paragraph-specific tags
                    # paragraph_tags = self.extract_paragraph_metatags(paragraph, doc_metatags)
                    
                    # Combine document-level tags with paragraph-specific tags
                    all_tags = list(set(doc_metatags + paragraph_tags))
                    
                    # Add document-level metadata as tags
                    if metadata.company:
                        all_tags.append(f"COMPANY:{metadata.company}")
                    if metadata.year:
                        all_tags.append(f"YEAR:{metadata.year}")
                    
                    chunks.append(DocumentChunk(
                        content=paragraph,
                        page_number=page_num + 1,
                        chunk_id=f"chunk_{chunk_id}",
                        metatags=all_tags
                    ))
                    
                    chunk_id += 1
            
            # If no chunks were created, create one with the whole document
            if not chunks and full_text:
                chunks.append(DocumentChunk(
                    content=full_text,
                    page_number=1,
                    chunk_id="chunk_0",
                    metatags=metadata.metatags
                ))
            
            return Document(metadata=metadata, chunks=chunks)
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            return None


class DocumentIndexer:
    """Index documents for quick search."""
    
    def __init__(self, documents_dir: str, index_file: str = "document_index.json"):
        """Initialize indexer with documents directory and index file path."""
        self.documents_dir = documents_dir
        self.index_file = index_file
        self.client = OpenAI()
        self.processor = PDFProcessor(self.client)
        self.document_index = {}
        
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
        """Create embeddings for document chunks."""
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
                    "type": "title",
                    "metatags": doc_data["metadata"].get("metatags", [])
                }
                
                # Create embeddings for each chunk
                for chunk in doc_data["chunks"]:
                    chunk_key = f"{doc_id}:{chunk['chunk_id']}"
                    chunk_text = chunk["content"]
                    
                    # Skip empty chunks
                    if not chunk_text.strip():
                        continue
                    
                    embeddings_index[chunk_key] = {
                        "embedding": self.get_embedding(chunk_text),
                        "text": chunk_text,
                        "doc_id": doc_id,
                        "chunk_id": chunk["chunk_id"],
                        "page_number": chunk["page_number"],
                        "type": "content",
                        "metatags": chunk["metatags"]
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
        self.client = OpenAI()
        self.indexer = DocumentIndexer(documents_dir)
        self.document_index = {}
        self.embeddings_index = {}
        
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
    
    def identify_relevant_tags(self, query: str) -> List[str]:
        """Use LLM to identify relevant tags for the query."""
        prompt = f"""
        Given a user query, identify which of the following document tags might be relevant.
        Choose ONLY tags that directly relate to the query content.

        Available tags:
        {', '.join(AVAILABLE_METATAGS)}
        
        User query: "{query}"
        
        Return only a JSON array of relevant tags (empty array if none apply):
        ["TAG1", "TAG2", ...]
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            if isinstance(result, list):
                return result
            elif "tags" in result:
                return result.get("tags", [])
            else:
                return []
        except Exception as e:
            logger.error(f"Error identifying tags: {str(e)}")
            return []
    
    def extract_search_metadata(self, query: str) -> Dict:
        """Extract search metadata like company names, years from query."""
        prompt = f"""
        Extract specific search metadata from this query. Look for:
        1. Company names
        2. Years or time periods
        3. Any specific document types mentioned
        
        Query: "{query}"
        
        Return in JSON format:
        {{
            "companies": ["Company1", "Company2", ...],
            "years": ["2023", "2022", ...],
            "document_types": ["annual report", "earnings call", ...]
        }}
        
        Return empty arrays if none found.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error extracting search metadata: {str(e)}")
            return {"companies": [], "years": [], "document_types": []}
    
    def filter_by_tags(self, embeddings_keys: List[str], required_tags: List[str]) -> List[str]:
        """Filter embedding keys by required tags."""
        if not required_tags:
            return embeddings_keys
            
        filtered_keys = []
        for key in embeddings_keys:
            item = self.embeddings_index.get(key)
            if not item:
                continue
                
            item_tags = item.get("metatags", [])
            # Check if any required tags are in the item's tags
            if any(tag in item_tags for tag in required_tags):
                filtered_keys.append(key)
                
        return filtered_keys
    
    def filter_by_metadata(self, embeddings_keys: List[str], search_metadata: Dict) -> List[str]:
        """Filter embedding keys by extracted metadata like company, year."""
        if not search_metadata or not any(search_metadata.values()):
            return embeddings_keys
            
        companies = [c.lower() for c in search_metadata.get("companies", [])]
        years = search_metadata.get("years", [])
        
        filtered_keys = []
        for key in embeddings_keys:
            item = self.embeddings_index.get(key)
            if not item:
                continue
                
            item_tags = item.get("metatags", [])
            
            # Check for company matches
            company_match = not companies or any(
                any(company in tag.lower() for tag in item_tags)
                for company in companies
            )
            
            # Check for year matches
            year_match = not years or any(
                any(year in tag for tag in item_tags)
                for year in years
            )
            
            if company_match and year_match:
                filtered_keys.append(key)
                
        return filtered_keys if filtered_keys else embeddings_keys  # Fall back to original if no matches
        
    def semantic_search(self, query: str, top_k: int = 5, filter_tags: List[str] = None, 
                        search_metadata: Dict = None) -> List[Dict]:
        """Search documents by semantic similarity with tag and metadata filtering."""
        query_embedding = self.indexer.get_embedding(query)
        
        # Start with all keys
        all_keys = list(self.embeddings_index.keys())
        
        # Apply tag filtering if specified
        if filter_tags:
            all_keys = self.filter_by_tags(all_keys, filter_tags)
            
        # Apply metadata filtering if specified
        if search_metadata:
            all_keys = self.filter_by_metadata(all_keys, search_metadata)
            
        results = []
        for key in all_keys:
            data = self.embeddings_index[key]
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
                "chunk_id": data.get("chunk_id"),
                "page_number": data.get("page_number"),
                "score": float(similarity),
                "metatags": data.get("metatags", [])
            })
            
        # Sort by similarity and return top results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
        
    def get_document_chunk(self, doc_id: str, chunk_id: str) -> Dict:
        """Get a specific chunk from a document."""
        if doc_id in self.document_index:
            doc_data = self.document_index[doc_id]
            for chunk in doc_data["chunks"]:
                if chunk["chunk_id"] == chunk_id:
                    return {
                        "content": chunk["content"],
                        "page_number": chunk["page_number"],
                        "doc_id": doc_id,
                        "doc_title": doc_data["metadata"]["title"],
                        "metatags": chunk["metatags"]
                    }
        return None
        
    def get_document_context(self, doc_id: str, chunk_id: str = None) -> Dict:
        """Get document and optionally chunk context."""
        if doc_id in self.document_index:
            doc_data = self.document_index[doc_id]
            result = {
                "metadata": doc_data["metadata"],
                "chunks": []
            }
            
            if chunk_id:
                # Get specific chunk and adjacent chunks for context
                chunk_indices = {c["chunk_id"]: i for i, c in enumerate(doc_data["chunks"])}
                if chunk_id in chunk_indices:
                    idx = chunk_indices[chunk_id]
                    
                    # Get the target chunk and possibly adjacent chunks
                    start_idx = max(0, idx - 1)
                    end_idx = min(len(doc_data["chunks"]), idx + 2)
                    result["chunks"] = doc_data["chunks"][start_idx:end_idx]
                    result["focus_chunk_id"] = chunk_id
            else:
                # Return all chunks if no specific chunk requested
                result["chunks"] = doc_data["chunks"]
                
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
        
        # Step 1: Identify relevant tags and metadata
        relevant_tags = self.search_engine.identify_relevant_tags(query)
        search_metadata = self.search_engine.extract_search_metadata(query)
        
        logger.info(f"Identified relevant tags: {relevant_tags}")
        logger.info(f"Extracted search metadata: {search_metadata}")
        
        # Step 2: Search by filename/title first
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
        
        # Step 3: Semantic search with filtering
        logger.info("Performing semantic search with tag and metadata filtering.")
        semantic_results = self.search_engine.semantic_search(
            query, 
            top_k=5, 
            filter_tags=relevant_tags,
            search_metadata=search_metadata
        )
        
        # Use LLM to evaluate results
        scored_results = self._evaluate_search_results(query, semantic_results)
        
        return scored_results
        
        # If we get here, we didn't find a good match
        logger.info("No relevant documents found.")
        return {
            "found_by": "none",
            "query": query,
            "message": "No relevant documents found after exhaustive search."
        }
