#!/usr/bin/env python3
"""
AI PPTX Creator - Automated presentation generation from PDF documents using LangChain and OpenAI.

This script loads PDF documents, creates embeddings, and uses LLMs to generate PowerPoint presentations.
"""

import os
import argparse
import logging
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PresentationConfig:
    """Configuration for presentation generation."""
    
    def __init__(
        self,
        pdf_directory: str = "pdfs",
        output_directory: str = "pptx",
        chroma_directory: str = "chroma_db",
        model_name: str = "gpt-4-turbo",
        temperature: float = 0,
        max_tokens: int = 1024,
        num_bullet_points: int = 10,
        bullet_point_words: int = 40,
        retriever_k: int = 2,
        retriever_lambda: float = 0.25
    ):
        """Initialize presentation configuration.
        
        Args:
            pdf_directory: Directory containing PDF files
            output_directory: Directory for output presentations
            chroma_directory: Directory for ChromaDB storage
            model_name: OpenAI model name
            temperature: Model temperature for code generation
            max_tokens: Maximum tokens for code generation
            num_bullet_points: Number of bullet points to generate
            bullet_point_words: Maximum words per bullet point
            retriever_k: Number of documents to retrieve
            retriever_lambda: Lambda multiplier for MMR algorithm
        """
        self.pdf_directory = Path(pdf_directory)
        self.output_directory = Path(output_directory)
        self.chroma_directory = Path(chroma_directory)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_bullet_points = num_bullet_points
        self.bullet_point_words = bullet_point_words
        self.retriever_k = retriever_k
        self.retriever_lambda = retriever_lambda
        
        # Ensure directories exist
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.chroma_directory.mkdir(parents=True, exist_ok=True)


class AIPresetationCreator:
    """Main class for creating AI-powered presentations."""
    
    def __init__(self, config: PresentationConfig):
        """Initialize the presentation creator.
        
        Args:
            config: Configuration object for presentation generation
        """
        self.config = config
        self.api_key = self._load_api_key()
        self.chat_model = ChatOpenAI(model=config.model_name)
        self.code_model = OpenAI(temperature=config.temperature, max_tokens=config.max_tokens)
        self.embeddings = OpenAIEmbeddings()
        self.parser = StrOutputParser()
        self.vector_db = None
        self.retriever = None
        
    def _load_api_key(self) -> str:
        """Load OpenAI API key from environment."""
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Please create a .env file with your API key."
            )
        return api_key
    
    def load_documents(self) -> List:
        """Load PDF documents from the configured directory.
        
        Returns:
            List of loaded document pages
        """
        logger.info(f"Loading PDF documents from {self.config.pdf_directory}")
        
        if not self.config.pdf_directory.exists():
            raise FileNotFoundError(
                f"PDF directory not found: {self.config.pdf_directory}"
            )
        
        loader = PyPDFDirectoryLoader(str(self.config.pdf_directory))
        pages = loader.load()
        
        logger.info(f"Loaded {len(pages)} pages from PDF documents")
        return pages
    
    def create_vector_database(self, documents: List, force_recreate: bool = False):
        """Create or load vector database from documents.
        
        Args:
            documents: List of documents to embed
            force_recreate: If True, recreate the database even if it exists
        """
        if force_recreate or not self.config.chroma_directory.exists():
            logger.info("Creating new vector database")
            self.vector_db = Chroma.from_documents(
                documents,
                self.embeddings,
                persist_directory=str(self.config.chroma_directory)
            )
        else:
            logger.info("Loading existing vector database")
            self.vector_db = Chroma(
                persist_directory=str(self.config.chroma_directory),
                embedding_function=self.embeddings
            )
        
        # Create retriever with MMR (Maximal Marginal Relevance) for diverse results
        self.retriever = self.vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.config.retriever_k,
                "lambda_mult": self.config.retriever_lambda
            }
        )
        logger.info("Vector database and retriever initialized")
    
    def generate_bullet_points(self, query: str) -> str:
        """Generate bullet points from a query using RAG.
        
        Args:
            query: Query to generate bullet points for
            
        Returns:
            Generated bullet points as a string
        """
        logger.info(f"Generating bullet points for query: {query}")
        
        template = f"""
            Given the context below and the question, 
            please generate a header and {self.config.num_bullet_points} bullet points.
            List with numbers the bullet points.
            Summarize each bullet point in {self.config.bullet_point_words} words.
            
            Put a line separator after `:` symbol.

            Context: {{context}}

            Question: {{question}}
            """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create RAG chain
        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.chat_model
            | self.parser
        )
        
        response = chain.invoke(query)
        logger.info("Bullet points generated successfully")
        return response
    
    def generate_presentation_code(self, bullet_points: str, output_filename: str) -> str:
        """Generate Python code to create a PowerPoint presentation.
        
        Args:
            bullet_points: Bullet points content to include in presentation
            output_filename: Name of the output PowerPoint file
            
        Returns:
            Generated Python code as a string
        """
        logger.info("Generating PowerPoint creation code")
        
        output_path = self.config.output_directory / output_filename
        
        template = f"""
            We have provided information below.
            Given this information, please generate a python code with python-pptx for three 
            slide presentation with this information. 
            
            Put the title in the first slide, 
            5 bullet points in the second slide and another 5 bullet in the third slide.
            Put list number in each bullet point.
                        
            Separate the bullet points into separate texts with line separator.
            Set font size to 20 for each bullet point. 
            Save the file in {output_path} path

            Information: {{context}}
            """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.code_model | self.parser
        
        code = chain.invoke({"context": bullet_points})
        logger.info("PowerPoint creation code generated")
        return code
    
    def execute_presentation_code(self, code: str, dry_run: bool = False):
        """Execute the generated PowerPoint creation code.
        
        Args:
            code: Python code to execute
            dry_run: If True, only print the code without executing
        """
        if dry_run:
            logger.info("Dry run mode - printing generated code:")
            print("\n" + "="*80)
            print(code)
            print("="*80 + "\n")
            return
        
        try:
            logger.info("Executing PowerPoint creation code")
            exec(code)
            logger.info("Presentation created successfully")
        except Exception as e:
            logger.error(f"Error executing presentation code: {e}")
            logger.info("Generated code:")
            print(code)
            raise
    
    def create_presentation(
        self,
        query: str,
        output_filename: str,
        force_recreate_db: bool = False,
        dry_run: bool = False
    ):
        """Complete workflow to create a presentation.
        
        Args:
            query: Query to generate presentation content for
            output_filename: Name of the output PowerPoint file
            force_recreate_db: If True, recreate the vector database
            dry_run: If True, only generate code without executing
        """
        # Load documents and create vector database
        documents = self.load_documents()
        self.create_vector_database(documents, force_recreate=force_recreate_db)
        
        # Generate bullet points
        bullet_points = self.generate_bullet_points(query)
        
        # Generate presentation code
        code = self.generate_presentation_code(bullet_points, output_filename)
        
        # Execute code to create presentation
        self.execute_presentation_code(code, dry_run=dry_run)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate PowerPoint presentations from PDF documents using AI"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What are the endnotes of the briefing?",
        help="Query to generate presentation content for"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="Red Sea Security Threats.pptx",
        help="Name of the output PowerPoint file"
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default="pdfs",
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="pptx",
        help="Directory for output presentations"
    )
    parser.add_argument(
        "--force-recreate-db",
        action="store_true",
        help="Force recreation of the vector database"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate code without executing (for testing)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create configuration
    config = PresentationConfig(
        pdf_directory=args.pdf_dir,
        output_directory=args.output_dir
    )
    
    # Create presentation
    creator = AIPresetationCreator(config)
    creator.create_presentation(
        query=args.query,
        output_filename=args.output,
        force_recreate_db=args.force_recreate_db,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
