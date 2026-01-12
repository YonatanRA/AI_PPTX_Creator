#!/usr/bin/env python3
"""
AI PPTX Creator - Improved Version

This script creates PowerPoint presentations from PDF documents using:
- LangChain for document processing and RAG
- OpenAI GPT models for content generation
- ChromaDB for vector storage
- python-pptx for presentation creation

Features:
- Configuration management
- Comprehensive error handling
- Security validation
- Modular design
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import traceback

from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai import OpenAI
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document


class Config:
    """Configuration class for AI PPTX Creator."""

    # API Configuration
    OPENAI_API_KEY: str = None

    # Model Configuration
    CHAT_MODEL: str = "gpt-4-turbo"
    CODE_GEN_MODEL: str = "gpt-3.5-turbo-instruct"  # More cost-effective for code generation
    CODE_GEN_TEMPERATURE: float = 0.0  # Deterministic output for code
    CODE_GEN_MAX_TOKENS: int = 2048  # Increased for longer code

    # Retriever Configuration
    RETRIEVER_K: int = 2  # Number of documents to retrieve
    RETRIEVER_LAMBDA_MULT: float = 0.25  # MMR diversity parameter

    # Directory Configuration
    BASE_DIR: Path = Path(__file__).parent
    PDF_DIR: Path = BASE_DIR / "pdfs"
    PPTX_DIR: Path = BASE_DIR / "pptx"
    CHROMA_DB_DIR: Path = BASE_DIR / "chroma_db"

    @classmethod
    def load_env(cls) -> None:
        """Load environment variables and validate configuration."""
        load_dotenv()
        cls.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please check your .env file."
            )

        # Ensure directories exist
        cls.PPTX_DIR.mkdir(exist_ok=True)

        print("✓ Configuration loaded successfully")
        print(f"✓ PDF directory: {cls.PDF_DIR}")
        print(f"✓ Output directory: {cls.PPTX_DIR}")


def test_llm_connection(test_query: str = "What is the Suez Canal?") -> str:
    """
    Test the LLM connection with a simple query.

    Args:
        test_query: The test question to ask

    Returns:
        The model's response

    Raises:
        Exception: If connection fails
    """
    try:
        model = ChatOpenAI(model=Config.CHAT_MODEL)
        response = model.invoke(test_query)
        print("✓ LLM connection successful")
        return response.content
    except Exception as e:
        print(f"✗ LLM connection failed: {e}")
        raise


def load_pdf_documents(pdf_dir: Path = None) -> List[Document]:
    """
    Load PDF documents from the specified directory.

    Args:
        pdf_dir: Path to the directory containing PDF files

    Returns:
        List of loaded documents

    Raises:
        FileNotFoundError: If PDF directory doesn't exist
        ValueError: If no PDF files are found
    """
    if pdf_dir is None:
        pdf_dir = Config.PDF_DIR

    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    # Check for PDF files
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_dir}")

    print(f"Found {len(pdf_files)} PDF file(s): {[f.name for f in pdf_files]}")

    # Load documents
    loader = PyPDFDirectoryLoader(str(pdf_dir))
    pages = loader.load()

    print(f"✓ Loaded {len(pages)} pages from PDF documents")
    return pages


def create_vector_store(documents: List[Document]) -> Chroma:
    """
    Create or load a ChromaDB vector store from documents.

    Args:
        documents: List of documents to embed

    Returns:
        Chroma vector store instance
    """
    try:
        # Initialize embedding model
        embeddings = OpenAIEmbeddings()

        # Create vector store
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(Config.CHROMA_DB_DIR)
        )

        print(f"✓ Vector store created with {len(documents)} documents")
        return vector_store

    except Exception as e:
        print(f"✗ Error creating vector store: {e}")
        raise


def create_retriever(vector_store: Chroma):
    """
    Create a retriever with Maximal Marginal Relevance (MMR).

    MMR balances relevance and diversity in retrieved documents.

    Args:
        vector_store: The vector store to create retriever from

    Returns:
        Configured retriever
    """
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": Config.RETRIEVER_K,
            "lambda_mult": Config.RETRIEVER_LAMBDA_MULT
        }
    )
    print("✓ Retriever configured with MMR search")
    return retriever


def create_content_generation_chain(retriever, model):
    """
    Create a RAG chain for generating structured bullet points.

    Args:
        retriever: Document retriever
        model: Language model to use

    Returns:
        Configured chain
    """
    template = """
    You are an expert at summarizing documents into clear, structured presentations.

    Given the context below, generate:
    1. A clear, descriptive header
    2. Exactly 10 numbered bullet points
    3. Each bullet point should be 30-40 words
    4. Focus on the most important information

    Format:
    **Header: [Your Header Here]**

    1. **[Topic]**: [Description]
    2. **[Topic]**: [Description]
    ...

    Context: {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    parser = StrOutputParser()

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | parser
    )

    return chain


def create_code_generation_chain(presentation_title: str, output_filename: str):
    """
    Create a chain for generating python-pptx code.

    Args:
        presentation_title: Title for the presentation
        output_filename: Name of the output PPTX file

    Returns:
        Configured chain
    """
    template = """
    You are an expert Python developer specializing in the python-pptx library.

    Task: Generate clean, executable Python code to create a PowerPoint presentation.

    Requirements:
    1. Import required modules: `from pptx import Presentation` and `from pptx.util import Pt`
    2. Create presentation with:
       - Slide 1 (layout 0): Title: "{title}", Subtitle: "Generated by AI"
       - Slide 2 (layout 1): Title: "Key Insights (Part 1)", Content: First 5 bullet points
       - Slide 3 (layout 1): Title: "Key Insights (Part 2)", Content: Last 5 bullet points
    3. Set body text font size to 18pt for readability
    4. Save to: "{output_path}"
    5. Output ONLY executable Python code, NO markdown formatting
    6. Add proper error handling for file operations
    7. Properly format bullet points by adding them to text_frame with proper paragraph handling

    Content to include:
    {context}

    Output format: Plain Python code only, no ```python``` markers.
    """

    output_path = Config.PPTX_DIR / output_filename

    prompt = ChatPromptTemplate.from_template(template)
    prompt = prompt.partial(
        title=presentation_title,
        output_path=str(output_path)
    )

    model = OpenAI(
        temperature=Config.CODE_GEN_TEMPERATURE,
        max_tokens=Config.CODE_GEN_MAX_TOKENS
    )
    parser = StrOutputParser()

    chain = prompt | model | parser
    return chain


def clean_python_code(code_str: str) -> str:
    """
    Remove markdown code block syntax from generated code.

    Args:
        code_str: Raw code string from LLM

    Returns:
        Cleaned Python code
    """
    # Remove markdown code blocks
    if '```python' in code_str:
        code_str = code_str.split('```python')[1]
    if '```' in code_str:
        code_str = code_str.split('```')[0]

    return code_str.strip()


def validate_code_safety(code: str) -> Tuple[bool, str]:
    """
    Perform basic safety checks on generated code.

    Args:
        code: The code to validate

    Returns:
        Tuple of (is_safe, message)
    """
    dangerous_patterns = [
        "os.system",
        "subprocess",
        "eval(",
        "__import__",
    ]

    # Check for dangerous patterns
    for pattern in dangerous_patterns:
        if pattern in code:
            return False, f"Potentially unsafe code detected: {pattern}"

    # Verify required imports are present
    required_imports = ["from pptx import Presentation"]
    for req in required_imports:
        if req not in code:
            return False, f"Missing required import: {req}"

    return True, "Code validation passed"


def execute_generated_code(code: str, verbose: bool = False) -> bool:
    """
    Safely execute the generated Python code.

    Args:
        code: The Python code to execute
        verbose: Whether to print execution details

    Returns:
        True if execution succeeded, False otherwise
    """
    # Clean the code
    cleaned_code = clean_python_code(code)

    if verbose:
        print("\nCleaned code:")
        print("=" * 80)
        print(cleaned_code)
        print("=" * 80)

    # Validate code safety
    is_safe, message = validate_code_safety(cleaned_code)
    if not is_safe:
        print(f"✗ {message}")
        print("Code execution blocked for safety reasons.")
        return False

    print(f"✓ {message}")

    # Execute the code
    try:
        exec(cleaned_code)
        print("✓ Presentation created successfully!")
        return True
    except Exception as e:
        print(f"✗ Error executing code: {e}")
        if verbose:
            traceback.print_exc()
        return False


def create_presentation(
    query: str = "What are the key points and implications of the briefing?",
    presentation_title: str = "EPRS Briefing Analysis",
    output_filename: str = "Red_Sea_Security_Threats.pptx",
    test_connection: bool = True,
    verbose: bool = False
) -> bool:
    """
    Main function to create a PowerPoint presentation from PDF documents.

    Args:
        query: Question to ask about the PDF content
        presentation_title: Title for the presentation
        output_filename: Name of the output PPTX file
        test_connection: Whether to test LLM connection first
        verbose: Whether to print detailed execution info

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load configuration
        print("\n" + "=" * 80)
        print("AI PPTX Creator - Improved Version")
        print("=" * 80 + "\n")

        Config.load_env()

        # Test connection if requested
        if test_connection:
            print("\n[1/7] Testing LLM connection...")
            test_llm_connection()

        # Load PDF documents
        print("\n[2/7] Loading PDF documents...")
        pages = load_pdf_documents()

        # Create vector store
        print("\n[3/7] Creating vector store...")
        chroma_db = create_vector_store(pages)
        retriever = create_retriever(chroma_db)

        # Generate content
        print("\n[4/7] Generating content from PDF...")
        chat_model = ChatOpenAI(model=Config.CHAT_MODEL)
        content_chain = create_content_generation_chain(retriever, chat_model)
        response = content_chain.invoke(query)

        if verbose:
            print("\nGenerated content:")
            print("-" * 80)
            print(response)
            print("-" * 80)

        # Generate Python code
        print("\n[5/7] Generating Python code for PPTX creation...")
        code_chain = create_code_generation_chain(presentation_title, output_filename)
        generated_code = code_chain.invoke({"context": response})

        # Execute code
        print("\n[6/7] Executing generated code...")
        success = execute_generated_code(generated_code, verbose=verbose)

        # Final status
        print("\n[7/7] Finalizing...")
        if success:
            output_file = Config.PPTX_DIR / output_filename
            print("\n" + "=" * 80)
            print(f"📊 SUCCESS! Presentation saved to: {output_file}")
            print("=" * 80 + "\n")
            return True
        else:
            print("\n" + "=" * 80)
            print("✗ FAILED to create presentation")
            print("=" * 80 + "\n")
            return False

    except Exception as e:
        print(f"\n✗ Error in main execution: {e}")
        if verbose:
            traceback.print_exc()
        return False


def main():
    """Main entry point for the script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create PowerPoint presentations from PDF documents using AI"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What are the key points and implications of the briefing?",
        help="Question to ask about the PDF content"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="EPRS Briefing Analysis",
        help="Title for the presentation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="Red_Sea_Security_Threats.pptx",
        help="Output filename for the presentation"
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Skip LLM connection test"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed execution information"
    )

    args = parser.parse_args()

    success = create_presentation(
        query=args.query,
        presentation_title=args.title,
        output_filename=args.output,
        test_connection=not args.no_test,
        verbose=args.verbose
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
