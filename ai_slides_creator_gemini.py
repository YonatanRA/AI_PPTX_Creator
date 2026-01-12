#!/usr/bin/env python3
"""
AI Google Slides Creator - Gemini + Nanobanana (Imagen 3) Version

This script creates Google Slides presentations from PDF documents using:
- Gemini API for content generation and RAG
- Imagen 3 (Nanobanana) for image generation
- Google Slides API for presentation creation
- ChromaDB for vector storage

Features:
- Full Google ecosystem integration
- Configuration management
- Comprehensive error handling
- Security validation
- Modular design
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import traceback
import logging
from io import BytesIO
import json

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from google import genai
from google.genai import types
from PIL import Image

# Google Slides API imports
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration class for AI Slides Creator."""

    # API Configuration
    GOOGLE_API_KEY: str = None
    GOOGLE_SERVICE_ACCOUNT_FILE: str = None

    # Model Configuration
    GEMINI_MODEL: str = "gemini-2.0-flash-exp"  # Gemini for content generation
    IMAGEN_MODEL: str = "imagen-3.0-generate-002"  # Nanobanana for images
    TEMPERATURE: float = 0.7  # Balance creativity and consistency
    MAX_OUTPUT_TOKENS: int = 2048

    # Retriever Configuration
    RETRIEVER_K: int = 3  # Number of documents to retrieve
    RETRIEVER_LAMBDA_MULT: float = 0.25  # MMR diversity parameter

    # Image Configuration
    IMAGE_ASPECT_RATIO: str = "16:9"  # Standard presentation ratio
    IMAGE_SAFETY_FILTER: str = "BLOCK_ONLY_HIGH"

    # Directory Configuration
    BASE_DIR: Path = Path(__file__).parent
    PDF_DIR: Path = BASE_DIR / "pdfs"
    IMG_DIR: Path = BASE_DIR / "imgs" / "generated_nanobanana"
    CHROMA_DB_DIR: Path = BASE_DIR / "chroma_db"

    # Presentation Configuration
    NUM_CONTENT_SLIDES: int = 3
    BULLETS_PER_SLIDE: int = 3

    @classmethod
    def load_env(cls) -> None:
        """Load environment variables and validate configuration."""
        load_dotenv()
        cls.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        cls.GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")

        if not cls.GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY not found in environment variables. "
                "Please check your .env file."
            )

        # Service account is optional - if not provided, will use manual auth
        if cls.GOOGLE_SERVICE_ACCOUNT_FILE and not Path(cls.GOOGLE_SERVICE_ACCOUNT_FILE).exists():
            logger.warning(
                f"Service account file not found: {cls.GOOGLE_SERVICE_ACCOUNT_FILE}. "
                "Will use alternative authentication method."
            )
            cls.GOOGLE_SERVICE_ACCOUNT_FILE = None

        # Ensure directories exist
        cls.IMG_DIR.mkdir(parents=True, exist_ok=True)

        logger.info("✓ Configuration loaded successfully")
        logger.info(f"✓ PDF directory: {cls.PDF_DIR}")
        logger.info(f"✓ Image directory: {cls.IMG_DIR}")


class GeminiRAGChain:
    """Handles RAG operations using Gemini."""

    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=Config.GOOGLE_API_KEY
        )
        self.chat_model = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            temperature=Config.TEMPERATURE,
            max_output_tokens=Config.MAX_OUTPUT_TOKENS,
            google_api_key=Config.GOOGLE_API_KEY
        )
        self.parser = StrOutputParser()
        self.vector_store = None
        self.retriever = None

    def load_pdf_documents(self, pdf_dir: Path = None) -> List[Document]:
        """
        Load PDF documents from the specified directory.

        Args:
            pdf_dir: Path to the directory containing PDF files

        Returns:
            List of loaded documents
        """
        if pdf_dir is None:
            pdf_dir = Config.PDF_DIR

        if not pdf_dir.exists():
            raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in {pdf_dir}")

        logger.info(f"Found {len(pdf_files)} PDF file(s): {[f.name for f in pdf_files]}")

        loader = PyPDFDirectoryLoader(str(pdf_dir))
        pages = loader.load()

        logger.info(f"✓ Loaded {len(pages)} pages from PDF documents")
        return pages

    def create_vector_store(self, documents: List[Document]) -> None:
        """
        Create or load a ChromaDB vector store from documents.

        Args:
            documents: List of documents to embed
        """
        try:
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(Config.CHROMA_DB_DIR)
            )

            self.retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": Config.RETRIEVER_K,
                    "lambda_mult": Config.RETRIEVER_LAMBDA_MULT
                }
            )

            logger.info(f"✓ Vector store created with {len(documents)} documents")
            logger.info("✓ Retriever configured with MMR search")

        except Exception as e:
            logger.error(f"✗ Error creating vector store: {e}")
            raise

    def generate_presentation_plan(self, query: str) -> Dict[str, Any]:
        """
        Generate a structured plan for the presentation using Gemini.

        Args:
            query: User's query about the content

        Returns:
            Dictionary with presentation structure
        """
        logger.info("Generating presentation plan with Gemini...")

        template = """
        You are an expert at creating engaging presentations from documents.

        Given the context below, create a structured presentation plan.

        Requirements:
        1. Generate a compelling presentation title
        2. Create exactly {num_slides} content slides
        3. Each slide should have a clear title
        4. Each slide should have exactly {bullets_per_slide} bullet points
        5. Each bullet point should be 25-35 words
        6. Focus on the most important and interesting information

        Output format (strict JSON):
        {{
            "title": "Presentation Title",
            "subtitle": "Brief subtitle or tagline",
            "slides": [
                {{
                    "title": "Slide Title",
                    "bullets": ["Bullet 1", "Bullet 2", "Bullet 3"],
                    "image_prompt": "Description for AI image generation"
                }}
            ]
        }}

        Context: {context}

        User Request: {question}

        Generate the JSON plan:
        """

        prompt = ChatPromptTemplate.from_template(template)
        prompt = prompt.partial(
            num_slides=Config.NUM_CONTENT_SLIDES,
            bullets_per_slide=Config.BULLETS_PER_SLIDE
        )

        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.chat_model
            | self.parser
        )

        response = chain.invoke(query)
        logger.info("✓ Plan generated with Gemini")

        # Parse JSON response
        return self._parse_plan_json(response)

    def _parse_plan_json(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the JSON response from Gemini.

        Args:
            response_text: Raw response text

        Returns:
            Parsed dictionary
        """
        try:
            # Try to extract JSON from markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            plan = json.loads(response_text.strip())

            # Validate structure
            if "title" not in plan:
                plan["title"] = "AI Generated Presentation"
            if "subtitle" not in plan:
                plan["subtitle"] = "Generated with Gemini & Nanobanana"
            if "slides" not in plan or not isinstance(plan["slides"], list):
                raise ValueError("Invalid plan structure: missing or invalid 'slides'")

            logger.info(f"✓ Parsed plan with {len(plan['slides'])} slides")
            return plan

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Response text: {response_text[:500]}")
            # Return a default structure
            return {
                "title": "AI Generated Presentation",
                "subtitle": "Generated with Gemini & Nanobanana",
                "slides": []
            }


class NanobananaImageGenerator:
    """Handles image generation using Imagen 3 (Nanobanana)."""

    def __init__(self):
        self.client = genai.Client(api_key=Config.GOOGLE_API_KEY)

    def generate_image(self, prompt: str, output_path: Path) -> bool:
        """
        Generate an image using Imagen 3 (Nanobanana).

        Args:
            prompt: Text description for the image
            output_path: Where to save the generated image

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Generating image: {prompt[:60]}...")

        try:
            # Enhance prompt for better presentation images
            enhanced_prompt = (
                f"Professional presentation illustration, modern, clean, high quality, "
                f"suitable for business presentation: {prompt}"
            )

            response = self.client.models.generate_images(
                model=Config.IMAGEN_MODEL,
                prompt=enhanced_prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio=Config.IMAGE_ASPECT_RATIO,
                    safety_filter_level=Config.IMAGE_SAFETY_FILTER,
                )
            )

            if response.generated_images:
                image_bytes = response.generated_images[0].image.image_bytes
                image = Image.open(BytesIO(image_bytes))
                image.save(output_path)
                logger.info(f"✓ Image saved to {output_path}")
                return True
            else:
                logger.warning("No images generated")
                return False

        except Exception as e:
            logger.error(f"✗ Failed to generate image: {e}")
            return False


class GoogleSlidesCreator:
    """Handles creation of Google Slides presentations."""

    def __init__(self):
        self.service = None
        self.presentation_id = None
        self._authenticate()

    def _authenticate(self):
        """Authenticate with Google Slides API."""
        try:
            if Config.GOOGLE_SERVICE_ACCOUNT_FILE:
                # Use service account
                credentials = service_account.Credentials.from_service_account_file(
                    Config.GOOGLE_SERVICE_ACCOUNT_FILE,
                    scopes=['https://www.googleapis.com/auth/presentations',
                           'https://www.googleapis.com/auth/drive']
                )
                logger.info("✓ Authenticated with service account")
            else:
                # For now, raise an error - in production, implement OAuth flow
                raise ValueError(
                    "Service account file required. Please set GOOGLE_SERVICE_ACCOUNT_FILE "
                    "in your .env file. Get it from Google Cloud Console."
                )

            self.service = build('slides', 'v1', credentials=credentials)
            logger.info("✓ Google Slides API service created")

        except Exception as e:
            logger.error(f"✗ Authentication failed: {e}")
            raise

    def create_presentation(self, title: str) -> str:
        """
        Create a new Google Slides presentation.

        Args:
            title: Presentation title

        Returns:
            Presentation ID
        """
        try:
            presentation = self.service.presentations().create(
                body={'title': title}
            ).execute()

            self.presentation_id = presentation.get('presentationId')
            logger.info(f"✓ Created presentation: {self.presentation_id}")
            logger.info(f"✓ View at: https://docs.google.com/presentation/d/{self.presentation_id}")

            return self.presentation_id

        except HttpError as e:
            logger.error(f"✗ Failed to create presentation: {e}")
            raise

    def add_title_slide(self, title: str, subtitle: str):
        """
        Add a title slide to the presentation.

        Args:
            title: Main title
            subtitle: Subtitle text
        """
        requests = [
            {
                'createSlide': {
                    'slideLayoutReference': {
                        'predefinedLayout': 'TITLE'
                    }
                }
            }
        ]

        response = self.service.presentations().batchUpdate(
            presentationId=self.presentation_id,
            body={'requests': requests}
        ).execute()

        slide_id = response.get('replies')[0].get('createSlide').get('objectId')

        # Update text
        requests = [
            {
                'insertText': {
                    'objectId': slide_id,
                    'text': title,
                    'insertionIndex': 0
                }
            }
        ]

        self.service.presentations().batchUpdate(
            presentationId=self.presentation_id,
            body={'requests': requests}
        ).execute()

        logger.info("✓ Title slide added")

    def add_content_slide(self, title: str, bullets: List[str], image_path: Optional[Path] = None):
        """
        Add a content slide with title, bullets, and optional image.

        Args:
            title: Slide title
            bullets: List of bullet points
            image_path: Optional path to image file
        """
        requests = [
            {
                'createSlide': {
                    'slideLayoutReference': {
                        'predefinedLayout': 'TITLE_AND_BODY'
                    }
                }
            }
        ]

        response = self.service.presentations().batchUpdate(
            presentationId=self.presentation_id,
            body={'requests': requests}
        ).execute()

        slide_id = response.get('replies')[0].get('createSlide').get('objectId')

        # Get the presentation to find text boxes
        presentation = self.service.presentations().get(
            presentationId=self.presentation_id
        ).execute()

        # Find the created slide
        slide = None
        for s in presentation.get('slides'):
            if s.get('objectId') == slide_id:
                slide = s
                break

        if not slide:
            logger.warning(f"Could not find slide {slide_id}")
            return

        # Update title and body
        requests = []
        for element in slide.get('pageElements', []):
            shape = element.get('shape')
            if shape and shape.get('shapeType') == 'TEXT_BOX':
                placeholder = shape.get('placeholder')
                if placeholder:
                    placeholder_type = placeholder.get('type')
                    object_id = element.get('objectId')

                    if placeholder_type == 'TITLE':
                        requests.append({
                            'insertText': {
                                'objectId': object_id,
                                'text': title,
                                'insertionIndex': 0
                            }
                        })
                    elif placeholder_type == 'BODY':
                        bullet_text = '\n'.join([f"• {bullet}" for bullet in bullets])
                        requests.append({
                            'insertText': {
                                'objectId': object_id,
                                'text': bullet_text,
                                'insertionIndex': 0
                            }
                        })

        if requests:
            self.service.presentations().batchUpdate(
                presentationId=self.presentation_id,
                body={'requests': requests}
            ).execute()

        logger.info(f"✓ Content slide added: {title}")


class AISlidesCreator:
    """Main orchestrator for AI-powered Google Slides creation."""

    def __init__(self):
        self.rag_chain = GeminiRAGChain()
        self.image_generator = NanobananaImageGenerator()
        self.slides_creator = None  # Initialize only if credentials available

    def create_presentation(
        self,
        query: str,
        use_google_slides: bool = True,
        verbose: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """
        Create a presentation from PDF documents.

        Args:
            query: Question to ask about the PDF content
            use_google_slides: If True, create Google Slides (requires auth).
                              If False, returns the plan only.
            verbose: Whether to print detailed info

        Returns:
            Tuple of (success, presentation_url_or_plan)
        """
        try:
            logger.info("\n" + "=" * 80)
            logger.info("AI Slides Creator - Gemini + Nanobanana Version")
            logger.info("=" * 80 + "\n")

            # Step 1: Load PDFs
            logger.info("[1/5] Loading PDF documents...")
            pages = self.rag_chain.load_pdf_documents()

            # Step 2: Create vector store
            logger.info("\n[2/5] Creating vector store...")
            self.rag_chain.create_vector_store(pages)

            # Step 3: Generate presentation plan
            logger.info("\n[3/5] Generating presentation plan with Gemini...")
            plan = self.rag_chain.generate_presentation_plan(query)

            if verbose:
                logger.info("\n" + "=" * 80)
                logger.info("PRESENTATION PLAN:")
                logger.info(json.dumps(plan, indent=2))
                logger.info("=" * 80 + "\n")

            # Step 4: Generate images
            logger.info("\n[4/5] Generating images with Nanobanana (Imagen 3)...")
            for i, slide in enumerate(plan.get('slides', [])):
                image_prompt = slide.get('image_prompt', slide.get('title', ''))
                image_path = Config.IMG_DIR / f"slide_{i+1}.png"
                self.image_generator.generate_image(image_prompt, image_path)
                slide['image_path'] = image_path

            # Step 5: Create Google Slides
            if use_google_slides:
                logger.info("\n[5/5] Creating Google Slides presentation...")
                self.slides_creator = GoogleSlidesCreator()

                presentation_id = self.slides_creator.create_presentation(
                    plan.get('title', 'AI Generated Presentation')
                )

                # Add title slide
                self.slides_creator.add_title_slide(
                    plan.get('title', 'AI Generated Presentation'),
                    plan.get('subtitle', 'Generated with Gemini & Nanobanana')
                )

                # Add content slides
                for slide_data in plan.get('slides', []):
                    self.slides_creator.add_content_slide(
                        title=slide_data.get('title', 'Slide'),
                        bullets=slide_data.get('bullets', []),
                        image_path=slide_data.get('image_path')
                    )

                presentation_url = f"https://docs.google.com/presentation/d/{presentation_id}"

                logger.info("\n" + "=" * 80)
                logger.info(f"📊 SUCCESS! Presentation created!")
                logger.info(f"🔗 View at: {presentation_url}")
                logger.info("=" * 80 + "\n")

                return True, presentation_url
            else:
                logger.info("\n[5/5] Skipping Google Slides creation (use_google_slides=False)")
                logger.info("✓ Plan generated successfully")
                return True, json.dumps(plan, indent=2)

        except Exception as e:
            logger.error(f"\n✗ Error in presentation creation: {e}")
            if verbose:
                traceback.print_exc()
            return False, None


def main():
    """Main entry point for the script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create Google Slides presentations from PDF documents using Gemini & Nanobanana"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What are the key points and implications of this document?",
        help="Question to ask about the PDF content"
    )
    parser.add_argument(
        "--no-slides",
        action="store_true",
        help="Generate plan only, don't create Google Slides (useful for testing)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed execution information"
    )

    args = parser.parse_args()

    try:
        # Load configuration
        Config.load_env()

        # Create presentation
        creator = AISlidesCreator()
        success, result = creator.create_presentation(
            query=args.query,
            use_google_slides=not args.no_slides,
            verbose=args.verbose
        )

        if success:
            if args.no_slides:
                print("\nGenerated Plan:")
                print(result)
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
