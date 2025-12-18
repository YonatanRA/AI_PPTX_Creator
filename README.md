# AI PPTX Creator

In the corporate world slides are everywhere, it is often used as a way to communicate idea and achievements. Making slides is something most people do every week and could be very time consuming.

Python has a library called Python-pptx which allows users to programmatically create PowerPoint presentations. In this project we will the [Recent threats in the Red Sea](https://www.europarl.europa.eu/RegData/etudes/BRIE/2024/760390/EPRS_BRI(2024)760390_EN.pdf) pdf file, published by the European Parliament in 2024, as information source to create a pptx file with the assistant of LLM.

The pipeline is as follows:

![rag](imgs/rag.webp)

1. We make a query to retrieve information from a vector database.
2. We feed a first LLM to create a bullet point list.
3. With that list, we ask a second LLM to create Python-pptx code.
4. We execute that code to create a pptx file.


This project is a proof of concept. It needs a better visual and design development.


### Technologies used

+ [LangChain](https://www.langchain.com/)
+ [OpenAI GPT4](https://openai.com/)
+ [Chroma](https://www.trychroma.com/)
+ [Python-pptx](https://python-pptx.readthedocs.io/en/latest/)


### How to use this repo

#### Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Keys:**
   Create a `.env` file in the project root with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   GOOGLE_API_KEY=your_google_gemini_api_key_here
   ```
   - OpenAI API key: Get it from https://platform.openai.com/api-keys
   - Google Gemini API key: Get it from https://aistudio.google.com/app/apikey
   
   Note: The Google API key is only required if you want to use the image generation feature (`create_presentation_with_images.py`)

3. **Add PDF documents:**
   Place your PDF files in the `pdfs/` directory.

#### Running the Application

**Option 1: Python Script (Recommended)**

The improved Python script provides a modular, production-ready implementation with better error handling and configuration options.

Basic usage:
```bash
python create_presentation.py
```

With custom parameters:
```bash
python create_presentation.py \
  --query "What are the main points of the document?" \
  --output "My Presentation.pptx" \
  --pdf-dir pdfs \
  --output-dir pptx
```

Available options:
- `--query`: Query to generate presentation content (default: "What are the endnotes of the briefing?")
- `--output`: Output filename (default: "Red Sea Security Threats.pptx")
- `--pdf-dir`: Directory containing PDFs (default: "pdfs")
- `--output-dir`: Output directory for presentations (default: "pptx")
- `--force-recreate-db`: Recreate the vector database from scratch
- `--dry-run`: Generate code without executing (useful for testing)
- `--verbose`: Enable detailed logging

**Option 2: Python Script with AI Image Generation (NEW! 🎨)**

The new image generation script uses Google's Gemini API (Imagen model, codenamed "Nano Banana") to create presentations with AI-generated images for each slide.

Basic usage:
```bash
python create_presentation_with_images.py
```

With custom parameters:
```bash
python create_presentation_with_images.py \
  --query "What are the main security threats discussed?" \
  --output "Security_Analysis.pptx" \
  --num-slides 5
```

Available options:
- `--query`: Query to generate presentation content (default: "What are the main topics...")
- `--output`: Output filename (default: "AI_Presentation_with_Images.pptx")
- `--pdf-dir`: Directory containing PDFs (default: "pdfs")
- `--output-dir`: Output directory for presentations (default: "pptx")
- `--num-slides`: Number of content slides to generate (default: 5)
- `--no-images`: Disable AI image generation (text-only slides)
- `--force-recreate-db`: Recreate the vector database from scratch
- `--verbose`: Enable detailed logging

**Option 3: Jupyter Notebook**

For interactive exploration and step-by-step execution, use the Jupyter notebook in the `notebooks/` folder:
```bash
jupyter notebook notebooks/AI_PPTX_Creator.ipynb
```

### Image Generation Features (v3.0 - NEW! 🎨)

The latest version introduces AI-powered image generation using **Google Gemini's Imagen model** (internally codenamed "Nano Banana").

#### What is Imagen / Nano Banana?
<cite index="3-9">Imagen 4, internally codenamed Nano Banana, is Google's image generation model optimized for speed and stylistic diversity</cite>. It powers the image generation capabilities in Google Workspace products like Slides and Vids.

#### Key Features:

**1. Context-Aware Image Generation**
- Generates unique images for each slide based on the slide title and content
- Uses bullet points as context to create relevant, professional illustrations
- Modern, clean business style optimized for presentations

**2. Seamless Integration**
- Automatically creates images during presentation generation
- Images are positioned on the right side of slides with text on the left
- 16:9 aspect ratio optimized for widescreen presentations

**3. Flexible Configuration**
- Toggle image generation on/off with `--no-images` flag
- Configurable image aspect ratios and safety filters
- Images saved to `imgs/generated/` directory for reuse

**4. Direct PPTX Creation**
- Creates presentation files directly using `python-pptx` library
- No need for intermediate code generation or execution
- Full control over slide layouts, fonts, and positioning

#### How It Works:

1. **Content Generation**: OpenAI GPT-4 analyzes PDF documents via RAG and generates structured slide content
2. **Image Generation**: For each slide, Google Gemini creates a contextual image based on the title and key points
3. **Presentation Assembly**: The script directly creates a PowerPoint file with professional layouts combining text and images

#### Comparison with Original Version:

| Feature | Original (v2.0) | With Images (v3.0) |
|---------|----------------|--------------------|
| Image Support | ❌ None | ✅ AI-generated |
| PPTX Creation | Via LLM code generation | Direct python-pptx |
| Slide Layout | Basic bullet lists | Text + Images |
| Customization | Limited | Highly configurable |
| API Requirements | OpenAI only | OpenAI + Google Gemini |

### Code Improvements (v2.0)

The new `create_presentation.py` script includes several improvements over the original notebook implementation:

#### 1. **Modular Architecture**
- **`PresentationConfig` class**: Centralized configuration management for all parameters
- **`AIPresetationCreator` class**: Encapsulated logic with clear method separation
- **Separation of concerns**: Each method has a single, well-defined responsibility

#### 2. **Error Handling & Validation**
- Validates API key existence before execution
- Checks for PDF directory existence
- Comprehensive error handling with informative messages
- Try-catch blocks around critical operations

#### 3. **Logging System**
- Built-in logging with configurable verbosity levels
- Progress tracking for long-running operations
- Detailed error messages for debugging

#### 4. **Command-Line Interface**
- Fully configurable via command-line arguments
- Sensible defaults for quick execution
- Help documentation (`python create_presentation.py --help`)

#### 5. **Performance Optimizations**
- Vector database caching: Reuses existing embeddings instead of recreating
- Force recreation option for when source documents change
- Efficient document loading and processing

#### 6. **Development Features**
- **Dry-run mode**: Preview generated code without execution
- **Verbose logging**: Detailed debugging information
- **Path handling**: Uses `pathlib.Path` for cross-platform compatibility

#### 7. **Code Quality**
- Type hints for better IDE support and documentation
- Comprehensive docstrings following Google style
- PEP 8 compliant formatting
- Reusable and testable components

#### 8. **Configuration Flexibility**
- Adjustable number of bullet points and words per point
- Configurable retriever parameters (k, lambda_mult)
- Customizable model parameters (temperature, max_tokens)
- Easy to extend with new configuration options

### Example Workflows

**Version 2.0 (Text-only presentations):**

```bash
# Quick start with defaults
python create_presentation.py

# Test code generation without creating presentation
python create_presentation.py --dry-run

# Update vector database and create presentation
python create_presentation.py --force-recreate-db

# Custom query with verbose output
python create_presentation.py \
  --query "What are the economic impacts discussed?" \
  --output "Economic Impact Analysis.pptx" \
  --verbose
```

**Version 3.0 (With AI-generated images):**

```bash
# Quick start with AI images
python create_presentation_with_images.py

# Create 7-slide presentation with custom query
python create_presentation_with_images.py \
  --query "What are the key security threats in the Red Sea?" \
  --num-slides 7 \
  --output "Red_Sea_Threats.pptx"

# Text-only mode (no images)
python create_presentation_with_images.py --no-images

# Verbose mode to see image generation progress
python create_presentation_with_images.py --verbose

# Force database recreation and generate 3-slide presentation
python create_presentation_with_images.py \
  --force-recreate-db \
  --num-slides 3 \
  --output "Quick_Brief.pptx"
```

### Project Structure

```
AI_PPTX_Creator/
├── create_presentation.py              # v2.0: Text-only presentations
├── create_presentation_with_images.py  # v3.0: With AI-generated images 🎨
├── notebooks/
│   └── AI_PPTX_Creator.ipynb           # Interactive notebook
├── pdfs/                               # Source PDF documents
├── pptx/                               # Generated presentations
├── chroma_db/                          # Vector database storage
├── imgs/
│   ├── generated/                       # AI-generated slide images
│   └── rag.webp                         # Documentation diagram
├── requirements.txt                    # Python dependencies
├── .env.example                        # Example environment file
├── .env                                # API keys (not in git)
└── README.md                           # This file
```

### Troubleshooting

**Issue: "OPENAI_API_KEY not found" error**
- Ensure `.env` file exists in project root
- Verify API key is correctly formatted: `OPENAI_API_KEY=sk-...`

**Issue: No PDFs found**
- Check that PDF files are in the `pdfs/` directory
- Verify directory path with `--pdf-dir` argument

**Issue: Vector database errors**
- Try recreating database: `--force-recreate-db`
- Ensure `chroma_db/` directory has write permissions

**Issue: Generated code fails to execute**
- Use `--dry-run` to inspect generated code
- Check that `python-pptx` is installed correctly
- Verify output directory exists and is writable

**Issue: "GOOGLE_API_KEY not found" error (Image Generation)**
- Ensure `.env` file contains `GOOGLE_API_KEY=your_key_here`
- Get your API key from https://aistudio.google.com/app/apikey
- Alternatively, use `GEMINI_API_KEY` environment variable

**Issue: Image generation fails or returns errors**
- Check that you have access to Gemini API (may require billing enabled)
- Verify API quota limits haven't been exceeded
- Use `--no-images` flag to create presentation without images
- Check logs with `--verbose` flag for detailed error messages

**Issue: Images don't appear in presentation**
- Ensure `imgs/generated/` directory exists and is writable
- Check image file was actually created in the directory
- Try using `--verbose` to see image generation progress
- Some slide titles may generate filtered content - check logs

**Issue: "Pillow" or "google-genai" import errors**
- Run `pip install -r requirements.txt` to install all dependencies
- Ensure you're using the latest requirements.txt file
