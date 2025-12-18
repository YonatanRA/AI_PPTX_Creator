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

2. **Configure API Key:**
   Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
   Get your API key from https://platform.openai.com/api-keys

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

**Option 2: Jupyter Notebook**

For interactive exploration and step-by-step execution, use the Jupyter notebook in the `notebooks/` folder:
```bash
jupyter notebook notebooks/AI_PPTX_Creator.ipynb
```

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

**Quick start with defaults:**
```bash
python create_presentation.py
```

**Test code generation without creating presentation:**
```bash
python create_presentation.py --dry-run
```

**Update vector database and create presentation:**
```bash
python create_presentation.py --force-recreate-db
```

**Custom query with verbose output:**
```bash
python create_presentation.py \
  --query "What are the economic impacts discussed?" \
  --output "Economic Impact Analysis.pptx" \
  --verbose
```

### Project Structure

```
AI_PPTX_Creator/
├── create_presentation.py    # Main Python script (NEW)
├── notebooks/
│   └── AI_PPTX_Creator.ipynb # Interactive notebook
├── pdfs/                     # Source PDF documents
├── pptx/                     # Generated presentations
├── chroma_db/                # Vector database storage
├── imgs/                     # Documentation images
├── requirements.txt          # Python dependencies
├── .env                      # API keys (not in git)
└── README.md                 # This file
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
