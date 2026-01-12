# AI Slides Creator - Gemini + Nanobanana Setup Guide

This guide will help you set up and run the AI Slides Creator that uses:
- **Gemini API** for content generation and RAG
- **Imagen 3 (Nanobanana)** for image generation
- **Google Slides API** for presentation creation

## Prerequisites

- Python 3.11 or higher
- Google Cloud Platform account
- Conda (optional, but recommended)

## Step 1: Install Dependencies

### Option A: Using Conda (Recommended)

```bash
# Create and activate environment
conda create -n pptx python=3.11 -y
conda activate pptx

# Install dependencies
pip install -r requirements_gemini.txt
```

### Option B: Using pip

```bash
pip install -r requirements_gemini.txt
```

## Step 2: Get Google API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Create API Key"
3. Copy your API key
4. Save it for the next step

## Step 3: Set Up Google Cloud Service Account (for Google Slides API)

### 3.1 Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Note your project ID

### 3.2 Enable Required APIs

1. In the Cloud Console, go to "APIs & Services" > "Library"
2. Search for and enable:
   - **Google Slides API**
   - **Google Drive API**

### 3.3 Create a Service Account

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "Service Account"
3. Fill in the details:
   - Service account name: `ai-slides-creator`
   - Description: "Service account for AI Slides Creator"
4. Click "Create and Continue"
5. Grant the role: **Editor** (or more restricted as needed)
6. Click "Done"

### 3.4 Create and Download Service Account Key

1. Click on the service account you just created
2. Go to the "Keys" tab
3. Click "Add Key" > "Create new key"
4. Select "JSON" format
5. Click "Create"
6. Save the downloaded JSON file securely
7. Rename it to something like `service-account.json`
8. Move it to your project directory

**IMPORTANT**: Never commit this file to Git! It contains sensitive credentials.

## Step 4: Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Google API Key (for Gemini and Imagen)
GOOGLE_API_KEY=your_google_api_key_here

# Path to service account JSON file (for Google Slides API)
GOOGLE_SERVICE_ACCOUNT_FILE=/absolute/path/to/service-account.json
```

**Example:**
```bash
GOOGLE_API_KEY=AIzaSyC1234567890abcdefghijklmnopqrstuv
GOOGLE_SERVICE_ACCOUNT_FILE=/Users/yonatan/AI_PPTX_Creator/service-account.json
```

## Step 5: Add PDF Documents

1. Place your PDF files in the `pdfs/` directory
2. The script will automatically process all PDF files in this directory

## Step 6: Run the Script

### Basic Usage

```bash
python ai_slides_creator_gemini.py
```

### With Custom Query

```bash
python ai_slides_creator_gemini.py \
  --query "What are the key security threats discussed in the document?"
```

### Test Mode (Generate Plan Only, No Slides)

Useful for testing without creating actual Google Slides:

```bash
python ai_slides_creator_gemini.py --no-slides
```

### Verbose Mode (for Debugging)

```bash
python ai_slides_creator_gemini.py --verbose
```

### All Options Combined

```bash
python ai_slides_creator_gemini.py \
  --query "Summarize the main economic impacts" \
  --verbose
```

## How It Works

### 1. Content Generation with Gemini
- Loads and processes PDF documents
- Creates embeddings using Google's embedding model
- Uses RAG (Retrieval-Augmented Generation) with Gemini to generate presentation content
- Structures content into slides with titles and bullet points

### 2. Image Generation with Nanobanana (Imagen 3)
- For each slide, generates a relevant image using Imagen 3
- Uses 16:9 aspect ratio for presentations
- Saves images to `imgs/generated_nanobanana/`

### 3. Google Slides Creation
- Creates a new Google Slides presentation
- Adds title slide
- Adds content slides with text and images
- Returns a shareable link

## Output

After successful execution, you'll see:

```
================================================================================
📊 SUCCESS! Presentation created!
🔗 View at: https://docs.google.com/presentation/d/{presentation_id}
================================================================================
```

Click the link to view and edit your presentation in Google Slides!

## Configuration Options

Edit the `Config` class in `ai_slides_creator_gemini.py` to customize:

```python
class Config:
    # Model settings
    GEMINI_MODEL = "gemini-2.0-flash-exp"  # Or "gemini-pro"
    IMAGEN_MODEL = "imagen-3.0-generate-002"
    TEMPERATURE = 0.7  # 0.0-1.0, higher = more creative

    # Content settings
    NUM_CONTENT_SLIDES = 3  # Number of slides to generate
    BULLETS_PER_SLIDE = 3   # Bullet points per slide

    # Retriever settings
    RETRIEVER_K = 3  # Number of document chunks to retrieve

    # Image settings
    IMAGE_ASPECT_RATIO = "16:9"  # Or "4:3", "1:1"
```

## Troubleshooting

### "GOOGLE_API_KEY not found"
- Make sure your `.env` file is in the project root
- Check that the variable name is exactly `GOOGLE_API_KEY`
- Ensure there are no extra spaces around the `=` sign

### "Service account file required"
- Verify the path in `GOOGLE_SERVICE_ACCOUNT_FILE` is absolute, not relative
- Check that the file exists and is readable
- Ensure the JSON file is valid

### "403 Forbidden" or "Permission Denied" errors
- Make sure you've enabled both Google Slides API and Google Drive API
- Verify the service account has the correct permissions
- Try granting "Editor" role to the service account

### "No PDF files found"
- Check that PDF files are in the `pdfs/` directory
- Ensure files have `.pdf` extension
- Verify the directory path is correct

### Image generation fails
- Check your Google API key is valid
- Verify your quota hasn't been exceeded
- Try with less complex prompts

### Authentication errors
- Delete the service account and create a new one
- Generate a new key file
- Double-check the API key

## Cost Considerations

### Gemini API
- Gemini 2.0 Flash: Free tier available
- Check current pricing at [Google AI Pricing](https://ai.google.dev/pricing)

### Imagen 3 (Nanobanana)
- Charges per generated image
- Check current pricing in Google Cloud Console

### Google Slides API
- **Free** - No charges for API calls

## Advanced Usage

### Use as a Python Module

```python
from ai_slides_creator_gemini import Config, AISlidesCreator

# Load configuration
Config.load_env()

# Create presentation
creator = AISlidesCreator()
success, url = creator.create_presentation(
    query="What are the main findings?",
    use_google_slides=True,
    verbose=True
)

if success:
    print(f"Presentation: {url}")
```

### Generate Multiple Presentations

```python
queries = [
    "Summarize the key findings",
    "What are the main challenges?",
    "List the recommendations"
]

creator = AISlidesCreator()

for i, query in enumerate(queries):
    success, url = creator.create_presentation(
        query=query,
        use_google_slides=True
    )
    print(f"Presentation {i+1}: {url}")
```

## Security Best Practices

1. **Never commit credentials**:
   ```bash
   # Add to .gitignore
   .env
   service-account.json
   *.json
   ```

2. **Use restricted service accounts**: Grant minimum necessary permissions

3. **Rotate keys regularly**: Generate new API keys and service account keys periodically

4. **Use environment-specific keys**: Different keys for development/production

## Support and Issues

- For Google API issues: [Google Cloud Support](https://cloud.google.com/support)
- For Gemini/Imagen issues: [Google AI Forum](https://discuss.ai.google.dev/)
- For LangChain issues: [LangChain Documentation](https://python.langchain.com/)

## Comparison with OpenAI Version

| Feature | OpenAI Version | Gemini Version |
|---------|---------------|----------------|
| Content Generation | GPT-4 Turbo | Gemini 2.0 Flash |
| Image Generation | DALL-E (not included) | Imagen 3 (Nanobanana) |
| Output Format | PowerPoint (PPTX) | Google Slides |
| Cost | Pay per token | Free tier available |
| Embeddings | OpenAI | Google |
| Collaboration | Download file | Cloud-based link |

## Next Steps

1. ✅ Set up your environment
2. ✅ Get API credentials
3. ✅ Add PDF documents
4. ✅ Run the script
5. 🎉 Share your AI-generated presentation!

---

**Happy presenting! 📊✨**
