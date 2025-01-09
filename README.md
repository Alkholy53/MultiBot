# Multimodal LLM Chatbot with Image Captioning, Memory, and Audio I/O

## Project Overview
This project is a multimodal chatbot that processes text, image, and audio inputs. It uses Salesforce's BLIP model for generating dynamic image descriptions, and audio I/O for speech recognition and synthesis. LangChain powers document handling and question answering, while Chroma provides vector storage for enhanced retrieval. The chatbot leverages Ollama LLM to generate accurate, context-aware responses, integrating Retrieval-Augmented Generation (RAG) to retrieve relevant documents for improved performance.

## Features
- **Image Captioning**: Generate captions for uploaded images using BLIP.
- **Text-based Chatbot**: Respond to user queries using LLM-powered QA.
- **Audio I/O**: Accepts audio input and provides speech output.
- **Document Upload**: Supports PDF, CSV, and arXiv uploads for question answering and summarization.
- **Persistent Memory**: Stores chat history in SQLite.
- **RAG Implementation**: Retrieves relevant documents for enhanced responses.

## Directory Structure
```
📂 CHATBOT
├── 📂 src
│   ├── __pycache__                   # Cached Python files
│   ├── blip-image-captioning-base    # BLIP pretrained models
│   ├── chroma_db                     # Chroma vector store for embeddings
│   ├── audio_processing.py           # Audio input/output logic
│   ├── chat_memory.db                # SQLite database for conversation history
│   ├── document_handler.py           # Document upload and processing
│   └── image_caption.py              # Image captioning logic
├── 📂 Test data
│   ├── dog.jpg                       # Sample image for testing
│   ├── test.csv                      # Sample CSV file
│   └── test.pdf                      # Sample PDF file
├── app.py                            # Main Streamlit application
├── chatbot_memory.db                 # Persistent memory for chatbot interactions
├── requirements.txt                  # Python dependencies
```

## Installation

### Prerequisites
- Python 3.8+
- Git

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/username/multimodal-chatbot.git
   cd multimodal-chatbot
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application
```bash
streamlit run app.py
```

### Uploading Documents
- PDFs, CSVs, and arXiv papers can be uploaded for question answering and summarization.

### Image Captioning
- Upload images via the Streamlit UI for automatic caption generation.

### Audio Interaction
- Click the "Speak" button to interact with the chatbot using your voice.

## Environment Variables
Create a `.env` file to store sensitive information (e.g., API keys):
```env
OLLAMA_API_KEY=your_key_here
```

## Contributing
1. Fork the repository.
2. Create a new branch (`feature/new-feature`).
3. Commit changes and open a pull request.

## License
MIT License - see `LICENSE` for details.

## Contact
For issues or suggestions, open an issue on GitHub or reach out to me at [abdallyalkhuoly@gmail.com](abdallyalkhuoly@gmail.com).

