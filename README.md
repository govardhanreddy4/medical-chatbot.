# AI Medical Assistant Chatbot

This project is a web-based AI medical assistant that uses Groq's Llama 4 Scout model to analyze medical images and provide preliminary health information.

## Features

- **Image Analysis**: Upload medical images (skin conditions, injuries, etc.) for analysis.
- **Text Query**: Ask questions about the uploaded image or general health concerns.
- **Multi-turn Chat**: Continue the conversation with the AI assistant.
- **Theme Toggle**: Switch between Light and Dark modes.
- **Quick Suggestions**: Predefined prompts to help users start conversations.
- **Responsive Design**: Works on both desktop and mobile devices.

## Prerequisites

- Python 3.7+
- pip (Python package installer)
- Groq API Key

## Installation

1.  **Clone the repository** (or download the source code).

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up environment variables**:
    Create a `.env` file in the root directory with your Groq API key:
    ```env
    GROQ_API_KEY=your_groq_api_key_here
    ```

## Usage

1.  **Run the application**:
    ```bash
    python app.py
    ```

2.  **Access the web interface**:
    Open your browser and navigate to `http://[IP_ADDRESS]`.

## Project Structure

```
web_ai_medical_chatbot/
├── app.py              # FastAPI application and API endpoints
├── main.py             # Main script for running the app
├── requirements.txt    # Project dependencies
├── .env                # Environment variables (should not be committed)
├── templates/
│   └── index.html      # Frontend HTML, CSS, and JavaScript
└── static/             # Static files (if any)
```

## Technology Stack

- **Backend**: FastAPI
- **Frontend**: HTML, CSS, Vanilla JavaScript
- **AI Model**: Groq Llama 4 Scout (via API)
- **Environment Management**: python-dotenv