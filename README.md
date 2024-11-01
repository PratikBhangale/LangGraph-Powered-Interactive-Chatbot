# LangGraph-Powered-Interactive-Chatbot
LangGraph-Powered Interactive Chatbot: Enhanced Document Retrieval and Web Search Integration

## Overview
This project is an interactive chatbot powered by LangGraph, designed to provide enhanced document retrieval and web search capabilities. The chatbot can answer user queries by leveraging a combination of stored documents and real-time web search results.

## Features
- **Interactive Chat Interface**: Users can ask questions and receive responses in real-time.
- **Document Retrieval**: The chatbot retrieves relevant information from a vector store of documents.
- **Web Search Integration**: For questions outside the document scope, the chatbot performs web searches to provide accurate answers.
- **Session Management**: Maintains chat history for context-aware responses.

## Technologies Used
- **LangChain**: For building the chatbot and managing document retrieval.
- **LangGraph**: For creating a state graph to manage the flow of the chatbot's logic.
- **Streamlit**: For building the interactive web interface.
- **Ollama**: For leveraging language models in the chatbot.
- **FAISS**: For efficient similarity search and retrieval of documents.
- **Python**: The primary programming language used for development.

## Installation
1. Clone the repository:
   ```bash
   git clone "https://github.com/PratikBhangale/LangGraph-Powered-Interactive-Chatbot.git"
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the root directory and add the necessary environment variables (excluding sensitive information).

## Usage
1. Start the Streamlit app:
   ```bash
   streamlit run pages/"Chat With Tools.py"
   ```

2. Upload PDF documents to create a data store:
   - Navigate to the "Creating a Data Store" page and upload your PDF files.

3. Interact with the chatbot:
   - Ask questions in the chat interface, and the chatbot will respond based on the context of the uploaded documents and web search results.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments
- Thanks to the developers of LangChain, LangGraph, Streamlit, and Ollama for their contributions to the open-source community.
