# Ollama Model Custom Assistants

A Streamlit-based web interface for customizing and comparing models available through [Ollama](https://ollama.ai/). This tool provides an intuitive interface for creating custom model assistants through adjusting model parameters and system messages and chatting with the new model. 

## Features

- 📊 Side-by-side model comparison interface
- 🔄 Real-time model response previews
- 📝 Custom system prompt configuration
- 🌡️ Adjustable model parameters (temperature, context window)

## Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running
- At least one base model pulled in Ollama (e.g., llama2, mistral, etc.)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install streamlit
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run main.py
```

2. Access the web interface at `http://localhost:8501`

![Screenshot 2025-03-26 at 9 36 00 AM](https://github.com/user-attachments/assets/abdae95d-1178-43db-a07f-45d7d4053083)

### Customizing a Model

1. Select a base model from the dropdown menu
2. Configure model parameters:
   - Temperature (controls response randomness)
   - Context Window Size (determines token context length)
   - Model Name (for your fine-tuned version)
   - System Prompt (defines model behavior)

3. Review the previews:
   - Training data preview
   - Formatted data structure
   - Complete Modelfile content

4. Click "Create Custom Model" to start the fine-tuning process

### Comparing Models

1. Navigate to the "Chat with Model" tab
2. Select two models to compare
3. Enter your prompt in the text input
4. View responses side-by-side

## Technical Details

This tool uses Ollama's model creation API to perform creation. The process involves:

1. Creating a Modelfile with:
   - Base model specification
   - System prompt
   - Model parameters

2. Using Ollama's create command to generate a new model based on the Modelfile

For more information about Ollama's fine-tuning capabilities:
- [Ollama Model Creation Guide](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)
- [Ollama API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Fine-tuning Best Practices](https://ollama.ai/blog/ollama-models)

## Limitations

- Model creation time varies based on hardware and base model size

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 


