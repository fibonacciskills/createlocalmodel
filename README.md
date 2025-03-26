# Ollama Model Fine-Tuning UI

A Streamlit-based web interface for fine-tuning and comparing [Ollama](https://ollama.ai/) models. This tool provides an intuitive interface for creating custom models through fine-tuning, generating synthetic training data, and comparing model responses.

## Features

- 🎯 Fine-tune Ollama models with custom training data
- 🤖 Generate synthetic training data using existing models
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

### Fine-Tuning a Model

1. Select a base model from the dropdown menu
2. Configure model parameters:
   - Temperature (controls response randomness)
   - Context Window Size (determines token context length)
   - Model Name (for your fine-tuned version)
   - System Prompt (defines model behavior)

3. Choose your training data source:
   - **Upload File**: Upload a text file with training examples
   - **Generate Synthetic Data**: Create training data using an existing model
     - Select a generation model
     - Specify the topic
     - Choose number of examples

4. Review the previews:
   - Training data preview
   - Formatted data structure
   - Complete Modelfile content

5. Click "Create Fine-Tuned Model" to start the fine-tuning process

### Comparing Models

1. Navigate to the "Chat with Model" tab
2. Select two models to compare
3. Enter your prompt in the text input
4. View responses side-by-side

## Training Data Format

Training data should be formatted as question-answer pairs, separated by newlines:
```
Question 1
Answer 1

Question 2
Answer 2
```

## Technical Details

This tool uses Ollama's model creation API to perform fine-tuning. The process involves:

1. Creating a Modelfile with:
   - Base model specification
   - System prompt
   - Training examples in TEMPLATE format
   - Model parameters

2. Using Ollama's create command to generate a new model based on the Modelfile

For more information about Ollama's fine-tuning capabilities:
- [Ollama Model Creation Guide](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)
- [Ollama API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Fine-tuning Best Practices](https://ollama.ai/blog/ollama-models)

## Limitations

- Fine-tuning is performed using Ollama's built-in capabilities
- Training data size may affect model performance
- Model creation time varies based on hardware and base model size

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 