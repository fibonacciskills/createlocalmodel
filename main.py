import streamlit as st
import time
import os
import json
import subprocess
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
OLLAMA_API_URL = "http://localhost:11434/api"
DEFAULT_BASE_MODEL = "llama2"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_CONTEXT_WINDOW = 4096

# Model purpose presets
MODEL_PRESETS = {
    "creative_writing": {
        "temperature": 0.9,
        "context_window": 4096,
        "description": "Higher temperature for more creative outputs"
    },
    "technical_documentation": {
        "temperature": 0.3,
        "context_window": 4096,
        "description": "Lower temperature for more precise and consistent outputs"
    },
    "data_analysis": {
        "temperature": 0.2,
        "context_window": 8192,
        "description": "Low temperature and larger context for handling data patterns"
    },
    "conversation": {
        "temperature": 0.7,
        "context_window": 4096,
        "description": "Balanced temperature for natural dialogue"
    },
    "code_generation": {
        "temperature": 0.4,
        "context_window": 8192,
        "description": "Moderate temperature and larger context for code generation"
    },
    "custom": {
        "temperature": 0.7,
        "context_window": 4096,
        "description": "Custom settings for specific needs"
    }
}

# Initialize session state
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'chat_history1' not in st.session_state:
    st.session_state.chat_history1 = []
if 'chat_history2' not in st.session_state:
    st.session_state.chat_history2 = []
if 'model_purpose' not in st.session_state:
    st.session_state.model_purpose = None
if 'custom_parameters' not in st.session_state:
    st.session_state.custom_parameters = False

def get_available_models() -> List[str]:
    """Get list of available Ollama models."""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            models = []
            for line in lines[1:]:  # Skip header line
                if line.strip():
                    model_name = line.split()[0]
                    models.append(model_name)
            return models
        else:
            st.error(f"Error getting models: {result.stderr}")
            return []
    except Exception as e:
        st.error(f"Error getting models: {str(e)}")
        return []

def get_model_response(model: str, prompt: str) -> str:
    """Get response from a model using the Ollama API."""
    try:
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        result = subprocess.run(
            ['curl', '-X', 'POST', f'{OLLAMA_API_URL}/generate', 
             '-H', 'Content-Type: application/json',
             '-d', json.dumps(data)],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            response = json.loads(result.stdout)
            return response.get('response', 'No response received')
        else:
            return f"Error: {result.stderr}"
    except Exception as e:
        return f"Error getting response: {str(e)}"

def create_custom_model(model_name: str, base_model: str, system_prompt: str, temperature: float, context_window: int) -> bool:
    """Create a custom model with specific parameters."""
    try:
        # Escape any quotes in the system prompt and wrap it in quotes
        escaped_prompt = system_prompt.replace('"', '\\"')
        modelfile_content = f'''FROM {base_model}
PARAMETER temperature {temperature}
PARAMETER num_ctx {context_window}
SYSTEM "{escaped_prompt}"'''

        with open('Modelfile', 'w') as f:
            f.write(modelfile_content)

        result = subprocess.run(
            ['ollama', 'create', model_name, '-f', 'Modelfile'],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            st.success(f"Successfully created model: {model_name}")
            return True
        else:
            st.error(f"Error creating model: {result.stderr}")
            return False
    except Exception as e:
        st.error(f"Error creating model: {str(e)}")
        return False

def generate_system_prompt(purpose: str, task_description: str) -> str:
    """Generate a system prompt based on the user's purpose and description."""
    return f"""You are an AI assistant specialized in {purpose}. 
Your primary focus is on: {task_description}
Your responses should be focused, relevant, and helpful for this specific purpose.
Always maintain a professional and knowledgeable tone while being clear and concise."""

def main():
    st.title("Local AI Model Customizer")
    
    # Sidebar for model creation
    with st.sidebar:
        st.header("Create Custom Model")
        
        # Model name input
        model_name = st.text_input("Enter model name", "custom_model")
        
        # Base model selection
        base_model = st.selectbox(
            "Select base model",
            get_available_models() or [DEFAULT_BASE_MODEL],
            index=0
        )
        
        # Purpose selection
        st.subheader("What kind of work will you do with this model?")
        purpose_type = st.selectbox(
            "Select the primary purpose",
            list(MODEL_PRESETS.keys()),
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        # Show preset description
        st.info(MODEL_PRESETS[purpose_type]["description"])
        
        # Task description
        task_description = st.text_area(
            "Describe your specific task or use case",
            "e.g., Writing technical blog posts about machine learning",
            height=100
        )
        
        # Parameter customization
        st.subheader("Model Parameters")
        if purpose_type == "custom":
            temperature = st.slider(
                "Temperature (creativity vs. precision)",
                0.0, 1.0,
                MODEL_PRESETS[purpose_type]["temperature"],
                0.1,
                help="Higher values make output more creative, lower values more precise"
            )
            context_window = st.select_slider(
                "Context Window",
                options=[2048, 4096, 8192, 16384],
                value=MODEL_PRESETS[purpose_type]["context_window"],
                help="Larger values allow the model to consider more context"
            )
        else:
            temperature = MODEL_PRESETS[purpose_type]["temperature"]
            context_window = MODEL_PRESETS[purpose_type]["context_window"]
            st.write(f"Temperature: {temperature}")
            st.write(f"Context Window: {context_window}")
        
        # Preview system prompt
        if task_description:
            system_prompt = generate_system_prompt(purpose_type, task_description)
            with st.expander("Preview System Prompt"):
                st.text(system_prompt)
        
        # Create model button
        if st.button("Create Custom Model"):
            if task_description:
                system_prompt = generate_system_prompt(purpose_type, task_description)
                if create_custom_model(model_name, base_model, system_prompt, temperature, context_window):
                    st.session_state.current_model = model_name
                    st.rerun()
            else:
                st.warning("Please describe your specific use case.")
    
    # Main content area - Chat interface
    st.header("Chat Interface")
    
    # Model selection for comparison
    col1, col2 = st.columns(2)
    
    with col1:
        model1 = st.selectbox(
            "Select Model 1",
            get_available_models(),
            index=0
        )
    
    with col2:
        model2 = st.selectbox(
            "Select Model 2",
            get_available_models(),
            index=1
        )
    
    # Chat interface
    st.subheader("Chat")
    
    # Display chat history
    for i in range(len(st.session_state.chat_history1)):
        with st.container():
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**User:** {st.session_state.chat_history1[i]['user']}")
                st.write(f"**{model1}:** {st.session_state.chat_history1[i]['response']}")
            
            with col2:
                st.write(f"**User:** {st.session_state.chat_history2[i]['user']}")
                st.write(f"**{model2}:** {st.session_state.chat_history2[i]['response']}")
    
    # User input
    user_input = st.text_input("Enter your message:")
    
    if st.button("Send Message") and user_input:
        # Get responses from both models
        response1 = get_model_response(model1, user_input)
        response2 = get_model_response(model2, user_input)
        
        # Update chat histories
        st.session_state.chat_history1.append({"user": user_input, "response": response1})
        st.session_state.chat_history2.append({"user": user_input, "response": response2})
        
        # Rerun to update the display
        st.rerun()

if __name__ == "__main__":
    main()
