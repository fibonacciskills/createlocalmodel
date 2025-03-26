import streamlit as st
import time
import os
import json
import tempfile
import subprocess

# Set up constants
MODELFILE_TEMPLATE = '''FROM {base_model}

# sets the temperature to {temperature} [higher is more creative, lower is more coherent]
PARAMETER temperature {temperature}
# sets the context window size to {context_window}, this controls how many tokens the LLM can use as context to generate the next token
PARAMETER num_ctx {context_window}

# sets a custom system message to specify the behavior of the chat assistant
SYSTEM "You are an AI assistant specializing in local AI development. Provide detailed, informative, and varied responses to user queries."

# Training examples
{training_data}'''

def format_training_data(data):
    """Format raw text data into Ollama's expected training format."""
    formatted_data = []
    # Split data into chunks (assuming each chunk is separated by double newlines)
    chunks = data.strip().split('\n\n')
    
    for chunk in chunks:
        # Split chunk into user and assistant parts (assuming they're separated by a newline)
        parts = chunk.strip().split('\n')
        if len(parts) >= 2:
            user_msg = parts[0].strip().replace('"', '\\"')
            assistant_msg = '\n'.join(parts[1:]).strip().replace('"', '\\"')
            formatted_data.append(f'TEMPLATE """Human: {user_msg}\nAssistant: {assistant_msg}"""')
    
    return '\n\n'.join(formatted_data)

def get_model_response(model_name, prompt):
    """Get response from a model."""
    try:
        cmd = ['ollama', 'run', model_name, prompt]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error: {result.stderr}"
    except Exception as e:
        return f"Error: {str(e)}"

def generate_synthetic_data(model_name, topic_prompt, num_examples=10):
    """Generate synthetic training data using a local model."""
    generation_prompt = f"""Generate {num_examples} training examples about {topic_prompt}. 
Each example should be a question-answer pair about {topic_prompt}.
Format each example as:
Question
Detailed answer

Separate each example with a blank line.
Make questions diverse but focused on {topic_prompt}.
Keep answers clear, direct, and informative."""

    try:
        response = get_model_response(model_name, generation_prompt)
        return response.strip()
    except Exception as e:
        return f"Error generating synthetic data: {str(e)}"

st.title("Fine-Tune & Chat with Your Ollama Models")

# Query available models from Ollama
try:
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
    available_models = [line.split()[0] for line in result.stdout.strip().split('\n')[1:]]
except Exception as e:
    available_models = []
    st.error(f"Error fetching models from Ollama: {e}")

if not available_models:
    st.warning("No local models found via Ollama. Please ensure you have models loaded.")
else:
    # Let the user select from the available models
    selected_model = st.selectbox("Select a base model", available_models)

# Create tabs for Training and Chat
tab_train, tab_chat = st.tabs(["Train Model", "Chat with Model"])

# ----------------------------
# TRAINING TAB
# ----------------------------
with tab_train:
    st.header("Fine Tune Your Selected Model")
    
    # Model configuration
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
        context_window = st.number_input("Context Window Size", min_value=512, max_value=8192, value=2048, step=512)
    
    with col2:
        new_model_name = st.text_input("New Model Name", value="my-custom-model")
        system_prompt = st.text_area("System Prompt", value="You are a helpful AI assistant. ")

    # Data source selection
    data_source = st.radio("Choose Training Data Source", ["Upload File", "Generate Synthetic Data"])
    
    training_data = None
    
    if data_source == "Upload File":
        # File uploader for training data
        uploaded_file = st.file_uploader("Upload your training data (txt)", type=["txt"])
        if uploaded_file is not None:
            training_data = uploaded_file.read().decode("utf-8")
            st.session_state.current_training_data = training_data
    else:
        # Synthetic data generation
        st.subheader("Generate Synthetic Training Data")
        
        # Filter out the selected base model from available models for generation
        generation_models = [m for m in available_models if m != selected_model]
        if not generation_models:
            st.warning("No other models available for synthetic data generation.")
        else:
            gen_model = st.selectbox("Select Model for Generation", generation_models)
            topic_prompt = st.text_area("Topic for Training Examples", 
                value="early childhood development milestones and assessment methods")
            num_examples = st.number_input("Number of Examples", min_value=1, max_value=50, value=10)
            
            if st.button("Generate Training Data"):
                with st.spinner("Generating synthetic training data..."):
                    generated_data = generate_synthetic_data(gen_model, topic_prompt, num_examples)
                    if not generated_data.startswith("Error"):
                        st.session_state.current_training_data = generated_data
                        training_data = generated_data
                    else:
                        st.error(generated_data)
    
    # Use the training data from session state if available
    if not training_data and 'current_training_data' in st.session_state:
        training_data = st.session_state.current_training_data
    
    # Preview and process training data
    if training_data:
        st.write("Data preview:")
        st.text(training_data[:500] + "..." if len(training_data) > 500 else training_data)
        
        # Show formatted data preview
        formatted_data = format_training_data(training_data)
        with st.expander("Preview Formatted Training Data"):
            st.text(formatted_data[:500] + "..." if len(formatted_data) > 500 else formatted_data)
        
        # Show complete Modelfile preview
        with st.expander("Preview Complete Modelfile"):
            modelfile_content = MODELFILE_TEMPLATE.format(
                base_model=selected_model,
                temperature=temperature,
                context_window=context_window,
                system_prompt=system_prompt.strip(),
                training_data=formatted_data
            )
            
            # Debug: Log the Modelfile content
            st.write("Modelfile Content Debug:")
            st.text(modelfile_content)  # Display the Modelfile content
        
        # Create a column for the fine-tuning button
        col1, col2 = st.columns([1, 3])
        with col1:
            create_model = st.button("Create Fine-Tuned Model")
        
        # Start the training process
        if create_model:
            st.write("Creating fine-tuned model...")
            
            try:
                # Create temporary Modelfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.modelfile', delete=False) as f:
                    modelfile_content = MODELFILE_TEMPLATE.format(
                        base_model=selected_model,
                        temperature=temperature,
                        context_window=context_window,
                        system_prompt=system_prompt.strip(),
                        training_data=formatted_data
                    )
                    f.write(modelfile_content)
                    modelfile_path = f.name
                
                # Show the exact content being written to the file
                st.write("Modelfile content:")
                st.text(modelfile_content)
                
                # Create the model using ollama create
                create_cmd = ['ollama', 'create', new_model_name, '-f', modelfile_path]
                result = subprocess.run(create_cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    st.success(f"Successfully created model: {new_model_name}")
                    st.session_state.current_model = new_model_name
                    # Clear the training data from session state after successful model creation
                    if 'current_training_data' in st.session_state:
                        del st.session_state.current_training_data
                else:
                    st.error(f"Error creating model: {result.stderr}")
                    st.error("Command output:")
                    st.text(result.stdout)
                
                # Clean up temporary file
                os.unlink(modelfile_path)
                
            except Exception as e:
                st.error(f"Error during model creation: {str(e)}")
                
        # Add a clear data button
        if st.button("Clear Training Data"):
            if 'current_training_data' in st.session_state:
                del st.session_state.current_training_data
            st.rerun()

# ----------------------------
# CHAT TAB
# ----------------------------
with tab_chat:
    st.header("Compare Models")
    
    # Select models to chat with
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        chat_models = [line.split()[0] for line in result.stdout.strip().split('\n')[1:]]
    except Exception as e:
        chat_models = []
        st.error(f"Error fetching models: {e}")
    
    if chat_models:
        # Create two columns for model selection
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model 1")
            model1 = st.selectbox("Select first model", chat_models, key="model1")
        
        with col2:
            st.subheader("Model 2")
            model2 = st.selectbox("Select second model", chat_models, key="model2")
        
        # Initialize chat histories if they don't exist
        if "chat_history1" not in st.session_state:
            st.session_state.chat_history1 = []
        if "chat_history2" not in st.session_state:
            st.session_state.chat_history2 = []
        
        # Display chat histories side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Chat with {model1}**")
            for entry in st.session_state.chat_history1:
                st.write(f"**You:** {entry['user']}")
                st.write(f"**{model1}:** {entry['response']}")
        
        with col2:
            st.write(f"**Chat with {model2}**")
            for entry in st.session_state.chat_history2:
                st.write(f"**You:** {entry['user']}")
                st.write(f"**{model2}:** {entry['response']}")
        
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
            
        # Add a clear button to reset chat histories
        if st.button("Clear Chat History"):
            st.session_state.chat_history1 = []
            st.session_state.chat_history2 = []
            st.rerun()
    else:
        st.warning("No models available for chat. Please create a model first.")
