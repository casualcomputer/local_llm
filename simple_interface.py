import tkinter as tk
from tkinter import scrolledtext, messagebox
from llama_cpp import Llama
import time
import torch

# Load the model
# Replace with your model path
model_path = "models/llava-v1.6-vicuna-13b.Q4_K_M.gguf"

# Initialize the LLM with specific configurations
llm = Llama(
    model_path=model_path,
    n_gpu_layers=20,  # Use GPU acceleration
    seed=1337,  # Set a specific seed for reproducibility
    n_ctx=10000,  # Increase the context window
    use_fp16=True  # Use mixed precision (FP16) instead of full precision (FP32)
)

# Detect if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def generate_response():
    """
    Function to generate a response from the model based on user input.
    """
    user_input = user_query.get().strip()
    if not user_input:
        messagebox.showinfo("Input Required", "Please enter a question.")
        return

    start_time = time.time()  # Start timing
    
    # Create the prompt
    prompt = f"""<s>[INST] <<SYS>>
You are a helpful assistant
<</SYS>>
Q: {user_input} A:  [/INST]"""
    
    # Generate the model output
    output = llm(
        prompt, 
        max_tokens=512, 
        temperature=0.1, 
        stop=["Q:", "\n"], 
        echo=False
    )
    
    response = output["choices"][0]["text"].strip()  # Strip leading/trailing whitespaces
    
    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time  # Calculate elapsed time
    
    # Display user question and model response in the chat
    chat_history.config(state='normal')
    chat_history.insert(tk.END, f"You: {user_input}\n", 'user')
    chat_history.insert(tk.END, f"Assistant: {response} (Time: {elapsed_time:.2f}s)\n", 'assistant')
    chat_history.config(state='disabled')
    chat_history.see(tk.END)  # Auto-scroll to the bottom
    
    user_query.delete(0, tk.END)  # Clear input field after sending

# Set up the main window
root = tk.Tk()
root.title("Chat with LLaMA Model")

# Configure text tag styles for the chat history
chat_history = scrolledtext.ScrolledText(root, height=20, width=80)
chat_history.tag_configure('user', foreground='blue')
chat_history.tag_configure('assistant', foreground='green')
chat_history.config(state='disabled', padx=10, pady=10)
chat_history.pack(padx=20, pady=10)

# Entry for user to type their question
user_query = tk.Entry(root, width=70)
user_query.pack(side=tk.LEFT, padx=(20, 10), pady=(0, 20))

# Send button to submit the query
send_button = tk.Button(root, text="Send", command=generate_response)
send_button.pack(side=tk.RIGHT, padx=(0, 20), pady=(0, 20))

# Start the GUI event loop
root.mainloop()
