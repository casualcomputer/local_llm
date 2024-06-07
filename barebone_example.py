from llama_cpp import Llama

# Put the location of to the GGUF model that you've download from HuggingFace here
# model_path = "models/llava-v1.6-mistral-7b.Q4_K_M.gguf"
model_path = "models/llava-v1.6-vicuna-13b.Q4_K_M.gguf"
llm = Llama(model_path=model_path)

# Prompt creation
system_message = "You are a helpful assistant"
user_message = "Q: Name the planets in the solar system? A: "

prompt = f"""<s>[INST] <<SYS>>
{system_message}
<</SYS>>
{user_message} [/INST]"""

# Run the model
output = llm(
  prompt, # Prompt
  max_tokens=32, # Generate up to 32 tokens
  stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
  echo=True # Echo the prompt back in the output
) # Generate a completion, can also call create_completion

#print(output) #json outputs 

print(output["choices"][0]["text"])