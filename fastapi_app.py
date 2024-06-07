from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama

# Initialize the FastAPI app
app = FastAPI()

# Load the LLM model
model_path = "models/llava-v1.6-mistral-7b.Q4_K_M.gguf"
llm = Llama(model_path=model_path)

# Define request and response models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# Define the query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_llm(request: QueryRequest):
    system_message = "You are a helpful assistant"
    user_message = f"Q: {request.question} A: "

    prompt = f"""<s>[INST] <<SYS>>
{system_message}
<</SYS>>
{user_message} [/INST]"""

    try:
        # Run the model to get the response
        output = llm(
            prompt,  # Prompt
            max_tokens=2000,  # Generate up to 2000 tokens
            stop=["Q:", "\n"],  # Stop generating just before the model would generate a new question
            echo=False  # Do not echo the prompt back in the output
        )

        # Extract and return the response
        response_text = output["choices"][0]["text"].strip()

        # Ensure the response is trimmed properly
        response_text = response_text.split("[/INST]")[-1].strip()

        return QueryResponse(answer=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Template for sending a request 
# curl -X POST "http://localhost:8000/query" -H "Content-Type: application/json" -d "{\"question\": \"Name the planets in the solar system?\"}"

# To create a public url (in this case https://fb44-2606-40-408-2d3-00-460-439d.ngrok-free.app/query) and forward it to local port 8000, install grok from cmd and type (ngrok http 8000)
# curl -X POST "https://fb44-2606-40-408-2d3-00-460-439d.ngrok-free.app/query" -H "Content-Type: application/json" -d "{\"question\": \"Name the planets in the solar system?\"}"
