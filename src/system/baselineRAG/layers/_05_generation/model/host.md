Install from pip

Copy
# Install vLLM from pip:
pip install vllm

Copy
# Load and run the model:
vllm serve "AITeamVN/Vi-Qwen2-1.5B-RAG"

Copy
# Call the server using curl:
curl -X POST "http://localhost:8000/v1/chat/completions" \
	-H "Content-Type: application/json" \
	--data '{
		"model": "AITeamVN/Vi-Qwen2-1.5B-RAG",
		"messages": [
			{
				"role": "user",
				"content": "What is the capital of France?"
			}
		]
	}'