import concurrent.futures
import time
from ollama import Client

# 1. Define the Edge Nodes based on Docker Compose service names
# Docker's internal DNS automatically resolves these names to the correct containers
phone_1 = Client(host='http://phone_node_1:11434')
phone_2 = Client(host='http://phone_node_2:11434')

# Use a highly quantized, small model suitable for 2GB RAM limits
MODEL_NAME = 'qwen:0.5b' 

def setup_node(client, node_name):
    """Ensures the LLM is downloaded to the simulated phone before starting."""
    print(f"[{node_name}] Checking for model '{MODEL_NAME}'...")
    client.pull(MODEL_NAME)
    print(f"[{node_name}] Model ready.")

def summarize_chunk(client, node_name, text_chunk, task_type="Map"):
    """Sends a text chunk to a phone node to be summarized."""
    print(f"[{node_name}] Starting {task_type} task...")
    start_time = time.time()
    
    # The prompt instructs the tiny LLM on how to behave
    messages = [
        {'role': 'system', 'content': 'You are a highly efficient summarization assistant. Provide concise summaries.'},
        {'role': 'user', 'content': f"Please summarize the following text:\n\n{text_chunk}"}
    ]
    
    # This call blocks while the throttled Docker container struggles to generate text
    response = client.chat(model=MODEL_NAME, messages=messages)
    summary = response['message']['content']
    
    elapsed = time.time() - start_time
    print(f"[{node_name}] Finished {task_type} in {elapsed:.2f} seconds.")
    
    return summary

def main():
    print("--- INITIALIZING EDGE NODES ---")
    # In a real scenario, models are pre-loaded. Here, we force Docker to pull them.
    setup_node(phone_1, "Phone 1")
    setup_node(phone_2, "Phone 2")

    # A sample large text (In reality, this would be a massive document or article)
    large_document = (
        "The history of artificial intelligence (AI) began in antiquity, with myths, "
        "stories and rumors of artificial beings endowed with intelligence or consciousness "
        "by master craftsmen. The seeds of modern AI were planted by philosophers who "
        "attempted to describe the process of human thinking as the mechanical manipulation "
        "of symbols. This work culminated in the invention of the programmable digital computer "
        "in the 1940s, a machine based on the abstract essence of mathematical reasoning. "
        "This device and the ideas behind it inspired a handful of scientists to begin "
        "seriously discussing the possibility of building an electronic brain. "
        "The field of AI research was founded at a workshop held on the campus of Dartmouth "
        "College, USA during the summer of 1956. Those who attended would become the "
        "leaders of AI research for decades."
    )

    # --- 1. CHUNKING ---
    # Splitting the text. For a production app, use an NLP tokenizer like NLTK or tiktoken.
    # For this simulation, we split it roughly in half by characters.
    midpoint = len(large_document) // 2
    chunk_1 = large_document[:midpoint]
    chunk_2 = large_document[midpoint:]

    print("\n--- STARTING MAP PHASE (Parallel Execution) ---")
    
    # --- 2. THE MAP STEP ---
    # ThreadPoolExecutor runs the API calls at the exact same time. 
    # Because both Docker containers are throttled to 1.5 CPUs, this will take time.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Dispatch tasks
        future_1 = executor.submit(summarize_chunk, phone_1, "Phone 1", chunk_1, "Map")
        future_2 = executor.submit(summarize_chunk, phone_2, "Phone 2", chunk_2, "Map")
        
        # Wait for both edge devices to return their summaries
        summary_1 = future_1.result()
        summary_2 = future_2.result()

    print("\n--- MAP PHASE COMPLETE ---")
    print(f"Summary 1 Snippet: {summary_1[:50]}...")
    print(f"Summary 2 Snippet: {summary_2[:50]}...")

    # --- 3. THE REDUCE STEP ---
    print("\n--- STARTING REDUCE PHASE (Recombination) ---")
    combined_summaries = f"Summary Part 1:\n{summary_1}\n\nSummary Part 2:\n{summary_2}"
    
    # We send the combined summaries back to Phone 1 to generate the final master summary.
    # (Since the orchestrator just runs Python, it delegates the final LLM task to an edge node).
    final_summary = summarize_chunk(phone_1, "Phone 1", combined_summaries, "Reduce")

    print("\n==========================================")
    print("           FINAL MASTER SUMMARY           ")
    print("==========================================")
    print(final_summary)

if __name__ == "__main__":
    main()