import concurrent.futures
import time
from ollama import Client
from datasets import load_dataset

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
    print(f"[{node_name}] Starting {task_type} task on {len(text_chunk)} characters...")
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
    setup_node(phone_1, "Phone 1")
    setup_node(phone_2, "Phone 2")

    print("\n--- DOWNLOADING BILLSUM DATASET ---")
    # Load the California test split of the BillSum dataset
    dataset = load_dataset("billsum", split="ca_test")
    
    # Grab the very first bill in the dataset
    sample_bill = dataset[0]
    
    # The 'text' column contains the full legal text of the bill
    large_document = sample_bill['text']
    
    print(f"Loaded a legal bill with {len(large_document)} characters.")

    # --- 1. CHUNKING ---
    # Split the document roughly in half by characters
    midpoint = len(large_document) // 2
    chunk_1 = large_document[:midpoint]
    chunk_2 = large_document[midpoint:]

    print("\n--- STARTING MAP PHASE (Parallel Execution) ---")
    
    # --- 2. THE MAP STEP ---
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Dispatch tasks to the resource-throttled containers simultaneously 
        future_1 = executor.submit(summarize_chunk, phone_1, "Phone 1", chunk_1, "Map")
        future_2 = executor.submit(summarize_chunk, phone_2, "Phone 2", chunk_2, "Map")
        
        # Wait for both edge devices to return their summaries
        summary_1 = future_1.result()
        summary_2 = future_2.result()

    print("\n--- MAP PHASE COMPLETE ---")
    print(f"Summary 1 Snippet: {summary_1[:100]}...")
    print(f"Summary 2 Snippet: {summary_2[:100]}...")

    # --- 3. THE REDUCE STEP ---
    print("\n--- STARTING REDUCE PHASE (Recombination) ---")
    combined_summaries = f"Summary Part 1:\n{summary_1}\n\nSummary Part 2:\n{summary_2}"
    
    # We send the combined summaries back to Phone 1 to generate the final master summary.
    final_summary = summarize_chunk(phone_1, "Phone 1", combined_summaries, "Reduce")

    print("\n==========================================")
    print("           FINAL MASTER SUMMARY           ")
    print("==========================================")
    print(final_summary)

if __name__ == "__main__":
    main()