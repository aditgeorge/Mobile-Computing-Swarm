import concurrent.futures
import time
from ollama import Client
from datasets import load_dataset

# 1. Define all three nodes
orch_llm = Client(host='http://orchestrator_llm:11434')
phone_1 = Client(host='http://phone_node_1:11434')
phone_2 = Client(host='http://phone_node_2:11434')

# Use a highly quantized, small model suitable for 2GB RAM limits
MODEL_NAME = 'llama3.2:1b' 

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
    setup_node(orch_llm, "Orchestrator LLM")
    setup_node(phone_1, "Phone 1")
    setup_node(phone_2, "Phone 2")

    print("\n--- DOWNLOADING BILLSUM DATASET ---")
    # Load the California test split of the BillSum dataset
    dataset = load_dataset("FiscalNote/billsum", split="ca_test")
    
    # Grab the very first bill in the dataset
    sample_bill = dataset[0]
    
    # The 'text' column contains the full legal text of the bill
    large_document = sample_bill['text']
    
    print(f"Loaded a legal bill with {len(large_document)} characters.")

    # --- 1. CHUNKING (Now in 3 parts) ---
    # Divide the document into thirds
    third = len(large_document) // 3
    chunk_1 = large_document[:third]
    chunk_2 = large_document[third:third*2]
    chunk_3 = large_document[third*2:]

    print("\n--- STARTING MAP PHASE (Parallel Execution) ---")
    
    # --- 2. THE MAP STEP ---
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Give the first chunk to the powerful Orchestrator LLM
        future_orch = executor.submit(summarize_chunk, orch_llm, "Orchestrator LLM", chunk_1, "Map")
        
        # Give the remaining chunks to the edge phones
        future_1 = executor.submit(summarize_chunk, phone_1, "Phone 1", chunk_2, "Map")
        future_2 = executor.submit(summarize_chunk, phone_2, "Phone 2", chunk_3, "Map")
        
        # Wait for everyone to finish
        summary_orch = future_orch.result()
        summary_1 = future_1.result()
        summary_2 = future_2.result()

    print("\n--- MAP PHASE COMPLETE ---")

    # --- 3. THE REDUCE STEP ---
    print("\n--- STARTING REDUCE PHASE (Recombination) ---")
    combined_summaries = (
        f"Part 1 (Beginning):\n{summary_orch}\n\n"
        f"Part 2 (Middle):\n{summary_1}\n\n"
        f"Part 3 (End):\n{summary_2}"
    )
    
    # We send the final task to the powerful Orchestrator node
    # Since it has 8GB RAM, it is much better equipped to read 3 summaries at once
    final_summary = summarize_chunk(orch_llm, "Orchestrator LLM", combined_summaries, "Reduce")

    print("\n==========================================")
    print("           FINAL MASTER SUMMARY           ")
    print("==========================================")
    print(final_summary)

if __name__ == "__main__":
    main()