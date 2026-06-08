import concurrent.futures
import time
from ollama import Client
from datasets import load_dataset
import re
from rouge_score import rouge_scorer
# from bert_score import score

# 1. Define all three nodes
orch_llm = Client(host='http://orchestrator_llm:11434')
phone_1 = Client(host='http://phone_node_1:11434')
phone_2 = Client(host='http://phone_node_2:11434')

# Use a highly quantized, small model suitable for 2GB RAM limits
MODEL_NAME = 'llama3.2:1b' 

def find_sentence_break(text, ideal_index):
    """
    Finds the nearest sentence boundary after the ideal splitting index.
    Prevents the LLM from receiving cut-off words or incomplete sentences.
    """
    if ideal_index >= len(text):
        return len(text)
        
    # Search for a period, exclamation, or question mark followed by a space or newline
    match = re.search(r'[.!?](?:\s|\n)', text[ideal_index:])
    
    if match:
        # Return the exact index just after the punctuation mark
        return ideal_index + match.end() - 1 
    
    # If no more punctuation is found, just return the end of the text
    return len(text)

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
    
    # the ground truth summary
    ground_truth_summary = sample_bill['summary']

    print(f"Loaded a legal bill with {len(large_document)} characters.")

    # --- 1. CHUNKING (Proportional & Sentence-Aware) ---
    print("\n--- CALCULATING PROPORTIONAL CHUNKS ---")
    
    node_capacities = {
        "Orchestrator LLM": 4.0,
        "Phone 1": 1.5,
        "Phone 2": 1.5
    }
    
    total_cpu = sum(node_capacities.values())
    total_chars = len(large_document)

    # 1. Calculate the ideal mathematical splits
    orch_ideal_size = int(total_chars * (node_capacities["Orchestrator LLM"] / total_cpu))
    phone_1_ideal_size = int(total_chars * (node_capacities["Phone 1"] / total_cpu))
    
    # 2. Adjust splits to the nearest sentence boundary
    orch_actual_split = find_sentence_break(large_document, orch_ideal_size)
    phone_1_actual_split = find_sentence_break(large_document, orch_actual_split + phone_1_ideal_size)

    # 3. Slice the text cleanly
    chunk_1 = large_document[:orch_actual_split]
    chunk_2 = large_document[orch_actual_split : phone_1_actual_split]
    chunk_3 = large_document[phone_1_actual_split:] 
    
    print(f"Total Text Length: {total_chars} characters")
    print(f"Orchestrator chunk: {len(chunk_1)} chars (Ends cleanly: '{chunk_1[-15:].strip()}')")
    print(f"Phone 1 chunk: {len(chunk_2)} chars (Ends cleanly: '{chunk_2[-15:].strip()}')")
    print(f"Phone 2 chunk: {len(chunk_3)} chars")

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

    print("\n==========================================")
    print("             EVALUATION METRICS           ")
    print("==========================================")
    
    # Initialize the ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Compare the LLM's final summary against the human ground truth
    scores = scorer.score(ground_truth_summary, final_summary)
    
    # Print the F1 scores (which balance precision and recall)
    print(f"ROUGE-1 (Word Match):  {scores['rouge1'].fmeasure * 100:.2f}%")
    print(f"ROUGE-2 (Phrase Match): {scores['rouge2'].fmeasure * 100:.2f}%")
    print(f"ROUGE-L (Structure):    {scores['rougeL'].fmeasure * 100:.2f}%")

    # # BERTScore expects lists of strings
    # candidates = [final_summary]
    # references = [ground_truth_summary]
    
    # # Calculate scores (lang="en" forces it to use the optimized English model)
    # P, R, F1 = score(candidates, references, lang="en", verbose=False)
    
    # print(f"Precision: {P.mean().item() * 100:.2f}% (How much of the AI summary is relevant)")
    # print(f"Recall:    {R.mean().item() * 100:.2f}% (How much of the Human summary was captured)")
    # print(f"F1 Score:  {F1.mean().item() * 100:.2f}% (The overall harmonic balance)")

if __name__ == "__main__":
    main()