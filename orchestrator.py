import concurrent.futures
import time
import re
import os
import csv
import requests
from ollama import Client
from datasets import load_dataset
from rouge_score import rouge_scorer

# Flexible criteria
MODEL_NAME = 'llama3.2:1b' 
total_rows_to_process = 1
node_capacities = {"Orchestrator LLM": 4.0, "Phone 1": 1.5, "Phone 2": 1.5}
hf_dataset_name = "FiscalNote/billsum"
hf_dataset_split = "ca_test"


# Define connection endpoints
orch_llm = Client(host='http://orchestrator_llm:11434')
phone_1 = Client(host='http://phone_node_1:11434')
phone_2 = Client(host='http://phone_node_2:11434')


def find_sentence_break(text, ideal_index):
    if ideal_index >= len(text):
        return len(text)
    match = re.search(r'[.!?](?:\s|\n)', text[ideal_index:])
    if match:
        return ideal_index + match.end() - 1 
    return len(text)

def setup_node(client, node_name):
    print(f"[{node_name}] Checking for model '{MODEL_NAME}'...")
    try:
        client.pull(MODEL_NAME)
        print(f"[{node_name}] Model ready.")
    except Exception as e:
        print(f"[{node_name}] Failed to reach node. Error: {e}")

def summarize_chunk(client, node_name, text_chunk, task_type="Map"):
    messages = [
        {'role': 'system', 'content': 'You are a highly efficient summarization assistant. Provide a concise summary.'},
        {'role': 'user', 'content': f"Please summarize the following text:\n\n{text_chunk}"}
    ]
    safe_options = {'num_ctx': 2048, 'num_predict': 500}
    
    try:
        response = client.chat(model=MODEL_NAME, messages=messages, options=safe_options)
        
        # Extract the raw text summary
        summary_text = response['message']['content']
        
        # Extract Ollama's native nanosecond parameters and convert to seconds
        prompt_tokens = response.get('prompt_eval_count', 0)
        ttft_seconds = response.get('prompt_eval_duration', 0) / 1e9
        
        output_tokens = response.get('eval_count', 0)
        gen_duration_seconds = response.get('eval_duration', 0) / 1e9
        total_time = response.get('total_duration', 0) / 1e9
        
        # Compute speed metrics
        tps = output_tokens / gen_duration_seconds if gen_duration_seconds > 0 else 0
        
        return {
            "text": summary_text,
            "metrics": {
                "ttft": round(ttft_seconds, 3),
                "tps": round(tps, 2),
                "total_time": round(total_time, 3)
            }
        }
    except requests.exceptions.ConnectionError:
        print(f"🚨 [{node_name}] Container likely crashed due to OOM.")
        return {"text": "[FAILED DUE TO OOM]", "metrics": {"ttft": 0, "tps": 0, "total_time": 0}}
    except Exception as e:
        print(f"🚨 [{node_name}] Error: {e}")
        return {"text": "[ERROR]", "metrics": {"ttft": 0, "tps": 0, "total_time": 0}}

def main():
    print("--- INITIALIZING ALL NODES ---")
    setup_node(orch_llm, "Orchestrator LLM")
    setup_node(phone_1, "Phone 1")
    setup_node(phone_2, "Phone 2")

    print("\n--- DOWNLOADING BILLSUM DATASET ---")
    dataset = load_dataset(hf_dataset_name, split=hf_dataset_split)
    
    r_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    output_dir = "/app/outputs"
    output_file = os.path.join(output_dir, "evaluation_results.csv")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define comprehensive table tracking structure
    headers = [
        "Timestamp", "Testcase_ID", 
        "ROUGE_1", "ROUGE_2", "ROUGE_L", 
        "Orch_Map_TTFT", "Orch_Map_TPS",
        "Phone1_Map_TTFT", "Phone1_Map_TPS",
        "Phone2_Map_TTFT", "Phone2_Map_TPS",
        "Reduce_TTFT", "Reduce_TPS", 
        "Total_Pipeline_Time", "Generated_Summary"
    ]

    if not os.path.isfile(output_file):
        with open(output_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    print(f"\n--- STARTING BATCH PROCESSING ({total_rows_to_process} ROWS) ---")
    
    for testcase_index in range(total_rows_to_process):
        print(f"\n--- [ROW {testcase_index + 1} / {total_rows_to_process}] ---")
        pipeline_start = time.time()
        
        try:
            sample_bill = dataset[testcase_index]
            large_document = sample_bill['text']
            ground_truth_summary = sample_bill['summary'] 
            
            total_cpu = sum(node_capacities.values())
            total_chars = len(large_document)

            orch_ideal = int(total_chars * (node_capacities["Orchestrator LLM"] / total_cpu))
            phone_1_ideal = int(total_chars * (node_capacities["Phone 1"] / total_cpu))
            
            orch_actual = find_sentence_break(large_document, orch_ideal)
            phone_1_actual = find_sentence_break(large_document, orch_actual + phone_1_ideal)

            chunk_1 = large_document[:orch_actual]
            chunk_2 = large_document[orch_actual : phone_1_actual]
            chunk_3 = large_document[phone_1_actual:] 

            # --- MAP PHASE ---
            print("Running parallel Map phase...")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_orch = executor.submit(summarize_chunk, orch_llm, "Orchestrator LLM", chunk_1, "Map")
                future_1 = executor.submit(summarize_chunk, phone_1, "Phone 1", chunk_2, "Map")
                future_2 = executor.submit(summarize_chunk, phone_2, "Phone 2", chunk_3, "Map")
                
                res_orch = future_orch.result()
                res_phone1 = future_1.result()
                res_phone2 = future_2.result()

            # --- REDUCE PHASE ---
            print("Running Reduce phase...")
            combined_summaries = f"Part 1:\n{res_orch['text']}\n\nPart 2:\n{res_phone1['text']}\n\nPart 3:\n{res_phone2['text']}"
            res_reduce = summarize_chunk(orch_llm, "Orchestrator LLM", combined_summaries, "Reduce")

            # --- ACCURACY EVALUATION ---
            r_scores = r_scorer.score(ground_truth_summary, res_reduce['text'])
            r1 = r_scores['rouge1'].fmeasure * 100
            r2 = r_scores['rouge2'].fmeasure * 100
            rl = r_scores['rougeL'].fmeasure * 100

            pipeline_total_time = time.time() - pipeline_start
            print(f"Success! R1: {r1:.2f}% | Total Time: {pipeline_total_time:.2f}s")

            # --- WRITE RESULTS TO CSV ---
            with open(output_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    testcase_index,
                    f"{r1:.2f}%", f"{r2:.2f}%", f"{rl:.2f}%",
                    f"{res_orch['metrics']['ttft']}s", f"{res_orch['metrics']['tps']}",
                    f"{res_phone1['metrics']['ttft']}s", f"{res_phone1['metrics']['tps']}",
                    f"{res_phone2['metrics']['ttft']}s", f"{res_phone2['metrics']['tps']}",
                    f"{res_reduce['metrics']['ttft']}s", f"{res_reduce['metrics']['tps']}",
                    f"{pipeline_total_time:.2f}s",
                    res_reduce['text'].replace('\n', ' ')
                ])

        except Exception as e:
            print(f"🚨 FATAL ERROR ON ROW {testcase_index}: {e}")
            with open(output_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Create a placeholder row matching the exact number of header columns
                error_row = [time.strftime("%Y-%m-%d %H:%M:%S"), testcase_index] + ["ERROR"] * 12 + [f"Failed: {e}"]
                writer.writerow(error_row)
            continue 

    print(f"\nBatch processing complete. Data stored in: {output_file}")

if __name__ == "__main__":
    main()