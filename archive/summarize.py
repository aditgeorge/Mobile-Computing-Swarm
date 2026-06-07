import ollama
import urllib.request
import json

def get_model_memory():
    """Queries the local Ollama server to see how much RAM the loaded models are using."""
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/ps")
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            if data.get('models'):
                # Grab the memory footprint of the active model and convert bytes to MB
                size_bytes = data['models'][0]['size']
                return size_bytes / (1024 * 1024) 
    except Exception:
        return 0
    return 0

def summarize_with_accurate_metrics(text, model_name="llama3.2:1b"):
    print(f"📱 Loading '{model_name}' and processing text...\n")
    
    prompt = f"Please summarize the following text:\n\n{text}"
    
    try:
        # 1. Generate the response
        response = ollama.chat(model=model_name, messages=[
            {'role': 'user', 'content': prompt}
        ])
        
        # 2. Immediately check the model's RAM usage
        model_ram_mb = get_model_memory()
        
        # 3. Extract Time and Token metrics (converting nanoseconds to seconds)
        ttft_seconds = response.get('prompt_eval_duration', 0) / 1e9
        generation_time = response.get('eval_duration', 0) / 1e9
        
        output_tokens = response.get('eval_count', 0)
        
        # 4. Calculate Tokens Per Second (TPS)
        tps = output_tokens / generation_time if generation_time > 0 else 0
        
        # --- Print Results ---
        print("✨ SUMMARY ✨")
        print(response['message']['content'].strip())
        print("\n" + "="*40)
        print("📊 CORE PERFORMANCE METRICS 📊")
        print("="*40)
        
        # Display TTFT
        print(f"Time to First Token (TTFT): {ttft_seconds:.2f} seconds")
        
        # Display TPS
        print(f"Generation Speed (TPS):     {tps:.2f} tokens/second")
        
        # Display RAM Usage
        if model_ram_mb > 0:
            print(f"Model Memory Footprint:     {model_ram_mb:.2f} MB")
        else:
            print("Model Memory Footprint:     Could not fetch (Model may have unloaded)")
            
        print("="*40)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    sample_text = """
    Large language models generate text token by token. A token can be a full word, a syllable, 
    or just a single character. By tracking the time it takes to generate the first token, and 
    the speed at which subsequent tokens are generated, developers can accurately gauge the 
    performance of their AI hardware.
    """
    
    summarize_with_accurate_metrics(sample_text)