import asyncio
import aiohttp
import json

# Configuration
NODES = [
    "http://localhost:11434",  # Node 1 (Mapped by Docker proxy if ports are exposed)
    # Alternatively, if running the script inside the Docker network, 
    # you can discover containers by their specific container IPs or service names.
]
MODEL_NAME = "qwen2.5:0.5b"
CHUNK_SIZE = 2000  # Characters per chunk
OVERLAP = 200      # Overlap between chunks

def chunk_text(text, chunk_size, overlap):
    """Splits text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

async def summarize_chunk(session, node_url, chunk, chunk_id):
    """Sends a single chunk to a specific container node for summarization."""
    url = f"{node_url}/api/generate"
    payload = {
        "model": MODEL_NAME,
        "prompt": f"Summarize the following text concisely:\n\n{chunk}",
        "stream": False
    }
    
    print(f"[Driver] Sending Chunk {chunk_id} to node {node_url}...")
    try:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                print(f"[Driver] Received summary for Chunk {chunk_id}.")
                return result.get("response", "")
            else:
                print(f"Error from node {node_url}: Status {response.status}")
                return ""
    except Exception as e:
        print(f"Failed to connect to node {node_url}: {e}")
        return ""

async def main_driver(large_text):
    # 1. Discover active container endpoints
    # For a local docker compose with exposed random ports, replace NODES with actual host ports
    # e.g., using `docker port` commands to map them dynamically.
    active_nodes = NODES 
    
    # 2. Split text into chunks
    chunks = chunk_text(large_text, CHUNK_SIZE, OVERLAP)
    print(f"[Driver] Split text into {len(chunks)} chunks across {len(active_nodes)} nodes.")
    
    # 3. Map Phase: Distribute chunks to containers concurrently
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, chunk in enumerate(chunks):
            # Round-robin distribution among available nodes
            node_url = active_nodes[i % len(active_nodes)]
            tasks.append(summarize_chunk(session, node_url, chunk, i))
            
        intermediate_summaries = await asyncio.gather(*tasks)
    
    # 4. Reduce Phase: Combine intermediate results
    combined_intermediate_text = "\n\n".join(filter(None, intermediate_summaries))
    print("[Driver] Map phase complete. Combining summaries into final output...")
    
    # Send the combined text to a final node to produce a coherent summary
    async with aiohttp.ClientSession() as session:
        final_summary = await summarize_chunk(
            session, 
            active_nodes[0], 
            f"Synthesize these partial summaries into one fluid narrative:\n\n{combined_intermediate_text}", 
            "FINAL"
        )
        
    print("\n=== FINAL COMBINED SUMMARY ===")
    print(final_summary)

# Example execution
if __name__ == "__main__":
    sample_large_text = "Your long text goes here..." * 50
    asyncio.run(main_driver(sample_large_text))