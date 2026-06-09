import docker
import time
import csv
import os

def main():
    # Connects to your host computer's Docker daemon
    client = docker.from_env()
    
    # Ensure these exactly match your container names in docker-compose.yml
    target_containers = ['orchestrator_llm', 'simulated_phone_1', 'simulated_phone_2']
    
    output_file = "docker_telemetry.csv"
    file_exists = os.path.isfile(output_file)
    
    with open(output_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        # Write headers if it's a brand new file
        if not file_exists:
            writer.writerow(["Unix_Timestamp", "Readable_Time", "Container", "RAM_MB"])
            
        print("📊 Telemetry Monitor Started. Press Ctrl+C to stop.")
        print(f"Logging live RAM usage to: {output_file}")
        
        try:
            while True:
                current_unix = time.time()
                current_readable = time.strftime("%Y-%m-%d %H:%M:%S")
                
                for name in target_containers:
                    try:
                        container = client.containers.get(name)
                        
                        # stream=False grabs a 1-second snapshot of the hardware
                        stats = container.stats(stream=False)
                        
                        # Extract the active RAM usage in bytes and convert to Megabytes
                        ram_usage = stats['memory_stats'].get('usage', 0) / (1024 * 1024)
                        
                        writer.writerow([current_unix, current_readable, name, f"{ram_usage:.2f}"])
                    except docker.errors.NotFound:
                        pass # Silently skip if the container hasn't booted up yet
                    except Exception as e:
                        pass # Ignore temporary read errors during container restarts
                
                # Force the file to save immediately so data isn't lost if you close the terminal
                f.flush()
                
                # Wait 1 second before polling again
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Telemetry Monitor Stopped safely.")

if __name__ == "__main__":
    main()