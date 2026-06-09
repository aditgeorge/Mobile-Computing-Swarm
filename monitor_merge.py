import pandas as pd
import os

def main():
    eval_file = "outputs/evaluation_results.csv"
    telemetry_file = "outputs/docker_telemetry.csv"
    output_file = "outputs/final_merged_report.csv"

    print("Loading datasets...")
    try:
        # Load the CSVs into Pandas DataFrames
        eval_df = pd.read_csv(eval_file)
        tel_df = pd.read_csv(telemetry_file)
    except FileNotFoundError as e:
        print(f"🚨 File not found: {e}")
        print("Make sure you are running this from your project root directory.")
        return

    print(f"Processing {len(eval_df)} rows...")

    # Create new empty columns to hold our findings
    eval_df["Peak_RAM_Orchestrator_MB"] = 0.0
    eval_df["Peak_RAM_Phone1_MB"] = 0.0
    eval_df["Peak_RAM_Phone2_MB"] = 0.0

    # Loop through every row in your evaluation results
    for index, row in eval_df.iterrows():
        start_time = row['Row_Start_Unix']
        end_time = row['Row_End_Unix']

        # Slice the telemetry data to only include the time window for this specific row
        # We add a 1-second buffer to the end_time to catch trailing memory spikes
        window_mask = (tel_df['Unix_Timestamp'] >= start_time) & (tel_df['Unix_Timestamp'] <= (end_time + 1.0))
        window_data = tel_df[window_mask]

        if not window_data.empty:
            # Group the sliced data by container name and find the maximum RAM value for each
            max_ram_by_container = window_data.groupby('Container')['RAM_MB'].max()

            # Safely extract and assign the peaks to our evaluation dataframe
            if 'orchestrator_llm' in max_ram_by_container:
                eval_df.at[index, "Peak_RAM_Orchestrator_MB"] = max_ram_by_container['orchestrator_llm']
            
            if 'simulated_phone_1' in max_ram_by_container:
                eval_df.at[index, "Peak_RAM_Phone1_MB"] = max_ram_by_container['simulated_phone_1']
                
            if 'simulated_phone_2' in max_ram_by_container:
                eval_df.at[index, "Peak_RAM_Phone2_MB"] = max_ram_by_container['simulated_phone_2']

    # Drop the raw Unix timestamps to clean up the final report (optional, but makes it prettier in Excel)
    eval_df = eval_df.drop(columns=['Row_Start_Unix', 'Row_End_Unix'])

    # Save the master dataframe to a new CSV
    eval_df.to_csv(output_file, index=False)
    
    print("\n==========================================")
    print("           MERGE COMPLETE!                ")
    print(f" Master report saved to: {output_file}    ")
    print("==========================================")

if __name__ == "__main__":
    main()