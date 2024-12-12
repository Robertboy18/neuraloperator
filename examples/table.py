import os
import pandas as pd
import numpy as np

# Define the file pattern
file_pattern = "darcy-full_size_matrixgalore_r{rank}_cplx_rollup:{dim}"
# Initialize an empty list to store the data
data = []

# Load baseline data first
file_pattern2 = "darcy-full_size"
file_path = os.path.join("/global/homes/r/rgeorge/repo/tensorgalore/memstats_darcy/", file_pattern2 + ".txt")
if os.path.exists(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    # Parse the memory usage information
    baseline_memory = {}
    for line in lines:
        if "Max" in line:
            parts = line.strip().split(": ")
            key = parts[0].split(" ")[1]
            value = float(parts[1].split(" ")[0])
            baseline_memory[key] = value
    baseline_memory["Baseline"] = 1.0
    data.append(baseline_memory)

# Function to process memory usage data
def process_memory_usage(file_path, rank_key, rank_value):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            lines = file.readlines()
        memory_usage = {}
        for line in lines:
            if "Max" in line:
                parts = line.strip().split(": ")
                key = parts[0].split(" ")[1]
                value = float(parts[1].split(" ")[0])
                memory_usage[key] = value
        
        # Adjust memory usage
        memory_usage['Intermediate_Activation'] = memory_usage.get('None', 0)
        memory_usage['Gradient_Intermediate'] = memory_usage.get('None', 0) - baseline_memory.get('None', 0)
        memory_usage['Activation'] += memory_usage['Intermediate_Activation']
        memory_usage['Gradients'] += memory_usage['Gradient_Intermediate']
        memory_usage.pop('None', None)
        memory_usage[rank_key] = rank_value
        
        return memory_usage
    return None

# Iterate over the ranks and load the files for matrix galore
for dim in [1, 2, 3]:
    for rank in [8, 16, 32, 64, 128]:
        file_name = file_pattern.format(rank=rank, dim=dim)
        file_path = os.path.join("/global/homes/r/rgeorge/repo/tensorgalore/memstats_darcy/", file_name + ".txt")
        memory_usage = process_memory_usage(file_path, f"Matrix_galore_Rank_{dim}", rank)
        if memory_usage:
            data.append(memory_usage)

# Load tensor galore data
file_pattern1 = "darcy-full_size_tensorgalore_r{rank}"
for rank in [0.01, 0.1, 0.25, 0.5, 0.75, 1.0]:
    file_name = file_pattern1.format(rank=rank)
    file_path = os.path.join("/global/homes/r/rgeorge/repo/tensorgalore/memstats_darcy/", file_name + ".txt")
    memory_usage = process_memory_usage(file_path, "Rank_tensor_galore", rank)
    if memory_usage:
        data.append(memory_usage)

# Create a pandas DataFrame from the data
df = pd.DataFrame(data)

# Calculate the total memory usage
columns_to_sum = ['Parameters', 'Gradients', 'Forward', 'Activation']
df['Total_Memory_Usage'] = df[columns_to_sum].sum(axis=1)

# Reorder columns
rank_columns = [col for col in df.columns if 'Rank' in col or 'Baseline' in col]
other_columns = [col for col in df.columns if col not in rank_columns and col != 'Total_Memory_Usage']
new_column_order = rank_columns + ['Total_Memory_Usage'] + other_columns

df = df[new_column_order]

# Display the table
print(df)

# Optionally, save to CSV
# df.to_csv('memory_usage_summary.csv', index=False)