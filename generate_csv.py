import os
import csv

# Path to your circuit files folder
folder_path = "Circuits_folder"
output_csv = "samples.csv"

# Open the CSV file for writing
with open(output_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    # Write the header required by your dataset loader
    writer.writerow(["circuit_path", "function_id", "split", "role"])
    
    # Loop through all files in the directory
    for fname in os.listdir(folder_path):
        if fname.endswith(".bench"):
            # Construct the relative file path
            circuit_path = os.path.join(folder_path, fname)
            
            # Extract a function_id from the filename 
            # (e.g., if files are named 'adder_v1.bench', you might parse 'adder')
            # Adjust this logic depending on how your filenames are structured:
            function_id = fname.split("_")[0] 
            
            split = "train"
            role = "view"
            
            # Write the row to the CSV
            writer.writerow([circuit_path, function_id, split, role])

print(f"Successfully generated {output_csv}!")