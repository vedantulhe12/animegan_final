import os
import glob

# Directory containing the files (modify if needed)
directory = "./"

# Pattern to match files like 'boundary_1 (4).npy', 'boundary_1 (x).npy', etc.
file_pattern = os.path.join(directory, "boundary_1 (*.npy")

# Get list of matching files
files = glob.glob(file_pattern)

# Sort files to ensure consistent numbering
files.sort()

# Rename files to 'boundary_x.npy' where x is 1 to n
for i, old_name in enumerate(files, start=1):
    new_name = os.path.join(directory, f"boundary_{i}.npy")
    os.rename(old_name, new_name)
    print(f"Renamed: {old_name} -> {new_name}")