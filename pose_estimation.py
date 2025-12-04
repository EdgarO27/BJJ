from io import StringIO
import os

import numpy as np
import csv

import pandas as pd



def read_pose_files(root_dir):
    # 1. Collect all CSV file paths recursively
    file_paths = []
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith(".csv"):
                file_paths.append(os.path.join(root, filename))
    
    print(f"Found {len(file_paths)} CSV files.")

    # 2. Read all CSVs into a single DataFrame
    # We verify that columns are numeric to avoid header errors
    df_list = []
    for f in file_paths:
        try:
            # header=None assumes no text header. If you have headers, change to header=0
            temp_df = pd.read_csv(f, header=None) 
            
            # optional: Verify column count matches 102
            if temp_df.shape[1] != 102:
                # If you have an index column, you might need: temp_df = temp_df.iloc[:, 1:]
                print(f"Warning: File {os.path.basename(f)} has {temp_df.shape[1]} columns. Expected 102.")
                continue
                
            df_list.append(temp_df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not df_list:
        return np.array([])

    # Concatenate all data
    full_data = pd.concat(df_list, ignore_index=True)
    
    # 3. Prepare for Reshaping
    # We need the total rows to be perfectly divisible by 30
    raw_array = full_data.values 
    
    num_sequences = len(raw_array) // 30
    cutoff = num_sequences * 30
    
    # Trim extra rows that don't make a full sequence of 30
    trimmed_data = raw_array[:cutoff]
    
    # 4. Reshape: (Batch_Size, Time_Steps, Features)
    # Dimensions: (N, 30, 102)
    final_array = trimmed_data.reshape((num_sequences, 30, 102))
    
    print(f"Processed shape: {final_array.shape}")
    return final_array        

# Convert the list of lists to a NumPy array
# Single_leg_data = np.array(seq, dtype=float)
Single_leg_data2 = read_pose_files(r'C:\Projects\AI\Image_class_bjj\Dataset_Pose\Single_Leg_pose')
Not_Single_leg_data2 = read_pose_files(r'C:\Projects\AI\Image_class_bjj\Dataset_pose\Not_Single_Leg_pose')

# Print the resulting NumPy array
print(Single_leg_data2.shape)
print(Single_leg_data2.dtype)
# print(Single_leg_data)


print(Not_Single_leg_data2.shape)
print(Not_Single_leg_data2.dtype)
# print(Not_Single_leg_data)

np.save("Single_leg",np.array(Single_leg_data2))

np.save("Not_Single_leg",np.array(Not_Single_leg_data2))

