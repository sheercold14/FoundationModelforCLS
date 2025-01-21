# split_json = '/data/lishichao/project/Foundation-Medical/data/v2/thyroid_3_quality_1.jsonl'
# request_json = '/data/lishichao/project/LLaVA-Med/qwen/result/Thyroid_hospital_3_202411.json'
# output_file =  '/data/lishichao/project/LLaVA-Med/qwen/result/thyroid_3_quality_1.json'
# import json
# import os 
# # Open the output file in append mode
# # with open(output_file, "w") as outfile:
# #     outfile.write("[\n")  # Start of the JSON array

# with open(split_json, "r") as split_file:
#     split_data = [json.loads(line) for line in split_file]
    
# request_dict = [q for q in json.load(open(os.path.expanduser(request_json)))] 

import json
import os

# Define file paths
split_json = '/data/lishichao/project/Foundation-Medical/data/v2/thyroid_3_quality_2.jsonl'
request_json = '/data/lishichao/project/LLaVA-Med/qwen/result/Thyroid_hospital_3_202411.json'
request_json_2 = '/data/lishichao/project/LLaVA-Med/qwen/result/results_hospital_3.json'
output_file = '/data/lishichao/project/LLaVA-Med/qwen/result/thyroid_3_quality_2.json'

# Load split_data from JSONL file
with open(split_json, "r") as split_file:
    split_data = [json.loads(line) for line in split_file]

# Load request_dict from JSON files
with open(request_json, "r") as request_file:
    request_dict = json.load(request_file)

with open(request_json_2, "r") as request_file_2:
    request_dict_2 = json.load(request_file_2)

# Convert request_dict and request_dict_2 to dictionaries indexed by the image key for faster lookup
request_dict_indexed = {item['image']: item for item in request_dict}
request_dict_2_indexed = {item['image']: item for item in request_dict_2}

# Iterate through each item in split_data and update with answers from request_dict or request_dict_2
for item in split_data:
    image_key = item['image']
    
    # First, check in request_dict
    if image_key in request_dict_indexed:
        item['answers'] = request_dict_indexed[image_key].get('answers', [])
    
    # If not found, check in request_dict_2
    elif image_key in request_dict_2_indexed:
        item['answers'] = request_dict_2_indexed[image_key].get('answers', [])

# Save the updated split_data to output_file
with open(output_file, "w") as outfile:
    json.dump(split_data, outfile, indent=4)

print(f"Data has been successfully updated and saved to {output_file}")
