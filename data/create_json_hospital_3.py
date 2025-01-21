import json

# Define the path to the input JSONL file and the output file
input_file_path = '/data/lishichao/project/Foundation-Medical/data/v2/medical_json_label.jsonl'
output_file_path = '/data/lishichao/project/Foundation-Medical/data/v2/thyroid_3_label.jsonl'


# Read the JSONL file, modify paths, and save to a new JSONL file
with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
    for line in infile:
        # Parse each line as JSON
        record = json.loads(line)
        
        # Extract only the filename from the image path
        filename = record['image'].split('/')[-1]
        
        # Update the record's image path to only contain the filename
        record['image'] = filename
        
        # Write the modified record to the output file
        outfile.write(json.dumps(record) + '\n')

print("Image paths have been updated to filenames and saved to", output_file_path)