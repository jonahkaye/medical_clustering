import re
import pandas as pd
import os

def is_valid_sentence(sentence):
    """Helper function to check if a sentence is valid (not too short and not just punctuation)."""
    return len(sentence) > 1 and not all(char in '.#,' for char in sentence)

def extract_sections(file_path):
    # List of section titles we care about
    section_titles = [
        "Allergies:",
        "Chief Complaint:",
        "Major Surgical or Invasive Procedure:",
        "History of Present Illness:",
        "Review of Systems:",
        "Past Medical History:",
        "Social History:",
        "Family History:",
        "Physical Exam:",
        "Pertinent Results:",
        "Brief Hospital Course:",
        "Medications on Admission:",
        "Discharge Medications:",
        "Discharge Disposition:",
        "Discharge Diagnosis:",
        "Discharge Condition:"
    ]
    
    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Initialize a dictionary to store section texts
    section_texts = {title: "" for title in section_titles}

    current_section = None
    for line in lines:
        line_stripped = line.strip()
        # Check if the line matches any section title
        if line_stripped in section_titles:
            current_section = line_stripped
            continue  # skip the title line itself
        # If inside a section, append the line to the section's text
        elif current_section:
            section_texts[current_section] += line
    
    # Removing excess newlines for each section
    for key, val in section_texts.items():
        section_texts[key] = val.strip()
    
    return section_texts

def create_dataframe_from_sections(file_path):
    # Extract sections from the file
    sections = extract_sections(file_path)
    
    # Keys we are interested in for regular sentence splits
    target_keys = [
        "Major Surgical or Invasive Procedure:",
        "History of Present Illness:",
        "Past Medical History:",
        "Brief Hospital Course:"
    ]

    # Keys we are interested in for comma and newline splits
    special_keys = [
        "Chief Complaint:",
        "Discharge Diagnosis:"
    ]
    
    # Data to populate the dataframe
    data = {
        'sentence': [],
        'section': [],
        'file': [],
        'type': []
    }

    # Helper function to split text into sentences
    def split_into_sentences(text):
        # Split on periods, question marks
        return re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s*', text)

    # For regular keys
    for key in target_keys:
        sentences = split_into_sentences(sections[key])
        for sentence in sentences:
            cleaned_sentence = sentence.replace("\n", " ").strip()
            if cleaned_sentence and is_valid_sentence(cleaned_sentence):  # Modified condition
                data['sentence'].append(cleaned_sentence)
                data['section'].append(key)
                data['file'].append(file_path)
                data['type'].append("underlying factor")

    # For special keys, split by comma and newline
    for key in special_keys:
        sentences = re.split(r'[\n,]', sections[key])
        for sentence in sentences:
            cleaned_sentence = sentence.replace("\n", " ").strip()
            if cleaned_sentence and is_valid_sentence(cleaned_sentence):  # Modified condition
                data['sentence'].append(cleaned_sentence)
                data['section'].append(key)
                data['file'].append(file_path)
                data['type'].append("condition")
    
    # Convert data into dataframe
    df = pd.DataFrame(data)
    return df

def process_all_files_in_directory(directory_path):
    # List all files in the given directory
    all_files = os.listdir(directory_path)

    # Filter out files that aren't text files
    text_files = [file for file in all_files if file.endswith('.txt')]

    # Initialize an empty dataframe to store data from all files
    all_data_df = pd.DataFrame(columns=['sentence', 'section', 'file', 'type'])

    # Iterate through each text file and process it
    for text_file in text_files:
        file_path = os.path.join(directory_path, text_file)
        df = create_dataframe_from_sections(file_path)
        all_data_df = pd.concat([all_data_df, df], ignore_index=True)

    return all_data_df

# Example usage:
directory_path = "../training_20180910"  # Directory containing text files
all_df = process_all_files_in_directory(directory_path)
print(all_df)

# split:
# HPI, PMH, Hospital Course, Major Surgical l


