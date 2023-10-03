import re

def extract_sections(file_path):
    # Sections we care about
    section_titles = [
        "Chief Complaint",
        "Discharge Diagnosis",
        "History of Present Illness",
        "Past Medical History"
    ]
    
    # Read the file
    with open(file_path, 'r') as f:
        text = f.read()
    
    # Initialize a dictionary to store section texts
    section_texts = {title: "" for title in section_titles}
    
    # Regex to match any of the section titles followed by colon and any text until two new lines
    for title in section_titles:
        pattern = re.compile(rf"{title}:(.*?)(?:\n{{2,}}|$)", re.DOTALL)
        match = pattern.search(text)
        if match:
            # Store the matched text without leading/trailing white spaces
            section_texts[title] = match.group(1).strip()
    
    return section_texts

# Example usage:
file_path = "../training_20180910/100035.txt"
sections = extract_sections(file_path)

# Example print:
for title, text in sections.items():
    print(f"\n{title}:\n{'-'*len(title)}\n{text}\n")
