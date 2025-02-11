import pandas as pd
import re

def parse_papers_by_double_empty_lines(lines):
    papers = []  # List to store individual papers
    buffer = []  # Temporary buffer to collect lines of the current paper

    for line in lines:
        # Detect a double empty line as the separator between papers
        if line.strip() == "" and buffer and buffer[-1].strip() == "":
            papers.append(buffer)  # Save the current paper
            buffer = []  # Reset for the next paper
        else:
            buffer.append(line)

    # Add the last paper if file ends without a double empty line
    if buffer and not all(l.strip() == "" for l in buffer):
        papers.append(buffer)

    return papers

def process_sections(paper):
    # Define the section names
    section_names = ["publishing_info", "title", "authors", "affiliations", "abstract", "extra"]
    sections = {}
    current_section = []
    section_count = 0
    extra_lines = []  # Buffer to store all extra lines

    sections["language"] = None  # Default language is None

    for line in paper:
        if line.strip() == "":  # Detect empty line as section boundary
            if current_section:
                # Check if the current section is a language entry
                language_match = re.match(r'^\[Article in .*?\]$', current_section[0].strip())
                if language_match:
                    sections["language"] = current_section[0].strip()
                else:
                    # Determine section name
                    if section_count < len(section_names):
                        section_name = section_names[section_count]
                        sections[section_name] = "\n".join(current_section).replace("\n", "")
                    else:
                        #get all the remainin content of the paper
                        remaining = "\n".join(current_section).replace("\n", " ")
                        #add it to the existing extra lines
                        sections["extra"] = sections.get("extra", "") + remaining
                    section_count += 1
                current_section = []
                #section_count += 1
        else:
            current_section.append(line)

    return sections

def process_file(file_path):
    # Read the file and split into lines
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Parse papers using double empty lines
    papers = parse_papers_by_double_empty_lines(lines)

    # List to store extracted information
    extracted_data = []

    # Process each paper into structured sections
    for paper in papers:
        sections = process_sections(paper)
        # Extract the information from the sections
        language = sections.get("language")
        publishing_info = sections.get("publishing_info").split(".")
        journal = publishing_info[1].strip() if len(publishing_info) > 1 else None
        year = publishing_info[2].strip().split(" ")[0] if len(publishing_info) > 2 else None
        doi_match = re.search(r'doi:\s*(10\.\S+)', sections.get("publishing_info", ""))
        doi = doi_match.group(1).rstrip(".") if doi_match else None
        title = sections.get("title")
        first_author = sections.get("authors", "").split(",")[0].split('(')[0].strip() if sections.get("authors") else None
        abstract = sections.get("abstract")

        # Append the extracted information to the list
        extracted_data.append({
            "DOI": doi,
            "Journal": journal,
            "Year": year,
            "Title": title,
            "First Author": first_author,
            "Abstract": abstract,
            "Language": language
        })

    # Convert the list to a DataFrame
    df = pd.DataFrame(extracted_data)
    return df

def process_wos(file_path):
    """
    Process the Web of Science saved records file and extract metadata for each paper.

    Args:
        file_path (str): Path to the "savedrecs.txt" file.

    Returns:
        list[dict]: A list of dictionaries containing extracted metadata for each paper.
    """
    extracted_data = []

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split content into individual papers using "ER\n" as the delimiter
    papers = content.split("\nER\n")

    for paper in papers:
        if not paper.strip():
            continue

        # Extract metadata using regex patterns
        doi = re.search(r'\nDI (.+)', paper)
        journal = re.search(r'\nSO (.+)', paper)
        year = re.search(r'\nPY (\d+)', paper)
        title = re.search(r'\nTI (.+(?:\n   .+)*)', paper)
        first_author = re.search(r'\nAU (.+)', paper)
        abstract = re.search(r'\nAB (.+(?:\n   .+)*)', paper)
        language = re.search(r'\nLA (.+)', paper)
        openaccess = re.search(r'\nOA (.+)', paper)
        document_type = re.search(r'\nDT (.+)', paper)
        keywords = re.search(r'\nDE (.+(?:\n   .+)*)', paper)

        # Clean up multiline fields
        title_text = title.group(1).replace("\n   ", " ") if title else None
        abstract_text = abstract.group(1).replace("\n   ", " ") if abstract else None

        # If doi is missing, discard the paper
        if not doi:
            continue
        
        # Append extracted data
        extracted_data.append({
            "DOI": doi.group(1) if doi else None,
            "Journal": journal.group(1) if journal else None,
            "Year": year.group(1) if year else None,
            "Title": title_text,
            "First Author": first_author.group(1) if first_author else None,
            "Abstract": abstract_text,
            "Language": language.group(1) if language else None,
            "Open Access": openaccess.group(1) if openaccess else None,
            "Document Type": document_type.group(1) if document_type else None,
            "Keywords": keywords.group(1) if keywords else None
        })
        
        

    return extracted_data

# Path to the input file
file_path = "abstract-maldi-tofT-set.txt"

# Process the file and get the DataFrame
pubmed_df = process_file(file_path)

# Set DOI as the index of the DataFrame
pubmed_df.set_index("DOI", inplace=True)

# Save the DataFrame to a CSV file
pubmed_df.to_csv("pubmed_data.csv")


file_path = "savedrecs.txt"

data = process_wos(file_path)

df = pd.DataFrame(data)

# Set DOI as the index of the DataFrame
df.set_index("DOI", inplace=True)

df.to_csv("wos_data.csv")


