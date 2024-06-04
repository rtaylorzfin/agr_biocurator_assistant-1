import argparse
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Define file paths
OBO_FILE_PATH = 'doid.obo'
EMBEDDINGS_FILE_PATH = 'embeddings.json'

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

def parse_obo(file_path):
    print("Parsing OBO file...")
    with open(file_path, 'r') as file:
        terms = []
        current_term = None
        for line in file:
            line = line.strip()
            if line == "[Term]":
                if current_term:
                    terms.append(current_term)
                current_term = {}
            elif line.startswith("id:"):
                current_term['id'] = line.split("id: ")[1]
            elif line.startswith("name:"):
                current_term['name'] = line.split("name: ")[1]
            elif line.startswith("def:"):
                current_term['definition'] = line.split("def: ")[1].split(' [')[0].strip('"')
            elif line.startswith("synonym:"):
                if 'synonyms' not in current_term:
                    current_term['synonyms'] = []
                current_term['synonyms'].append(line.split("synonym: ")[1].split(' [')[0].strip('"'))
        if current_term:
            terms.append(current_term)
    return terms

obo_terms = parse_obo(OBO_FILE_PATH)

# Generate and save embeddings
print("Generating embeddings...")
embeddings = {}
for term in tqdm(obo_terms):
    term_id = term['id']
    name_vector = model.encode(term.get('name', ''))
    definition_vector = model.encode(term.get('definition', ''))
    synonyms_vector = model.encode(' '.join(term.get('synonyms', [])))
    embeddings[term_id] = {
        "nameVector": name_vector.tolist(),
        "definitionVector": definition_vector.tolist(),
        "synonymsVector": synonyms_vector.tolist()
    }

print(f"Saving embeddings to {EMBEDDINGS_FILE_PATH}...")
with open(EMBEDDINGS_FILE_PATH, 'w') as f:
    json.dump(embeddings, f)

print("Embeddings generation complete.")
