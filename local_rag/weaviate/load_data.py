import argparse
import weaviate
from tqdm import tqdm

# Define file paths
OBO_FILE_PATH = 'doid.obo'

# Parse command line arguments
parser = argparse.ArgumentParser(description='Load OBO data into Weaviate and vectorize with GPT-4all.')
parser.add_argument('--weaviate_url', type=str, default='http://localhost:8090', help='Weaviate instance URL')
parser.add_argument('--reset', action='store_true', help='Flag to reset the database')
args = parser.parse_args()

# Connect to Weaviate
client = weaviate.Client(args.weaviate_url)

def reset_database(client):
    print("Resetting database...")
    schema = client.schema.get()
    for cls in schema['classes']:
        client.schema.delete_class(cls["class"])

# Reset the database if the flag is set
if args.reset:
    reset_database(client)

# Create schema for disease terms with vectorization
print("Creating schema in Weaviate...")
class_obj = {
    "class": "DiseaseTerm",
    "vectorizer": "text2vec-gpt4all",  # Specify the GPT-4all vectorizer
    "properties": [
        {
            "name": "name",
            "dataType": ["text"]
        },
        {
            "name": "definition",
            "dataType": ["text"]
        },
        {
            "name": "synonyms",
            "dataType": ["text[]"]
        }
    ],
    "moduleConfig": {
        "text2vec-gpt4all": {
            "vectorizeClassName": False
        }
    }
}

client.schema.create_class(class_obj)

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

# Add OBO terms to Weaviate
print("Adding OBO terms to Weaviate...")

for term in tqdm(obo_terms):
    data_object = {
        "name": term.get('name', ''),
        "definition": term.get('definition', ''),
        "synonyms": term.get('synonyms', [])
    }
    client.data_object.create(
        class_name="DiseaseTerm",
        data_object=data_object
    )

print("Data loading complete.")
