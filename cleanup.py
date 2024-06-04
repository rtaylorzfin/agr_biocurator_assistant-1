import os
import argparse
from openai import OpenAI

# Configurable variable for maximum retries
MAX_RETRIES = 6

def retry_deletion(delete_function, check_function, client, item_name, max_retries=MAX_RETRIES):
    retries = 0
    while retries < max_retries:
        delete_function(client)
        if check_function(client):
            print(f"All {item_name} deleted successfully.")
            return
        retries += 1
        print(f"Retrying to delete {item_name}. Attempt {retries} of {max_retries}.")
    print(f"Failed to delete {item_name} after {max_retries} attempts.")

def delete_assistants(client):
    response = client.beta.assistants.list()
    for assistant in response.data:
        if 'Biocurator' in assistant.name:
            try:
                print(f"Deleting assistant: {assistant.id}")
                client.beta.assistants.delete(assistant.id)
            except Exception as e:
                print(f"Failed to delete assistant {assistant.id}: {e}")

def check_assistants_deleted(client):
    response = client.beta.assistants.list()
    return all('Biocurator' not in assistant.name for assistant in response.data)

def delete_files(client):
    response = client.files.list()
    for file in response.data:
        try:
            print(f"Deleting file: {file.id}")
            client.files.delete(file.id)
        except Exception as e:
            print(f"Failed to delete file {file.id}: {e}")

def check_files_deleted(client):
    response = client.files.list()
    return len(response.data) == 0

def delete_vector_stores(client):
    limit = 100
    after = None

    while True:
        if after:
            response = client.beta.vector_stores.list(limit=limit, after=after)
        else:
            response = client.beta.vector_stores.list(limit=limit)
        
        if not response.data:
            break

        for vs in response.data:
            try:
                print(f"Deleting vector store: {vs.id}")
                client.beta.vector_stores.delete(vector_store_id=vs.id)
            except Exception as e:
                print(f"Failed to delete vector store {vs.id}: {e}")

        if not response.has_more:
            break
        after = response.data[-1].id

def check_vector_stores_deleted(client):
    response = client.beta.vector_stores.list(limit=1)
    return len(response.data) == 0

def main():
    parser = argparse.ArgumentParser(description='Cleanup OpenAI objects')
    parser.add_argument('--api_key', help='OpenAI API key', required=True)
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key)

    try:
        retry_deletion(delete_assistants, check_assistants_deleted, client, "Biocurator assistants")
        retry_deletion(delete_files, check_files_deleted, client, "files")
        retry_deletion(delete_vector_stores, check_vector_stores_deleted, client, "vector stores")
    except Exception as e:
        print(f"An error occurred during cleanup: {e}")

if __name__ == '__main__':
    main()
