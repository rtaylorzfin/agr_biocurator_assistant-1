import os
import argparse
from openai import OpenAI

def delete_assistants(client):
    response = client.beta.assistants.list()
    for assistant in response.data:
        if 'Biocurator' in assistant.name:
            try:
                print(f"Deleting assistant: {assistant.id}")
                client.beta.assistants.delete(assistant.id)
            except Exception as e:
                print(f"Failed to delete assistant {assistant.id}: {e}")

def delete_files(client):
    response = client.files.list()
    for file in response.data:
        try:
            print(f"Deleting file: {file.id}")
            client.files.delete(file.id)
        except Exception as e:
            print(f"Failed to delete file {file.id}: {e}")

def delete_vector_stores(client):
    limit = 100
    after = None
    error_vector_stores = []

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
                error_vector_stores.append(vs.id)

        if not response.has_more:
            break
        after = response.data[-1].id
    
    return error_vector_stores

def retry_delete_vector_stores(client, max_retries=6):
    retries = 0
    while retries < max_retries:
        error_vector_stores = delete_vector_stores(client)
        if not error_vector_stores:
            print("All vector stores deleted successfully.")
            return
        print(f"Retrying to delete vector stores. Attempt {retries + 1} of {max_retries}.")
        retries += 1
        if retries >= max_retries:
            print(f"Failed to delete the following vector stores after {max_retries} attempts: {error_vector_stores}")
            break

def main():
    parser = argparse.ArgumentParser(description='Cleanup OpenAI objects')
    parser.add_argument('--api_key', help='OpenAI API key', required=True)
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key)

    try:
        delete_assistants(client)
        delete_files(client)
        retry_delete_vector_stores(client)
    except Exception as e:
        print(f"An error occurred during cleanup: {e}")

if __name__ == '__main__':
    main()
