import os
import yaml
import time
import re
from openai import OpenAI
import argparse
import requests
import configparser
import json
from pathlib import Path

# Function to read configurations from config.cfg
def read_config(file_path):
    config = configparser.ConfigParser(interpolation=None)
    config.read(file_path)
    return config

# Function to find an existing assistant by name on the OpenAI platform.
def find_assistant_by_name(client, name):
    response = client.beta.assistants.list()
    for assistant in response.data:
        if assistant.name == name:
            return assistant
    return None

# Function to create a new biocurator assistant on OpenAI with specified parameters.
def create_biocurator_assistant(client, model, name='Biocurator', description='Assistant for Biocuration', tools_path=None, instructions=''):
    # Load custom tools from the JSON file if provided
    tools = []
    if tools_path:
        with open(tools_path, 'r') as file:
            tools = json.load(file)

    # Add the default 'file_search' tool to the tools list
    tools.append({"type": "file_search"})

    assistant = client.beta.assistants.create(
        name=name,
        description=description,
        model=model,
        tools=tools,
        instructions=instructions
    )
    return assistant

# Function to modify the assistant to include the vector store ID
def modify_assistant_vector_store(client, assistant_id, vector_store_id):
    response = client.beta.assistants.update(
        assistant_id=assistant_id,
        tool_resources={
            "file_search": {
                "vector_store_ids": [vector_store_id]
            }
        }
    )
    return response

def get_or_create_biocurator_assistant(client, config, functions_json_path, vector_store_id):
    existing_biocurator = find_assistant_by_name(client, 'Biocurator')

    if existing_biocurator:
        assistant_id = existing_biocurator.id
        print(f"Found existing Biocurator assistant with ID: {assistant_id}", flush=True)
        # Modify the existing assistant to include the vector store ID
        modify_assistant_vector_store(client, assistant_id, vector_store_id)
        return assistant_id
    else:
        biocurator = create_biocurator_assistant(client, config['DEFAULT']['model'], 'Biocurator', 'Assistant for Biocuration', tools_path=functions_json_path, instructions=config['DEFAULT']['assistant_instructions'])
        if biocurator and biocurator.id:
            assistant_id = biocurator.id
            print(f"Created new Biocurator assistant with ID: {assistant_id}", flush=True)
            # Modify the new assistant to include the vector store ID
            modify_assistant_vector_store(client, assistant_id, vector_store_id)
            return assistant_id
        else:
            raise Exception("Assistant creation failed, no ID returned.")

# Function to create a vector store
def create_vector_store(client, name="My Vector Store"):
    response = client.beta.vector_stores.create(name=name)
    return response.id

# Function to delete a vector store
def delete_vector_store(client, vector_store_id):
    try:
        response = client.beta.vector_stores.delete(vector_store_id=vector_store_id)
        return response
    except Exception as e:
        print(f"Error: {e}", flush=True)

# Function to upload a file to OpenAI and attach it to the assistant for processing.
def upload_and_attach_file(client, file_path, vector_store_id):
    # Uploading the file to OpenAI with the purpose set to 'assistants'
    print(f"Uploading file: {file_path}", flush=True)
    uploaded_file = client.files.create(file=Path(file_path), purpose='assistants')
    file_id = uploaded_file.id

    # Attaching the uploaded file to the vector store.
    print("Attaching file to vector store...", flush=True)
    vector_store_file = client.beta.vector_stores.files.create(
        vector_store_id=vector_store_id,
        file_id=file_id
    )
    return file_id, vector_store_file.id

def process_input_files(client, assistant_id, vector_store_id, input_files, config):
    for index, input_file in enumerate(input_files, start=1):
        print(f"Processing file: {input_file} ({index}/{len(input_files)})", flush=True)
        file_id, vector_store_file_id = upload_and_attach_file(client, input_file, vector_store_id)

        process_queries_with_biocurator(client, assistant_id, file_id, vector_store_file_id, config['DEFAULT']['prompts_yaml_file'], config['DEFAULT']['output_dir'], input_file, int(config['DEFAULT']['timeout_seconds']))

        # Delete the file after it has been processed
        print(f"Deleting file after processing: {file_id}", flush=True)
        delete_file_from_openai(client, vector_store_id, file_id)

def delete_file_from_openai(client, vector_store_id, file_id):
    try:
        response = client.beta.vector_stores.files.delete(vector_store_id=vector_store_id, file_id=file_id)
        return response
    except Exception as e:
        print(f"Error: {e}", flush=True)

def clean_json_response(data):
    if isinstance(data, dict):
        return {k: clean_json_response(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_json_response(element) for element in data]
    elif isinstance(data, str):
        cleaned_data = re.sub(r"\[\d+†evidence\]|【.*?†source】", "", data)
        return cleaned_data
    else:
        return data

def process_json_output(output):
    try:
        data = json.loads(output) if isinstance(output, str) else output
        cleaned_data = clean_json_response(data)
        return json.dumps(cleaned_data, indent=4, ensure_ascii=False)
    except json.JSONDecodeError:
        return output

def run_thread_return_last_message(client, thread_id, assistant_id, timeout_seconds):
    # Initiating the processing run.
    run = client.beta.threads.runs.create(thread_id=thread_id, assistant_id=assistant_id)
    run_id = run.id

    # Checking periodically if the run has completed.
    start_time = time.time()
    while True:
        current_time = time.time()
        if current_time - start_time > timeout_seconds:
            print(f"Timeout: The operation took longer than {timeout_seconds} seconds.")
            print("-----", flush=True)
            print(f"Diagnostic information:")
            print(f"Run Status: {run.status}")
            print(f"Run Cancelled At: {run.cancelled_at}")
            print(f"Run Completed At: {run.completed_at}")
            print(f"Run Failed At: {run.failed_at}")
            print(f"Run Last Error: {run.last_error}")
            print("Proceeding to the next prompt.", flush=True)
            print("-----", flush=True)
            break

        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id,
        )

        if run.status == "requires_action":  # Check if the run requires action
            # Parse structured response from the tool call
            structured_response = json.loads(
                run.required_action.submit_tool_outputs.tool_calls[0].function.arguments
            )
            return structured_response
        elif run.status == "completed":
            print("Run completed.", flush=True)
            # Get the messages from the thread
            print("Retrieving messages from the thread...", flush=True)
            messages = client.beta.threads.messages.list(thread_id=thread_id).data
            print("Messages retrieved.", flush=True)
            # Return the content of the second message if it exists
            if len(messages) > 1:
                second_message = messages[0]
                if hasattr(second_message, 'content'):
                    return second_message.content[0].text.value
        elif run.status in ['failed', 'cancelled', 'expired']:
            if run.status == 'failed' and run.last_error is not None:
                print(f"Run failed: {run.last_error}", flush=True)
                raise Exception("Run failed.")
            elif run.status == 'cancelled':
                print("Run was cancelled.", flush=True)
            elif run.status == 'expired':
                print("Run expired.", flush=True)
            print("Proceeding to the next prompt.", flush=True)
            break

        time.sleep(5)  # Wait for 5 seconds before retrying

    raise Exception("Run did not complete in the expected manner.")


    raise Exception("Run did not complete in the expected manner.")



# Function to process queries using the Biocurator assistant.
def process_queries_with_biocurator(client, assistant_id, file_id, assistant_file_id, yaml_file, output_dir, pdf_file, timeout_seconds):
    # Creating a new thread for processing.
    thread = client.beta.threads.create()
    thread_id = thread.id

    # Loading prompts from the YAML file.
    prompts = yaml.safe_load(open(yaml_file, 'r'))
    for prompt, value in prompts.items():
        print('Processing prompt: ' + prompt, flush=True)
        # Sending each prompt to the assistant for processing.
        thread_message = client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=value
        )

        # Running the thread and retrieving the last message.
        final_text_to_write = run_thread_return_last_message(client, thread_id, assistant_id, timeout_seconds)

        # Cleaning and saving the processed content to an output file.
        output_file = Path(output_dir) / (pdf_file.stem + '_' + prompt + '.txt')
        with open(output_file, 'w') as f:
            cleaned_text = process_json_output(final_text_to_write)
            f.write(cleaned_text)

def cleanup_resources(client, vector_store_id, assistant_id=None):
    if assistant_id:
        try:
            print(f"Deleting assistant: {assistant_id}", flush=True)
            client.beta.assistants.delete(assistant_id)
        except Exception as e:
            print(f"Failed to delete assistant with ID {assistant_id}: {e}", flush=True)

    try:
        print(f"Deleting vector store: {vector_store_id}", flush=True)
        delete_vector_store(client, vector_store_id)
    except Exception as e:
        print(f"Failed to delete vector store with ID {vector_store_id}: {e}", flush=True)

def main():
    vector_store_id = None
    assistant_id = None

    try:
        # Read configurations from the config.cfg file
        config = read_config('config.cfg')

        # Parsing command-line argument for the API key
        parser = argparse.ArgumentParser(description='Process files with OpenAI GPT-4 and Biocurator Assistant')
        parser.add_argument('--api_key', help='OpenAI API key', required=True)
        args = parser.parse_args()

        # Initializing the OpenAI client with the API key from the command line
        client = OpenAI(api_key=args.api_key)

        # Create a new vector store
        vector_store_id = create_vector_store(client)

        functions_json_path = 'functions.json'

        assistant_id = get_or_create_biocurator_assistant(client, config, functions_json_path, vector_store_id)

        # Get the list of files to process
        input_dir = Path(config['DEFAULT']['input_dir'])
        input_files = sorted([f for f in input_dir.glob('*.pdf') if f.name != '.gitignore'])

        # Start the processing timer
        start_time = time.time()

        # Process each file
        process_input_files(client, assistant_id, vector_store_id, input_files, config)

        # Calculate and print the total time elapsed
        end_time = time.time()
        total_time_elapsed = end_time - start_time
        print(f"Total time elapsed: {total_time_elapsed:.2f} seconds")

        # Calculate the average time per file
        if len(input_files) > 0:
            average_time_per_file = total_time_elapsed / len(input_files)
            print(f"Average time per input file: {average_time_per_file:.2f} seconds")
        else:
            print("No files were processed.")

    except Exception as e:
        print(f"An error occurred: {e}", flush=True)

    finally:
        cleanup_resources(client, vector_store_id, assistant_id)

if __name__ == '__main__':
    main()