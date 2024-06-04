import argparse
import pandas as pd
import weaviate
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.retrievers import WeaviateHybridSearchRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence

# Define file paths
TSV_FILE_PATH = 'WB disease data test 1 - WBPaper00006014.tsv'

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run a hybrid search with RAG model.')
parser.add_argument('--api_key', type=str, required=True, help='OpenAI API key')
parser.add_argument('--weaviate_url', type=str, default='http://localhost:8090', help='Weaviate instance URL')
args = parser.parse_args()

# Set OpenAI API key
openai_api_key = args.api_key

# Connect to Weaviate
client = weaviate.Client(
    url=args.weaviate_url,
    additional_headers={
        "X-Openai-Api-Key": openai_api_key,
    },
)

# Load TSV file and prepare for searching
print("Loading TSV file...")
tsv_data = pd.read_csv(TSV_FILE_PATH, sep='\t')

def prepare_tsv_data(tsv_data):
    print("Combining 'Disease' and 'Evidence' columns for TSV data...")
    tsv_data['combined_info'] = tsv_data.apply(lambda row: f"Disease: {row['Disease']}. Evidence: {row['Evidence']}.", axis=1)
    return tsv_data

tsv_data = prepare_tsv_data(tsv_data)

# Set up Weaviate Hybrid Search Retriever
retriever = WeaviateHybridSearchRetriever(
    client=client,
    index_name="DiseaseTerm",
    text_key="definition",
    attributes=["name", "definition", "synonyms"],
    create_schema_if_missing=True,
)

# Perform hybrid search in Weaviate and use OpenAI for RAG
def hybrid_search_and_rag(tsv_data, retriever, openai_api_key):
    print("Performing hybrid search and using OpenAI for RAG...")
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0)

    prompt_template = """You are an assistant for matching disease terms. Your task is to match the disease term and evidence information provided with the most appropriate term from the disease ontology context retrieved. 
    Use the following pieces of retrieved context to find the best match. Each context piece represents a disease description retrieved from the database. If you cannot find a match, state that you don't know. Keep your answer concise.
    
    Disease Term: {question}
    Context:
    {context}
    Best Match:
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    results = []

    for idx, row in tqdm(tsv_data.iterrows(), total=tsv_data.shape[0]):
        query = row['combined_info']
        response = retriever.invoke(query)
        context = "\n\n".join([f"Document {i+1}:\nContent: {doc.page_content}\nMetadata: {doc.metadata}" for i, doc in enumerate(response)])
        question = row['combined_info']
        
        # Print the full prompt being sent to GPT-3
        full_prompt = prompt.format(question=question, context=context)
        print(f"Full prompt sent to GPT-3:\n{full_prompt}")
        
        rag_chain = RunnableSequence(
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()},
            prompt,
            llm,
            StrOutputParser()
        )
        answer = rag_chain.invoke({"context": context, "question": question})
        results.append({
            "Query": query,
            "Response": response,
            "Answer": answer,
            "FullPrompt": full_prompt  # Add the full prompt to the results for reference
        })

    return results

results = hybrid_search_and_rag(tsv_data, retriever, openai_api_key)

# Print the results
for result in results:
    print(f"Query: {result['Query']}")
    print(f"Response: {result['Response']}")
    print(f"Answer: {result['Answer']}\n")
    # print(f"Full prompt sent to GPT-3:\n{result['FullPrompt']}\n")

print("Hybrid search with RAG processing complete.")
