import os
import chromadb
from chromadb.utils import embedding_functions
import textwrap

INPUT_FILE = "input.txt"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DB_DIRECTORY = "vector_db"

def read_and_chunk_text(file_path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    print(f"Reading file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if '|' in line:
            parts = line.split('|', 1)
            if len(parts) > 1:
                cleaned_lines.append(parts[1].strip())
        else:
            cleaned_lines.append(line.strip())
    
    cleaned_text = '\n'.join(cleaned_lines)
    
    chunks = []
    for i in range(0, len(cleaned_text), chunk_size - chunk_overlap):
        chunk = cleaned_text[i:i + chunk_size]
        if chunk:
            chunks.append(chunk)
    
    print(f"Created {len(chunks)} chunks from the text")
    return chunks

def create_vector_db(chunks):
    print("Creating vector database...")
    
    os.makedirs(DB_DIRECTORY, exist_ok=True)
    
    client = chromadb.PersistentClient(path=DB_DIRECTORY)
    
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    try:
        client.delete_collection("constitution")
    except:
        pass
    
    collection = client.create_collection(
        name="constitution",
        embedding_function=sentence_transformer_ef
    )
    
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            metadatas=[{"source": "US Constitution", "chunk_id": i}],
            ids=[f"chunk_{i}"]
        )
    
    print(f"Added {len(chunks)} chunks to the vector database")
    return client

def query_vector_db(query, n_results=3):
    client = chromadb.PersistentClient(path=DB_DIRECTORY)
    
    collection = client.get_collection(
        name="constitution",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    return results

def format_results(results):
    formatted_output = []
    
    if not results or not results['documents'] or not results['documents'][0]:
        return "No results found."
    
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0], 
        results['metadatas'][0], 
        results['distances'][0]
    )):
        relevance = 1 - distance
        formatted_doc = textwrap.fill(doc, width=80)
        
        result = f"\n--- Result {i+1} (Relevance: {relevance:.2f}) ---\n"
        result += f"{formatted_doc}\n"
        
        formatted_output.append(result)
    
    return "\n".join(formatted_output)

def main():
    if not os.path.exists(DB_DIRECTORY) or not os.listdir(DB_DIRECTORY):
        print("Vector database not found. Creating new database...")
        chunks = read_and_chunk_text(INPUT_FILE)
        create_vector_db(chunks)
        print("Vector database created successfully!")
    else:
        print("Vector database already exists.")
    
    print("\nUS Constitution Vector Database Query System")
    print("Type 'exit' to quit the program")
    
    while True:
        query = input("\nEnter your question: ")
        
        if query.lower() in ['exit', 'quit', 'q']:
            break
        
        if not query.strip():
            continue
        
        results = query_vector_db(query)
        formatted_results = format_results(results)
        
        print("\n=== Search Results ===")
        print(formatted_results)

if __name__ == "__main__":
    main()