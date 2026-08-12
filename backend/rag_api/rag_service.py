import re
import torch

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# Loading the embedding model for the RAG pipeline
print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Importing the Chroma vector store
print("Connecting to Chroma...")

vector_store = Chroma(
    collection_name="rag_documents",
    embedding_function=embeddings,
    persist_directory="../chroma_db"
)

print("Documents in Chroma:", vector_store._collection.count())

# Creating a retriever from the Chroma vector store for the 3 most relevant document chunks based on user questions.
retriever = vector_store.as_retriever(
    search_type="similarity", # Using vector distance similarity matching
    search_kwargs={"k": 3} # Targeting the top 3 closest matches for question answers
)

print("Loading Qwen3-4B...")

model_name = "Qwen/Qwen3-4B"

# Loading the AutoTokenizer, which is used to convert text into tokens
tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

# Loading the Qwen3-4B model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto"
)

# Running the model on CPU
model = model.to("cpu")


# Creating the text-generation pipeline using Qwen3-4B
generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    do_sample=False, # Disabling sampling for stable output
    repetition_penalty=1.1, # Added penalty to prevent phrase repetition
    return_full_text=False # Applied to exclude the prompt from the output
)

# Wrapping the Hugging Face pipeline so it can be used with LangChain
llm = HuggingFacePipeline(
    pipeline=generation_pipeline
)

# Defining the retrieval function called by our API layer
def ask_question(question):
    # Query the active vector database index to pull down relevant document chunks
    results = retriever.invoke(question)

    print("Question:", question)
    print("Number of retrieved chunks:", len(results))

    # Processing and printing out information for each document chunk captured in the results list
    for i, doc in enumerate(results, start=1):

        print(f"CHUNK {i}")
        print(
            "Source:",
            doc.metadata.get("source")
        )
        print(
            "Page:",
            doc.metadata.get("page")
        )
        print(
            "Content:",
            doc.page_content[:500]
        )

    # Added a fallback option for cases where the model is not able to retrieve the desired answer
    fallback = (
        "I cannot answer this based on the provided documents."
    )

    if not results:
        return fallback

# Combining the retrieved document chunks into a single context string
    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('source')}\n"
            f"Page: {doc.metadata.get('page')}\n"
            f"Content: {doc.page_content}"
            for doc in results
        ]
    )
    
    print("Context length:", len(context))

# Setting the RAG prompt using the retrieved context and the user's question
    prompt = f"""
You are a helpful assistant answering questions about HIV service delivery.

Answer the QUESTION using ONLY the information in the CONTEXT.

Give ONE concise answer consisting of 1-2 sentences.

Stop immediately after giving the answer.

Do not repeat the answer.
Do not repeat any sentence.
Do not explain your reasoning.
Do not evaluate your answer.
Do not mention these instructions.
Do not use outside knowledge.
Do not add information that is not supported by the context.

If the context does not contain enough information to answer the question,
respond exactly:

I cannot answer this based on the provided documents.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
"""
    # Generating the answer by passing the prompt to the LLM
    response = llm.invoke(prompt)
    
    # Printing a raw response
    print(repr(response))

    # Cleaning up the response
    response = response.strip() # Removing extra spaces from the answer, if required

    # Checking if the answer contains Qwen's thinking process
    if "</think>" in response:
        # Used to split text at the tag and pick everything that occurs after it
        response = response.split()[1].strip()

    # Check if the fallback was generated in the response
    if fallback in response:
        return fallback

    # Strip out a case-insensitive "ANSWER:" prefix from the beginning of the text
    response = re.sub(r"^\s*ANSWER:\s*", "", response, flags=re.IGNORECASE).strip()

    # Verifying if the "answer:" label appears in the genrated output
    if response.lower().count("answer:") > 0:
        # Splitting the response string every time the case-insensitive label appears
        parts = re.split(r"answer:\s*", response, flags=re.IGNORECASE)
        response = parts[-1].strip()

    # Stripping out all inline source references matching the (Source: ...) bracket pattern#
    response = re.sub(r"\(Source:.*?\)", "", response, flags=re.IGNORECASE).strip()

    # Defined an array of phrases that should not appear in the final answer
    stop_phrases = ["Wait,", "Wait but", "However,", "But the user", "The user said", "The user wants", "The answer is", "The response is", "The answer provided", "The response provided", "Let me check", "Let me make sure", "I should", "I need to", "I think", "I can confirm", "This is correct", "This answer is correct", "No errors","There is no error", "There are no errors"]
    lower_response = response.lower()

    for phrase in stop_phrases:
        # Within the for loop, searching for the starting position index of the stop phrase
        index = lower_response.find(phrase.lower())

        # Stripping the original response string up to the match index if a match is found
        if index != -1:
            response = response[:index].strip()
            break

    # Removing any remaining "Answer" or "Final Answer" labels
    response = re.sub(r"\s*(Answer|Final Answer)\s*:?\s*$", "", response, flags=re.IGNORECASE).strip()

    # Splitting on spaces that follow ., ! or ?
    sentences = re.split(r"(?<=[.!?])\s+", response)
    # Joining the text together by taking only the first two sentence elements and adding a space
    response = " ".join(sentences[:2]).strip()


    # Returning the fallback if no answer was retrieved
    if not response:
        return fallback

    # Getting the metadata from the most relevant retrieved chunk
    source = results[0].metadata.get("source", "document5.pdf")
    page = results[0].metadata.get("page", "unknown")

    # Normalizing the path separators to extract just the final filename
    source = (source.replace("\\", "/").split("/")[-1])

    # Setting the citation and final response variables
    citation = (f"(Source: {source}, Page: {page})")
    final_response = (f"{response} {citation}")

    print("\nFinal Response:\n")
    print(final_response)
    return final_response