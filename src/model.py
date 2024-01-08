import os
import logging
import chromadb
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain_community.llms import LlamaCpp
from vectorize import vectorize_documents
from chromadb.utils import embedding_functions
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

"""

"""

logging.basicConfig(level=logging.INFO)


chroma_client = chromadb.HttpClient(host=os.environ['DB_HOST'],       # Use 'localhost' for local run
                                    port='8000')

src_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) # Set the absolute path of the sql query that could be located in a separated folder depending on where the function is run.
# logging.info('src_folder %s', src_folder)

MODEL_PATH = os.path.join(src_folder, "src/models/llama-2-7b-chat.Q4_K_M.gguf")


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vector stores
    :return:
    """
    custom_prompt_template = """ Use the following pieces of information to answer the user's question.
    If you don"t know the answer, please just say you don't know, don't try yo make up the answer.
    
    Context: {context}
    Question: {question}
    
    Only returns the helpful answer below and nothing else.
    Helpful answer:
    """

    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

    return prompt


def load_llm():

    #llm1 = Llama(model_path=MODEL_PATH, n_ctx=1024, n_batch=126)

    llm2 = LlamaCpp(
        model_path=MODEL_PATH,
        n_gpu_layers=0, # Only one layer of the model will be loaded into GPU memory (1 is often sufficient).
        n_batch=512, # number of tokens the model should process in parallel
        n_ctx=2048, # The model will consider a window of 2048 tokens at a time
        f16_kv=True, # The model will use half-precision for the key/value cache, which can be more memory efficient; Metal only supports True.
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=True,
        max_tokens=512,
        temperature=0.5,
        #top_p=0.5,
        echo=True,              # Echo the prompt back in the output
        #stop=["#"],  # Stop generating just before the model would generate a new question
    )
    return llm2


def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(             # RetrievalQAWithSourcesChain to test (https://stackoverflow.com/questions/77454057/different-functions-of-qa-retrieval-in-langchain)
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="similarity", search_kwargs={'k':2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain


def qa_bot():

    logging.info('********************************** %s', os.environ['RESET_DB'])

    if os.environ['RESET_DB']:
        chroma_client.reset()
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2", device='cpu')     # We need to use a transformers in the following list https://docs.trychroma.com/embeddings
        vectorize_documents(chroma_client, embedding_function, "data/", 'medical_collection', 'l2')

    db = Chroma(
        client=chroma_client,
        collection_name="medical_collection",
        embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device':'cpu'})
    )

    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa


def final_result(query):    # Not used
    qa_result = qa_bot()
    response = qa_result({'query', query})
    return response


def main():

    os.environ['RESET_DB'] = 'False'
    os.environ['DB_HOST'] = 'localhost'

    # embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device':'cpu'}) # Equivalent to SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2", device='cpu')     # We need to use a transformers in the following list https://docs.trychroma.com/embeddings

    if os.environ['RESET_DB'] == 'True':
        chroma_client.reset()
        vectorize_documents(chroma_client, embedding_function, "data/", 'medical_collection', 'l2')

    # tell LangChain to use our client and collection name
    db = Chroma(
        client=chroma_client,
        collection_name="medical_collection",
        embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device':'cpu'}),        # Using SentenceTransformerEmbeddingFunction doesn't work with Langchain Chroma class
    )

    logging.info(chroma_client.list_collections())

    data = chroma_client.get_collection(name='medical_collection').get()
    logging.info(data)

    query = "What is a disorder?"
    docs = db.similarity_search(query)
    logging.info(docs[0].page_content)


if __name__ == '__main__':
    main()