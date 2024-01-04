from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from langchain.llms import LlamaCpp

DB_FAISS_PATH = "../vectorstores/db_faiss"

custom_prompt_template = """ Use the following pieces of information to answer the user's question.
If you don"t know the answer, please just say you don't know, don't try yo make up the answer.

Context: {}
Question: {question}

Only returns the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vector stores
    :return:
    """

    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

    return prompt


def load_llm():

    llm = LlamaCpp(
        model_path="/Users/rlm/Desktop/Code/llama.cpp/models/openorca-platypus2-13b.gguf.q4_0.bin",
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
    return llm


def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k':2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device':'cpu'})

    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query', query})
    return response

## Chainlit ##

@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot.....")
    await msg.send()
    msg.content = "Hi, Welcome to the Medicine Bot. What is your query?"
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.set("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    answer = res['result']
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += f"\nNo Sources Found"

    await cl.Message(content=answer).send()