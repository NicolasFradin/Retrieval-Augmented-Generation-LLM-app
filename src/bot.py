from model import qa_bot
import chainlit as cl

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
    chain = cl.user_session.get("chain")

    # Define the Asynchronous CallBack Handler
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = False

    res = await chain.acall(message.content, callbacks=[cb])
    answer = res['result']
    sources = res["source_documents"]

    if sources:
        answer += f"\n\nSources:" + str(sources)
    else:
        answer += f"\n\nNo Sources Found"

    await cl.Message(content=answer).send()
