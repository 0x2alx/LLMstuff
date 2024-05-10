from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.memory import ChatMessageHistory
from langchain.memory import ZepMemory
from langchain.retrievers import ZepRetriever

import chainlit as cl

ZEP_API_URL = "http://localhost:8000"


@cl.on_chat_start
async def on_chat_start():
    chat_model = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2, streaming=True)
    msg_hist = ChatMessageHistory()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer all questions to the best of your ability.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    runnable_chain = prompt | chat_model | StrOutputParser()
    cl.user_session.set("runnable_chain", runnable_chain)
    cl.user_session.set("msg_hist", msg_hist)


@cl.on_message
async def on_message(message: cl.Message):
    runnable_chain = cl.user_session.get("runnable_chain")  # type: Runnable
    msg_hist = cl.user_session.get("msg_hist")  # type: Runnable

    msg_hist.add_user_message(message.content) 

    msg = cl.Message(content="")

    async for chunk in runnable_chain.astream(
        {"messages": msg_hist.messages},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    print(f"\n{dir(msg)=}")
    print(f"\n{msg.content=}")
    msg_hist.add_ai_message(msg.content)
    print(f"\n\n{msg_hist=}")# Add the response

    await msg.send()
