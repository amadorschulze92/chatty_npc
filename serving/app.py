import uvicorn
from fastapi import FastAPI
import gradio as gr
from loguru import logger
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.merge import MergedDataLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import openai
import os
from os.path import isfile, join
from langchain.document_loaders import TextLoader
from getpass import getpass
import time
import json




def read_lore(path: str):
    onlyfiles = [f for f in os.listdir(path) if isfile(join(path, f))]
    world = {}
    local = {}
    personal = {}
    for file in onlyfiles:
        loader = TextLoader(path+file)
        if 'world' in file: 
            world[file] = loader
        elif 'local' in file:
            local[file] = loader
        elif 'personal' in file:
            personal[file] = loader
    all_lore = {'world': world,
                'local': local,
                'personal': personal}
    return all_lore


def pick_split_docs(all_lore: dict, characters: dict, choose_char: str):
    loader_all = MergedDataLoader(loaders=[world['world_lore_boxle.txt'], 
                                        local['local_lore_'+characters[choose_char][0]+'.txt'],
                                        personal['personal_lore_'+characters[choose_char][1]+'.txt']
                                        ]
                            )
    all_docs = loader_all.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    split_doc = text_splitter.split_documents(all_docs)
    return split_doc


# character info
characters = {'Leo': ['coxet', '1'], 
            'Lila': ['wythoff', '2'], 
            'Ethan': ['hanner', '3'], 
            'Axel': ['hedron', '4']}


def doc_into_db(split_doc, embedding_function: SentenceTransformerEmbeddings, choose_char: str):
    db = Chroma.from_documents(split_doc, embedding_function, collection_name=choose_char)
    characters[choose_char].append(db)


def question_answer(choose_char: str, user_query: str, db):
    query = """If you don't know the answer, just say that you don't know, don't try to make up 
                an answer. Use three sentences maximum and keep the answer as concise as 
                possible. Answer from the perspective of """+choose_char+". "+user_query
    retriever = db.as_retriever()
    OpenAI_key = os.environ.get("OPENAI_API_KEY")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, verbose=True, openai_api_key=OpenAI_key)
    qa = RetrievalQA.from_chain_type(llm, 
                                    chain_type='stuff', 
                                    retriever=retriever,
                                    )
    answer = qa.run(query)
    return answer



# get embedding
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# get files
all_lore = read_lore('../lore/')
world = all_lore['world']
local = all_lore['local']
personal = all_lore['personal']
app = FastAPI()


# web stuff
@app.get("/")
def get_root() -> dict:
    logger.info("Received request on the root endpoint")
    return {"status": "ok"}


@app.post("/ask")
async def ask_api(character_name: str, user_query: str) -> str:
    # log timing and network
    started_at = time.time()
    # process
    split_doc = pick_split_docs(all_lore, characters, character_name)
    doc_into_db(split_doc, embedding_function, character_name)
    db = characters[character_name][2]
    answer = question_answer(character_name, user_query, db)
    # log stats
    total_time = time.time() - started_at
    log = {
        "question": user_query,
        "answer": answer,
        "latency": total_time
        }
    logger.info(json.dumps(log))
    return answer


# gradio app
def gradio_question_answer(character_name, question):
    split_doc = pick_split_docs(all_lore, characters, character_name)
    doc_into_db(split_doc, embedding_function, character_name)
    db = characters[character_name][2]
    answer = question_answer(character_name, question, db)
    return answer


demo = gr.Interface(
    fn=gradio_question_answer,
    inputs=[gr.Textbox(lines=1, placeholder=f"Choose Character {*characters.keys(),}"), 
            gr.Textbox(lines=1, placeholder="Enter Question")
            ],
    outputs="text",
    live=False
)

app = gr.mount_gradio_app(app, demo, path='/gradio')



if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
