from fastapi import FastAPI, UploadFile, File
import os 
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import panel as pn
import tempfile
os.environ["OPENAI_API_KEY"] = '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'
app = FastAPI()

@app.get("/")
async def default_return():
    return {"tis the time to":"celebrate christmas"}

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    with open(file.filename, "wb") as f:
        f.write(await file.read())
    contents = await file.read()
    return {"file_contents": contents.decode("utf-8")}
k=2
file="./example.pdf"
chain_type = "map_reduce"
loader = PyPDFLoader(file)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

db = Chroma.from_documents(texts, embeddings)

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

qa1 = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type=chain_type, retriever=retriever, return_source_documents=True)

def qa(query):    
    result = qa1({"query": query})
    print(result['result'])
    print(result)
    return result

@app.get("/banana/")
async def get_me_an_answer(question: str):
    print(question)
    return {"answer":qa(question)}
