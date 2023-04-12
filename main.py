from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.document_loaders import TextLoader

loader = TextLoader('./ttt.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

url = "http://localhost:6333/"
qdrant = Qdrant.from_documents(
    docs, embeddings, 
    url, prefer_grpc=True, 
    collection_name="my_documents",
)

query = "日本首相"
found_docs = qdrant.similarity_search(query)

print(found_docs[0].page_content)