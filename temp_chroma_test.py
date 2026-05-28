import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(path='test_chroma', settings=Settings())
col = client.create_collection(name='test')
col.add(ids=['1','2'], documents=['hello','world'], metadatas=[{'source':'a'},{'source':'b'}], embeddings=[[0.1,0.1,0.1],[0.2,0.2,0.2]])
res = col.query(query_texts=['hello'], n_results=1)
print(res)
