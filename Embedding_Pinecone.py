from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import TextLoader
import api_keys


"""Create an Index"""

# pc = Pinecone(api_key=api_keys.os.environ["PINECONE_API_KEY"])

# pc.create_index(
#   name="ai-test",
#   dimension=768,
#   metric="cosine",
#   spec=ServerlessSpec(
#     cloud="aws",
#     region="us-east-1"
#   )
# )




pc = Pinecone(api_key=api_keys.os.environ["PINECONE_API_KEY"])

index=pc.Index("ai-test")
index_name = "ai-test"

embeddings = HuggingFaceEmbeddings()



"""**Store the Embedding in Pinecone**"""

import uuid

class DocumentProcessor:
    def __init__(self):
        self.encoded_data = []

    def file_loader(self, filename):
        loader = TextLoader(filename)
        return loader.load()

    def split_embeddind_docs(self, documents, chunk_size=1000, chunk_overlap=100):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(documents)
        for document in docs:
            document.metadata['text'] = document.page_content

        for record in docs:
            doc_uuid = uuid.uuid4()
            self.encoded_data.append({
                'uuid': str(doc_uuid),
                'vector': embeddings.embed_query(record.page_content),
                'metadata': record.metadata
            })

        return self.encoded_data

    def upsertion(self, index):
        for item in self.encoded_data:
            index.upsert(
                vectors=[{
                    'id': item["uuid"],
                    'values': item['vector'],
                    'metadata': item['metadata']
                }]
            )
        return "The data has been upserted to PINECONE."

processor = DocumentProcessor()



filename = "/content/AI Engineer Test txt.txt"
documents = processor.file_loader(filename)
processed_data = processor.split_embeddind_docs(documents)
result = processor.upsertion(index)
print(result)

