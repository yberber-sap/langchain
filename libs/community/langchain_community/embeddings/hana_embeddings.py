
from langchain_core.embeddings import Embeddings
from hdbcli import dbapi
from typing import  List
import json

class HANAEmbeddings(Embeddings):

    def __init__(self,  connection: dbapi.Connection, model_version:str):
        self.connection = connection
        self.model_version = model_version


    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if texts is None:
            return None

        embeddings: List[List[float]] = []
        sql_str = "SELECT TO_NVARCHAR(VECTOR_EMBEDDING(:content, 'DOCUMENT', :model_version)) FROM DUMMY;"

        try:
            # returns strings or bytes instead of a locator
            cur = self.connection.cursor()
            for text in texts:
                cur.execute(sql_str, content=text, model_version=self.model_version)
                print(f"-> cur.execute was called with the sql_str: \"{sql_str}\"")
                if cur.has_result_set():
                    res = cur.fetchall()
                    for row in res:
                        embeddings.append(json.loads(row[0]))

        finally:
            cur.close()

        return embeddings



    def embed_query(self, query: str) -> List[float]:
        if query is None:
            return None

        embedding: List[float] = None
        sql_str = "SELECT TO_NVARCHAR(VECTOR_EMBEDDING(:content, 'QUERY', :model_version)) FROM DUMMY;"

        try:
            # returns strings or bytes instead of a locator
            cur = self.connection.cursor()
            cur.execute(sql_str, content=query, model_version=self.model_version)
            print(f"-> cur.execute was called with the sql_str: \"{sql_str}\"")
            if cur.has_result_set():
                res = cur.fetchall()
                embedding=json.loads(res[0][0])
        finally:
            cur.close()

        return embedding

