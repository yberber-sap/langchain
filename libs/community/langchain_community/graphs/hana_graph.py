from __future__ import annotations

import io
import csv

from typing import (
    TYPE_CHECKING,
    Optional,
)

if TYPE_CHECKING:
    import rdflib
    from hdbcli import dbapi

try:
    import rdflib
except ImportError:
    raise ImportError(
        "Could not import rdflib. Please install it with `pip install rdflib`."
    )
try:
    from hdbcli import dbapi
except ImportError:
    raise ImportError(
        "Could not import hdbcli. Please install it with `pip install hdbcli`."
    )

class HanaGraph:
    """HANA Graph SPARQL endpoint wrapper for graph operations.

    This class connects to a SAP HANA Graph SPARQL endpoint, executes queries,
    and can load ontology schema data from either a local file or via a
    CONSTRUCT SPARQL query.

    *Security note*: Make sure that the database connection uses credentials
    that are narrowly-scoped. Do not expose unnecessary privileges. For more
    information on security best practices, see:
    https://python.langchain.com/docs/security
    """

    def __init__(
        self,
        connection: dbapi.Connection,
        ontology_uri: str,
        graph_uri: Optional[str] = None, #  use default graph if no graph_uri was provided
    ) -> None:


        self.connection = connection
        self.ontology_uri = ontology_uri
        self.graph_uri = graph_uri

        ontology_schema_graph = self._load_ontology_schema_graph()
        self.schema = ontology_schema_graph.serialize(format="turtle")


    def _add_graph_uri_to_query(self, query):
        if " FROM " not in query:
            query.replace(" WHERE", f" FROM <{self.graph_uri}> WHERE")
        return query

    def query(self, qrystr: str, rqx_hdrs=None, add_graph_uri=False):
        '''query() - executes SPARQL query
             Returns a tuple of (content-type, response data) to be send back
        '''

        if rqx_hdrs is None: rqx_hdrs = '''Accept: application/sparql-results+csv\r\nContent-Type: application/sparql-query'''

        if qrystr is None: return ('', '')

        if add_graph_uri and self.graph_uri:
            qrystr = self._add_graph_uri_to_query(qrystr)

        cursor = self.connection.cursor()
        try:
            r = cursor.callproc('SYS.SPARQL_EXECUTE', (qrystr, rqx_hdrs, '?', None))
            resp = r[2]
            # csize = -1
            # for rh in r[3].split('\n'):
            #     hv = rh.split(':', 1)
            #     if len(hv) != 2: continue
            #     nm = hv[0].strip().lower()
            #     val = hv[1].strip()
            #     if nm == 'content-type':
            #         ctype = val
            #     elif nm == 'content-size':
            #         csize = int(val)
            # if 0 == csize:
            #     ctype = 'text/plain'
        except dbapi.Error as dberr:
            resp = dberr.errortext.split('; Server Connection', 1)[0]
        cursor.close()
        return resp


    def _load_ontology_schema_graph(self):
        ontology_query = f"SELECT   ?s ?o ?p FROM <{self.ontology_uri}> WHERE" + "{?s ?o ?p .}"
        response = self.query(ontology_query)
        ontology_triples = self.convert_csv_response_to_list(response)

        graph = rdflib.Graph()
        for s_val, o_val, p_val in ontology_triples:
            # Subject
            if s_val.startswith("_:"):
                subject = rdflib.BNode(s_val[2:])  # remove '_:' prefix for bnodes
            else:
                subject = rdflib.URIRef(s_val)

            # Predicate (usually a URI)
            if o_val.startswith("_:"):
                predicate = rdflib.BNode(o_val[2:])
            else:
                predicate = rdflib.URIRef(o_val)

            # Object could be a URI, bnode, or literal
            if p_val.startswith("http://") or p_val.startswith("https://"):
                obj = rdflib.URIRef(p_val)
            elif p_val.startswith("_:"):
                obj = rdflib.BNode(p_val[2:])
            else:
                # If the object is not a URI, treat it as a literal.
                # If you know the data type, you can provide it. For instance:
                # If there's a known datatype, like xsd:string, you can do:
                # obj = Literal(p_val, datatype=XSD.string)
                # Without extra info, just treat as a plain literal:
                obj = rdflib.Literal(p_val)

            graph.add((subject, predicate, obj))

        return graph

    @staticmethod
    def convert_csv_response_to_list(csv_string):
        csv_file = io.StringIO(csv_string)
        reader = csv.DictReader(csv_file)
        return [list(row.values()) for row in reader]

    @staticmethod
    def convert_csv_response_to_dataframe(result):
        import pandas as pd
        result_df = pd.read_csv(io.StringIO(result))
        return result_df.fillna('')


    def _load_ontology_schema_with_query(self, query: str):  # type: ignore[no-untyped-def]
        """
        Execute the CONSTRUCT query against the HANA Graph endpoint to load the ontology schema.
        """
        from rdflib.exceptions import ParserError

        try:
            results = self.graph.query(query)
        except ParserError as e:
            raise ValueError(f"Invalid SPARQL query:\n{e}")

        return results.graph

    def refresh_schema(self):
        ontology_schema_graph = self._load_ontology_schema_graph()
        self.schema = ontology_schema_graph.serialize(format="turtle")

    @property
    def get_schema(self) -> str:
        """
        Return the schema in Turtle format.
        """
        return self.schema
