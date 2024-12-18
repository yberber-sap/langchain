"""Question answering over a SAP HANA graph using SPARQL."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain_core.callbacks.manager import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.base import BasePromptTemplate
from pydantic import Field
from langchain.prompts import PromptTemplate
from langchain_community.graphs.hana_graph import HanaGraph


HANA_GRAPH_SPARQL_GENERATION_TEMPLATE = """
You are an expert at writing SPARQL queries. You have a given RDF schema, and a natural language request.
Your task: Interpret the natural language request and produce a valid SPARQL query that answers it
based on the given schema. Make sure to correctly use IRIs, classes, and properties from the schema.

RDF Schema (in Turtle):
{schema}

User request: {prompt}

Please write only a SPARQL query that retrieves the requested information. Do not wrap your answer in backticks or code fences.
"""
HANA_GRAPH_SPARQL_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "prompt"],
    template=HANA_GRAPH_SPARQL_GENERATION_TEMPLATE,
)

HANA_GRAPH_QA_TEMPLATE = """
You are an assistant who answers questions based on given data.

User request: {prompt}

Below are the results obtained from the SPARQL query in a structured format (e.g., JSON):
{context}

Your task:
- Using the provided SPARQL query results, answer the user's question in a concise and direct manner.
- If you cannot find an answer in the data, say so.
"""
HANA_GRAPH_QA_PROMPT = PromptTemplate(
    input_variables=["prompt", "context"], template=HANA_GRAPH_QA_TEMPLATE
)


class HanaGraphQAChain(Chain):
    """Question-answering against a SAP HANA Knowledge Graph by generating SPARQL queries.

    *Security note*: As with any database interaction, ensure that your credentials and
    permissions are scoped appropriately. The chain can potentially produce SPARQL that
    modifies data if allowed by HANA permissions. Always restrict credentials to read-only
    if possible.
    """

    graph: HanaGraph = Field(exclude=True)
    sparql_generation_chain: LLMChain
    qa_chain: LLMChain
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:

    allow_dangerous_requests: bool = False

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the chain."""
        super().__init__(**kwargs)
        if self.allow_dangerous_requests is not True:
            raise ValueError(
                "In order to use this chain, you must acknowledge that it can make "
                "dangerous requests by setting `allow_dangerous_requests` to `True`."
                "You must narrowly scope the permissions of the database connection "
                "to only include necessary permissions. Failure to do so may result "
                "in data corruption or loss or reading sensitive data if such data is "
                "present in the database."
                "Only use this chain if you understand the risks and have taken the "
                "necessary precautions. "
                "See https://python.langchain.com/docs/security for more information."
            )

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        *,
        sparql_generation_prompt: BasePromptTemplate = HANA_GRAPH_SPARQL_GENERATION_PROMPT,
        qa_prompt: BasePromptTemplate = HANA_GRAPH_QA_PROMPT,
        **kwargs: Any,
    ) -> HanaGraphQAChain:
        sparql_generation_chain = LLMChain(llm=llm, prompt=sparql_generation_prompt)
        qa_chain = LLMChain(llm=llm, prompt=qa_prompt)
        return cls(
            qa_chain=qa_chain,
            sparql_generation_chain=sparql_generation_chain,
            **kwargs,
        )

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        """Generate SPARQL query, use it to look up in the graph and answer the question."""
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()

        # Extract user question
        question = inputs[self.input_key]

        # Generate SPARQL query from the question and schema
        sparql_result = self.sparql_generation_chain.invoke(
            {"prompt": question, "schema": self.graph.get_schema},
            callbacks=callbacks
        )
        # Extract the generated SPARQL string from the result dictionary
        generated_sparql = sparql_result[self.sparql_generation_chain.output_key]

        # Log the generated SPARQL
        _run_manager.on_text("Generated SPARQL:", end="\n", verbose=self.verbose)
        _run_manager.on_text(generated_sparql, color="green", end="\n", verbose=self.verbose)

        # Execute the generated SPARQL query against the graph
        context = self.graph.query(generated_sparql)

        # Log the full context (SPARQL results)
        _run_manager.on_text("Full Context:", end="\n", verbose=self.verbose)
        _run_manager.on_text(str(context), color="green", end="\n", verbose=self.verbose)

        # Pass the question and query results into the QA chain
        qa_chain_result = self.qa_chain.invoke(
            {"prompt": question, "context": context},
            callbacks=callbacks
        )
        # Extract the final answer from the result dictionary
        result = qa_chain_result[self.qa_chain.output_key]

        # Return the final answer
        return {self.output_key: result}