# rdf_knowledge_graph.py
import logging
from SPARQLWrapper import SPARQLWrapper, JSON
import json
import base64
import torch
import os
from dotenv import load_dotenv
import csv
import pandas as pd

load_dotenv()
logging.basicConfig(level=logging.INFO)

class RDFKnowledgeGraph:
    def __init__(self, mastodon_client, fuseki_url=os.getenv("FUSEKI_SERVER_URL"), dataset="my-knowledge-base"):
        self.update_url = f"{fuseki_url}/{dataset}/update"
        self.query_url = f"{fuseki_url}/{dataset}/query"
        self.fuseki_url = fuseki_url + "/" + dataset
        self.mastodon_client = mastodon_client
        self.sparql = SPARQLWrapper(self.fuseki_url)

    def save_model(self, model_name, model):
        """
        Inserts the model parameters into the Fuseki knowledge base using base64 encoding.
        """
        state_dict = {k: v.cpu().tolist() for k, v in model.state_dict().items()}  # Convert tensors to lists
        state_json = json.dumps(state_dict)
        state_encoded = base64.b64encode(state_json.encode('utf-8')).decode('utf-8')

        sparql_insert_query = f'''
        PREFIX ex: <http://example.org/>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        INSERT DATA {{
            ex:{model_name} a ex:BERTModel ;
                            ex:modelState "{state_encoded}" .
        }}
        '''
        self._execute_update_query(sparql_insert_query)
        logging.info(f"Model '{model_name}' saved successfully.")

    def load_model(self, model_name, model):
        """
        Loads the model parameters from the Fuseki knowledge base and updates the model.
        """
        sparql_select_query = f'''
        PREFIX ex: <http://example.org/>
        SELECT ?modelState
        WHERE {{
            ex:{model_name} a ex:BERTModel ;
                            ex:modelState ?modelState .
        }}
        '''
        results = self._execute_select_query(sparql_select_query)

        if results:
            state_encoded = results[0]["modelState"]["value"]
            state_json = base64.b64decode(state_encoded).decode('utf-8')
            state_dict = json.loads(state_json)
            state_dict = {k: torch.tensor(v) for k, v in state_dict.items()}  # Convert lists back to tensors
            model.load_state_dict(state_dict)
            logging.info(f"Model '{model_name}' loaded successfully.")
        else:
            logging.warning(f"Model '{model_name}' not found in the knowledge base.")

    def store_qa_pair(self, question, answer):
        """
        Stores a question-answer pair in the Fuseki knowledge base.
        """
        qa_id = f"qa_{abs(hash(question))}"  # Generate a unique ID for the QA pair
        sparql_insert_query = f'''
        PREFIX ex: <http://example.org/>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        INSERT DATA {{
            ex:{qa_id} a ex:QAPair ;
                        ex:question "{question}" ;
                        ex:answer "{answer}" .
        }}
        '''
        self._execute_update_query(sparql_insert_query)
        logging.info(f"QA pair (question: '{question}', answer: '{answer}') stored successfully.")

    def fetch_qa_pairs(self):
        """
        Retrieves all question-answer pairs from the Fuseki knowledge base.
        """
        sparql_select_query = '''
        PREFIX ex: <http://example.org/>
        SELECT ?question ?answer
        WHERE {
            ?qa a ex:QAPair ;
                ex:question ?question ;
                ex:answer ?answer .
        }
        '''
        results = self._execute_select_query(sparql_select_query)

        qa_pairs = [{"question": result["question"]["value"], "answer": result["answer"]["value"]} for result in results]
        logging.info(f"Fetched {len(qa_pairs)} QA pairs from the knowledge base.")
        return qa_pairs

    def _execute_update_query(self, query):
        """
        Executes a SPARQL update query.
        """
        sparql = SPARQLWrapper(self.update_url)
        sparql.setQuery(query)
        sparql.setMethod('POST')
        sparql.setReturnFormat(JSON)

        try:
            sparql.query()
        except Exception as e:
            logging.error(f"Error executing update query: {e}")

    def _execute_select_query(self, query):
        """
        Executes a SPARQL SELECT query and returns the results.
        """
        sparql = SPARQLWrapper(self.query_url)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)

        try:
            results = sparql.query().convert()
            return results["results"]["bindings"]
        except Exception as e:
            logging.error(f"Error executing select query: {e}")
            return []
