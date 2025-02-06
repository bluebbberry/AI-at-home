# rdf_knowledge_graph.py
import logging
from SPARQLWrapper import SPARQLWrapper, JSON
import json
import base64
import torch
import os
from dotenv import load_dotenv
from SPARQLWrapper import SPARQLWrapper, POST, JSON
from rdflib import URIRef, Literal
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)

class RDFKnowledgeGraph:
    def __init__(self, mastodon_client, fuseki_url=os.getenv("FUSEKI_SERVER_URL"), dataset="my-knowledge-base"):
        self.update_url = f"{fuseki_url}/{dataset}/update"
        self.query_url = f"{fuseki_url}/{dataset}/query"
        self.fuseki_url = fuseki_url + "/" + dataset
        self.mastodon_client = mastodon_client
        self.sparql = SPARQLWrapper(self.fuseki_url)

    def save_model(self, model_name, model_encoded):
        """
        Inserts the model parameters into the Fuseki knowledge base using base64 encoding.
        """
        logging.info(f"Saving model in knowledge base: {model_name}")
        #state_dict = {k: v.cpu().tolist() for k, v in model.state_dict().items()}  # Convert tensors to lists
        #state_json = json.dumps(state_dict)
        #state_encoded = base64.b64encode(state_json.encode('utf-8')).decode('utf-8')

        sparql_insert_query = f'''
        PREFIX ex: <http://example.org/>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        INSERT DATA {{
            ex:{model_name} a ex:BERTModel ;
                            ex:modelState "{model_encoded}" .
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
            return model
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

    def look_for_song_data_in_statuses_to_insert(self, messages):
        """
        Parses messages for general knowledge data (e.g., facts, Q&A) and inserts it into the RDF graph using SPARQL.
        """
        sparql_endpoint = SPARQLWrapper(self.fuseki_url)
        sparql_endpoint.setMethod('POST')

        for message in messages:
            content = message['content']
            if "question:" in content and "answer:" in content:
                try:
                    # Example format: "question: What is the capital of France? answer: Paris"
                    question_start = content.find("question:") + len("question:")
                    answer_start = content.find("answer:", question_start)
                    question = content[question_start:answer_start].strip()
                    answer = content[answer_start + len("answer:"):].strip()

                    # SPARQL Update to insert the Q&A pair
                    query = f"""
                    PREFIX ns: <http://example.org/>
                    INSERT DATA {{
                        ns:{question.replace(' ', '_')} a ns:Question ;
                            ns:text "{question}" ;
                            ns:answer "{answer}" .
                    }}
                    """
                    sparql_endpoint.setQuery(query)
                    sparql_endpoint.query()
                    logging.info(f"[INSERT] Inserted Q&A pair: '{question}' -> '{answer}' into the RDF graph.")
                except Exception as e:
                    logging.error(f"[ERROR] Failed to insert Q&A data: {e}", exc_info=True)

    def on_found_group_to_join(self, link_to_model):
        """
        Handles the event of finding a new knowledge repository by joining and updating the local graph using SPARQL.
        """
        sparql_endpoint = SPARQLWrapper(self.fuseki_url)
        sparql_endpoint.setMethod('POST')

        try:
            group_uri = URIRef(link_to_model)

            # SPARQL Update to add metadata for joining the group
            query = f"""
            PREFIX ns: <http://example.org/>
            INSERT DATA {{
                <{group_uri}> a ns:KnowledgeRepository ;
                              ns:joined true .
            }}
            """
            sparql_endpoint.setQuery(query)
            sparql_endpoint.query()
            logging.info(f"[JOIN] Successfully joined knowledge repository: {link_to_model}")
        except Exception as e:
            logging.error(f"[ERROR] Failed to join knowledge repository: {e}", exc_info=True)

    def fetch_all_model_from_knowledge_base(self, link_to_model):
        """
        Fetches all knowledge entries from the RDF graph for a specific repository link using SPARQL.
        """
        sparql_endpoint = SPARQLWrapper(self.fuseki_url)
        sparql_endpoint.setReturnFormat(JSON)

        try:
            query = f"""
            PREFIX ns: <http://example.org/>
            SELECT ?entry
            WHERE {{
                <{link_to_model}> ns:containsEntry ?entry .
            }}
            """
            sparql_endpoint.setQuery(query)
            results = sparql_endpoint.query().convert()

            knowledge_entries = [result['entry']['value'] for result in results['results']['bindings']]
            logging.info(f"[FETCH] Retrieved {len(knowledge_entries)} knowledge entries from the knowledge graph.")
            return knowledge_entries
        except Exception as e:
            logging.error(f"[ERROR] Failed to fetch knowledge entries: {e}", exc_info=True)
            return []

    def aggregate_model_states(self, local_model_state, all_models):
        """
        Aggregates knowledge data from the local graph and other repositories.
        """
        try:
            # Aggregation logic: combine unique entries
            aggregated_state = set(local_model_state)
            for model in all_models:
                aggregated_state.update(model)
            logging.info("[AGGREGATE] Aggregated knowledge entries successfully.")
            return list(aggregated_state)
        except Exception as e:
            logging.error(f"[ERROR] Failed to aggregate knowledge entries: {e}", exc_info=True)
            return local_model_state
