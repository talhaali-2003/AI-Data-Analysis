from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import ssl
from ssl import create_default_context
import json
import requests
import os
import re
import logging
import datetime
import pytz
import urllib3
import pprint
import argparse
from pprint import pprint
from datetime import datetime, timedelta
urllib3.disable_warnings()

def setup_logger():
    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  

    # Create handlers for file and console
    file_handler = logging.FileHandler('data_analyse.log')
    console_handler = logging.StreamHandler()

    # Create formatters and add it to handlers
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file_formatter = logging.Formatter(log_format)
    console_formatter = logging.Formatter(log_format)

    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logging = setup_logger()

# Load environment variables from .env file
load_dotenv()

es_host = os.getenv('ES_HOST', '')
es_port = int(os.getenv('ES_PORT', 9250))
es_scheme = os.getenv('SCHEME', 'https')
index_name = os.getenv('INDEX', '')
user = os.getenv('user')
password = os.getenv('pass')


def get_metrics(hostname, gte, lte):


    # Define your headers
    headers = {
        'Content-Type': 'application/json'
    }
    # construct the full host URL
    es_full_host = f"{es_scheme}://{es_host}:{es_port}"
    logging.info(f"Querying Elasticsearch at {es_full_host} for health metrics hostname: {hostname}")

    # Elasticsearch query
    query = {
              "track_total_hits": "true",
              "sort": [
                {
                  "TIMESTAMP": {
                    "order": "desc",
                    "unmapped_type": "boolean"
                  }
                }
              ],
              "fields": [
                {
                  "field": "@timestamp",
                  "format": "strict_date_optional_time"
                },
                {
                  "field": "START_TIME",
                  "format": "strict_date_optional_time"
                },
                {
                  "field": "TIMESTAMP",
                  "format": "strict_date_optional_time"
                },
                {
                  "field": "CPU"
                },
                {
                  "field": "RAM"
                },
                {
                  "field": "SWAP"
                }
              ],
              "_source": "false",
              "size" : "10000",
              "query": {
                "bool": {
                  "must": [],
                  "filter": [
                    {
                      "range": {
                        "TIMESTAMP": {
                          "format": "strict_date_optional_time",
                          "gte": gte,
                          "lte": lte
                        }
                      }
                    },
                    {
                      "match_phrase": {
                        "type.keyword": "mmr_metro"
                      }
                    },
                    {
                      "match_phrase": {
                        "host.keyword": hostname
                      }
                    }
                  ],
                  "should": [],
                  "must_not": []
                }
              }
            }

    try:
        # Scroll parameter (keep the search context alive for this duration)
        scroll = '5m'  

        # Connect to Elasticsearch with basic authentication and SSL context
        es = Elasticsearch(hosts=es_full_host , basic_auth=(user, password), verify_certs=False)
        logging.info(f"connexion established: {es}")

        # Perform a search query
        response = es.search(index=index_name, body=query, scroll=scroll)
        logging.info(f"Search retrieved successfully for hostname {hostname}")

        sid = response['_scroll_id']
        scroll_size = response['hits']['total']['value']
        logging.info(f"Scroll size: {scroll_size}")
       
        # Initialize a list to keep track of the results
        results = []

        # Start scrolling
        while scroll_size > 0:
            # Before fetching the next page, add the current batch to our results list
            results += response['hits']['hits']
            
            # Fetch the next page of results
            response = es.scroll(scroll_id=sid, scroll=scroll)
            
            # Update the scroll ID and size
            sid = response['_scroll_id']
            scroll_size = len(response['hits']['hits'])

        return results

    except Exception as e:

        error_message = "Error pulling metrics :" + str(e)
        logging.error(f"Error during request: {error_message}")
        return error_message
    


def get_process(hostname, gte, lte):

    # Define your headers
    headers = {
        'Content-Type': 'application/json'
    }
    # construct the full host URL
    es_full_host = f"{es_scheme}://{es_host}:{es_port}"
    logging.info(f"Querying Elasticsearch at {es_full_host} for processes for hostname: {hostname}")
    # Elasticsearch query
    query = {
              "track_total_hits": "true",
              "sort": [
                {
                  "TIMESTAMP": {
                    "order": "desc",
                    "unmapped_type": "boolean"
                  }
                }
              ],
              "fields": [
                {
                  "field": "*",
                  "include_unmapped": "true"
                },
                {
                  "field": "@timestamp",
                  "format": "strict_date_optional_time"
                },
                {
                  "field": "START_TIME",
                  "format": "strict_date_optional_time"
                },
                {
                  "field": "TIMESTAMP",
                  "format": "strict_date_optional_time"
                }
              ],
              "size": 10000,
              "script_fields": {
                "day_of_week": {
                  "script": {
                    "source": "if (doc['@timestamp'] == null || doc['@timestamp'].size() == 0) {\r\n    return \"0 - Inconnu\";\r\n}\r\n[ \"0 - Inconnu\", \"1 - Lundi\", \"2 - Mardi\", \"3 - Mercredi\", \"4 - Jeudi\", \"5 - Vendredi\", \"6 - Samedi\", \"7 - Dimanche\"][doc['@timestamp'].value.getDayOfWeek()]",
                    "lang": "painless"
                  }
                }
              },
              "stored_fields": [
                "*"
              ],
              "runtime_mappings": {},
              "_source": "false",
              "query": {
                "bool": {
                  "must": [],
                  "filter": [
                    {
                      "bool": {
                        "should": [
                          {
                            "match_phrase": {
                              "host.keyword": hostname
                            }
                          }
                        ],
                        "minimum_should_match": 1
                      }
                    },
                    {
                      "range": {
                        "TIMESTAMP": {
                          "format": "strict_date_optional_time",
                          "gte": gte,
                          "lte": lte
                        }
                      }
                    },
                    {
                      "match_phrase": {
                        "teich": "wave1"
                      }
                    },
                    {
                      "match_phrase": {
                        "type.keyword": "ps_metro2"
                      }
                    }
                  ],
                  "should": [],
                  "must_not": [
                    {
                      "match_phrase": {
                        "user": "oracle"
                      }
                    },
                    {
                      "match_phrase": {
                        "ruser": "zabbix"
                      }
                    },
                    {
                      "match_phrase": {
                        "user": "root"
                      }
                    },
                    {
                      "match_phrase": {
                        "user": "sysux"
                      }
                    },
                    {
                      "match_phrase": {
                        "user": "zabbix"
                      }
                    },
                    {
                      "match_phrase": {
                        "user": "sys"
                      }
                    }
                  ]
                }
              }
            }
    try:
        scroll = '5m'  
        # Connect to Elasticsearch with basic authentication and SSL context
        es = Elasticsearch(hosts=es_full_host , basic_auth=(user, password), verify_certs=False)
        logging.info(f"connexion established: {es}")

        # Perform a search query
        response = es.search(index=index_name, body=query, scroll=scroll)
        logging.info(f"Search retrieved successfully for hostname {hostname}")
       
        sid = response['_scroll_id']
        scroll_size = response['hits']['total']['value']
        logging.info(f"Scroll size: {scroll_size}")

        # Initialize a list to keep track of the results
        results = []

        # Start scrolling
        while scroll_size > 0:
            # Before fetching the next page, add the current batch to our results list
            results += response['hits']['hits']
            
            # Fetch the next page of results
            response = es.scroll(scroll_id=sid, scroll=scroll)
            logging.debug(f"Response: {response}")
            
            # Update the scroll ID and size
            sid = response['_scroll_id']
            scroll_size = len(response['hits']['hits'])
            

        return results

    except Exception as e:
        error_message = "Errot pulling process :" + str(e)
        logging.error(f"Error during request: {error_message}")
        return error_message
    



def get_hostnames():
    
    current_time = datetime.utcnow()

    time_24_hours_ago = current_time - timedelta(days=1)

    lte = current_time.isoformat() + 'Z'
    gte = time_24_hours_ago.isoformat() + 'Z'
    
    # construct the full host URL
    es_full_host = f"{es_scheme}://{es_host}:{es_port}"
    logging.info(f"Querying Elasticsearch at {es_full_host} for hostnames")

    # Elasticsearch query
    query = {
              "track_total_hits": "false",
              "size": 0,
              "version": "true",
              "stored_fields": [
                "*"
              ],
              "runtime_mappings": {},
              "_source": "false",
              "query": {
                "bool": {
                  "must": [],
                  "filter": [
                    {
                      "range": {
                        "TIMESTAMP": {
                          "format": "strict_date_optional_time",
                          "gte": gte,
                          "lte": lte
                        }
                      }
                    },
                    {
                      "match_phrase": {
                        "teich.keyword": "wave1"
                      }
                    },
                  {
                      "match_phrase": {
                        "type.keyword": "mmr_metro"
                      }
                    }
                  ],
                  "should": [],
                  "must_not": []
                }
              },
              "aggs": {
                "hosts": {
                  "terms": {
                    "field": "host.keyword",
                    "size": 10000
                  }
                }
              }
            }

    try:
        # Connect to Elasticsearch with basic authentication and SSL context
        es = Elasticsearch(hosts=es_full_host , basic_auth=(user, password), verify_certs=False)
        logging.info(f"Querying Elasticsearch at {es_full_host} for hostname list")

        # iPerform a search query
        response = es.search(index=index_name, body=query)
        response = response['aggregations']['hosts']['buckets']

        # extract only hostnames form response in a list
        response = [item['key'] for item in response]
        
        #pprint(response)
        logging.info(f"Description retrieved successfully for hostname list")

        return response

    except Exception as e:
        logging.error("Error while querying Elasticsearch", exc_info=True)
        return "Error: " + str(e)
