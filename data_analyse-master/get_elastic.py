from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import ssl
from ssl import create_default_context
import json
import requests
import os
import re
import datetime
import pytz
import urllib3
import pprint
import argparse
from pprint import pprint
from datetime import datetime, timezone
urllib3.disable_warnings()




# Load environment variables from .env file
load_dotenv()


def get_metrics(hostname, gte, lte):

    es_host = os.getenv('ES_HOST', '')
    es_port = int(os.getenv('ES_PORT', 9250))
    es_scheme = os.getenv('SCHEME', 'https')
    index_name = os.getenv('INDEX', '')
    user = os.getenv('username')
    password = os.getenv('pass')

    # Define your headers
    headers = {
        'Content-Type': 'application/json'
    }
    # construct the full host URL
    es_full_host = f"{es_scheme}://{es_host}:{es_port}"

    # Elasticsearch query
    query = {
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
        # Connect to Elasticsearch with basic authentication and SSL context
        es = Elasticsearch(hosts=es_full_host , basic_auth=(user, password), verify_certs=False)
        
        # Perform a search query
        response = es.search(index=index_name, body=query)
        response = response['hits']['hits']
        #print(query) 
        return response

    except Exception as e:
        print("Error:", str(e))
        return str(e)
    




def get_process(hostname, gte, lte):

    es_host = os.getenv('ES_HOST', '')
    es_port = int(os.getenv('ES_PORT', 9250))
    es_scheme = os.getenv('SCHEME', 'https')
    index_name = os.getenv('INDEX', '')
    user = os.getenv('username')
    password = os.getenv('pass')

    # Define your headers
    headers = {
        'Content-Type': 'application/json'
    }
    # construct the full host URL
    es_full_host = f"{es_scheme}://{es_host}:{es_port}"

    # Elasticsearch query
    query = {
              "track_total_hits": "false",
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
                    }
                  ]
                }
              }
            }
    try:
        # Connect to Elasticsearch with basic authentication and SSL context
        es = Elasticsearch(hosts=es_full_host , basic_auth=(user, password), verify_certs=False)
        
        # Perform a search query
        response = es.search(index=index_name, body=query)
        response = response['hits']['hits']
        print(query) 
        return response

    except Exception as e:
        print("Error:", str(e))
        return str(e)
    

