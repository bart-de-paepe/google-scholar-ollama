import re

import bs4
import ollama
from bson import ObjectId

from scrapegraphai.graphs import SmartScraperGraph

from app.src.domain.email_body import EmailBody
from app.src.domain.search_result import SearchResult
from app.src.services.db_service import DBService
from app.src.services.logging_service import LoggingService
from app.src.shared.helper import undo_escape_double_quotes

"""
class Serp(BaseModel):
    title: str
    original_url: str
    authors: str
    year_of_publication: int
    journal_name: str
    snippet: str
"""

graph_config = {
    "llm": {
        "model": "ollama/gemma3:27b",  # Specifies the large language model to use
        "temperature": 0,          # Temperature controls randomness; 0 makes it deterministic
        "format": "json",          # Output format is set to JSON
        "base_url": "http://localhost:11434",  # Base URL where the Ollama server is running
    },
    "embeddings": {
        "model": "ollama/nomic-embed-text",  # Specifies the embedding model to use
        "temperature": 0,                    # Keeps the generation deterministic
        "base_url": "http://localhost:11434",  # Base URL for the embeddings model server
    },
    "verbose": True,  # Enables verbose output for more detailed log information
}

class ParseService:
    def __init__(self, db_service: DBService, logging_service: LoggingService):
        self.db_service = db_service
        self.logging_service = logging_service

    # query all the unprocessed _id's
    def get_unprocessed_ids(self):
        where = {"is_processed": False, "is_spam": False}
        what = {"_id": 1}
        self.db_service.set_collection("emails")
        unprocessed_ids = self.db_service.select_what_where(what, where)
        return unprocessed_ids

    # for every _id get the corresponding document body
    def get_body(self, email_id):
        where = {"_id": email_id}
        what = {"body": 1, "_id": 0}
        self.db_service.set_collection("emails")
        body_cursor = self.db_service.select_what_where(what, where)
        body = body_cursor.next()
        email_body = EmailBody(body=body['body']['text_html'])
        body_cursor.close()
        return email_body

    """
        <h3 style="font-weight:normal;margin:0;font-size:17px;line-height:20px;">
            <span style="font-size:11px;font-weight:bold;color:#1a0dab;vertical-align:2px">[HTML]</span> 
            <a href="https://scholar.google.com/scholar_url?url=https://www.nature.com/articles/s41598-025-88482-7&amp;hl=nl&amp;sa=X&amp;d=1565152685938670113&amp;ei=_kqpZ4uAD5iA6rQPtLi4-AQ&amp;scisig=AFWwaeYx4eCOtKIyv7HLoYObbtsW&amp;oi=scholaralrt&amp;hist=uSV2duYAAAAJ:1031754403081217048:AFWwaeadJUTxUhknCeqfHAKi7i4u&amp;html=&amp;pos=0&amp;folt=kw-top" class="gse_alrt_title" style="font-size:17px;color:#1a0dab;line-height:22px">
                Evaluation of 3D seed structure and cellular <b>traits </b>in-situ using X-ray microscopy
            </a>
        </h3>
        <div style="color:#006621;line-height:18px">
            M Griffiths, B Gautam, C Lebow, K Duncan, X Ding…&nbsp;- Scientific Reports, 2025
        </div>
        <div class="gse_alrt_sni" style="line-height:17px">Phenotyping methods for seed morphology are mostly limited to two-dimensional <br>
                imaging or manual measures. In this study, we present a novel seed phenotyping <br>
                approach utilizing lab-based X-ray microscopy (XRM) to characterize 3D seed&nbsp;…
        </div>
    """
    def parse_body(self, email_id, email_body):
        parse_log_message = ""
        body_text = email_body.text_html
        # undo escaping the double quotes
        body_text = undo_escape_double_quotes(body_text)

        body_text = re.sub(r'<head.*?>.*?</head>', '', body_text, flags=re.DOTALL)
        # Remove all occurrences of content between <script> and </script>
        body_text = re.sub(r'<script.*?>.*?</script>', '', body_text, flags=re.DOTALL)
        # Remove all occurrences of content between <style> and </style>
        body_text = re.sub(r'<style.*?>.*?</style>', '', body_text, flags=re.DOTALL)
        # Chat Completion API from OpenAI


        smart_scraper_graph = SmartScraperGraph(
            prompt="You are an AI assistant specialized in processing Google Scholar search engine result pages and returning structured JSON data. Always provide your response as valid, well-formatted JSON without any additional text or comments. Focus on extracting and organizing the most relevant information from Google Scholar search engine result pages, including title, original url, authors, name of journal, year of publication, snippet.",  # The AI prompt specifies what to extract
            source=body_text,  # URL of the website from which to scrape data
            config=graph_config,  # Uses predefined configuration settings
        )

        # Execute the scraping process
        result = smart_scraper_graph.run()

        # Print the results to the console
        print(result)

        """
        # Initialize model and messages
        model = 'llama3.1'

        # Revised system message focused on structured JSON output
        system_message = {
            'role': 'system',
            'content': 'You are an AI assistant specialized in processing Google Scholar search engine result pages and returning structured JSON data. Always provide your response as valid, well-formatted JSON without any additional text or comments. Focus on extracting and organizing the most relevant information from Google Scholar search engine result pages, including title, original url, authors, name of journal, year of publication, snippet.'
        }

        # User message requesting the scraping of content
        user_message = {
            'role': 'user',
            'content': body_text
        }

        # Initialize conversation with the system message and user query
        messages = [system_message, user_message]

        # First API call: Send the query and function description to the model
        response = ollama.chat(
            model=model,
            messages=messages,
            tools=[
                {
                    'type': 'function',
                    'function': {
                        'name': 'parse_data',
                        'description': 'Scrapes the content of a Google Scholar search engine result page and returns the structured JSON object with titles, original urls, authors, names of journal, years of publication, snippets.',
                        "parameters": {
                            'type': 'object',
                            'properties': {
                                'data': {
                                    'type': 'array',
                                    'items': {
                                        'type': 'object',
                                        'properties': {
                                            'title': {'type': 'string'},
                                            'original_url': {'type': 'string'},
                                            'authors': {'type': 'string'},
                                            'year_of_publication': {'type': 'integer'},
                                            'journal_name': {'type': 'string'},
                                            'snippet': {'type': 'string'}
                                        }
                                    }
                                }
                            }
                        }
                    },
                },
            ],
            tool_choice={
                "type": "function",
                "function": {"name": "parse_data"}
            }
        )

        # Append the model's response to the existing messages
        messages.append(response['message'])

        # Check if the model decided to use the provided function
        if not response['message'].get('tool_calls'):
            print("The model didn't use the function. Its response was:")
            print(response['message']['content'])
        else:
            print("Process function calls made by the model")
            print(response['message']['content'])

        """
        # Calling the data results

        """
        serp = self.extract_serp(body_text)
        print(serp)
        """

        """
        argument_dict = json.loads(argument_str)
        data = argument_dict['data']
        """

        # Print in a nice format
        """
        for result in data:
            search_result = SearchResult(result['title'], result["authors"], result["journal_name"], result["year_of_publication"], result["snippet"], result["original_url"])
            db_search_result_id = self.store_body_content(email_id, search_result)
            self.logging_service.logger.debug(f'search result id: {db_search_result_id} parsed and stored in database')
            print(result['title'])
            print(result["authors"] or '')
            print(result["journal_name"] or '')
            print(result["year_of_publication"] or '')
            print(result["snippet"] or '')
            print(result["original_url"] or '')
            print('---')
        """

    """
    @llm.call("ollama", "llama3.1:8b", response_model=Serp)
    def extract_serp(self, serp: str) -> str:
        return f"Extract {serp}"
    """

    def store_body_content(self, email_id, search_result: SearchResult):
        search_result.log_message = "Search result parsed successfully."
        if(search_result.media_type is not None):
            post = {
                "created_at": search_result.get_created_at_formatted(),
                "updated_at": search_result.get_updated_at_formatted(),
                "email": ObjectId(email_id),
                "title": search_result.title,
                "author": search_result.author,
                "publisher": search_result.publisher,
                "year": search_result.date,
                "text": search_result.text,
                "link": {
                    "url": search_result.link.url,
                },
                "media_type": search_result.media_type,
                "log_message": search_result.log_message,
                "is_processed": search_result.is_processed
            }
        else:
            post = {
                "created_at": search_result.get_created_at_formatted(),
                "updated_at": search_result.get_updated_at_formatted(),
                "email": ObjectId(email_id),
                "title": search_result.title,
                "author": search_result.author,
                "publisher": search_result.publisher,
                "year": search_result.date,
                "text": search_result.text,
                "link": {
                    "url": search_result.link.url,
                },
                "log_message": search_result.log_message,
                "is_processed": search_result.is_processed
            }
        self.db_service.set_collection("search_results")
        post_id = self.db_service.insert_one(post)
        return post_id

    def update_search_result(self, search_result_update_what, search_result_update_where):
        self.db_service.set_collection("search_results")
        result = self.db_service.update_one_what_where(search_result_update_what, search_result_update_where)


    def get_current_search_result(self, search_result_id):
        self.db_service.set_collection("search_results")
        result = self.db_service.select_one(search_result_id)
        if 'media_type' in result:
            current_search_result = SearchResult(result["title"], result["author"], result["publisher"], result["year"], result["text"], result["link"]["url"], result["media_type"])
        else:
            current_search_result = SearchResult(result["title"], result["author"], result["publisher"], result["year"],
                                                 result["text"], result["link"]["url"])
        return current_search_result

    def raise_google_scholar_format(self, email_id, item, message):
        log_message = message + item
        is_parsed = True
        is_google_scholar_format = False
        raise IndexError(email_id, log_message, is_parsed,
                         is_google_scholar_format)

