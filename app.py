import time
import logging
import requests
import hashlib
import os
import gradio as gr
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from fake_useragent import UserAgent

from langchain_community.document_loaders import WebBaseLoader, YoutubeLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import CharacterTextSplitter


# Global variable to store the retriever state
retriever = None
vectorstore = None 
ids = None
global_is_product_review = "No"

# Load LLM model
model_local = ChatOllama(model="llama3")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def safe_request(url):
    """Fetch the webpage content using a randomized user-agent to simulate different browsers."""
    try:
        ua = UserAgent()
        headers = {'User-Agent': ua.random}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        logging.error(f"Request failed for {url}: {e}")
        return None

def is_youtube_url(url):
    """Check if the given URL is a YouTube URL."""
    return "youtube.com" in url or "youtu.be" in url

def is_valid_url(url):
    """Check if the URL is a valid webpage link and not a mailto, tel, or social media link, and does not contain fragments."""
    parsed = urlparse(url)
    # Check if the URL has a fragment and exclude it if it does
    if parsed.fragment:
        return False
    if parsed.scheme not in ["http", "https"]:
        return False
    if 'mailto:' in url or 'tel:' in url:
        return False
    netloc = parsed.netloc.lower()
    social_media_domains = [
        'facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com',
        'pinterest.com', 'youtube.com', 'tiktok.com', 'snapchat.com'
    ]
    if any(domain in netloc for domain in social_media_domains):
        return False
    return True

def fetch_all_links(url, all_links=None, visited=None):
    """Fetch links from the given URL without depth control but avoiding cycles and unnecessary URL fragments."""
    if visited is None:
        visited = set()
    if all_links is None:
        all_links = set()

    if url in visited:
        logging.info(f'Skipping {url} as it is already visited.')
        return all_links
    
    visited.add(url)
    logging.info(f'Fetching links from: {url}')

    # Use the safe_request function to get the webpage content
    html = safe_request(url)
    if html is None:
        return all_links  # If fetching fails, return the current state of all_links

    try:
        soup = BeautifulSoup(html, 'html.parser')
        base_url = urlparse(url).netloc
        for link in soup.find_all('a', href=True):
            full_link = urljoin(url, link.get('href'))
            if is_valid_url(full_link) and urlparse(full_link).netloc == base_url and full_link not in visited:
                all_links.add(full_link)
                logging.info(f'Fetching new link: {full_link}')
                fetch_all_links(full_link, all_links, visited)
                time.sleep(1)  # Sleep to be polite and avoid hitting rate limits
    except Exception as e:
        logging.error(f"Error processing links from {url}: {e}")

    return all_links

def prepare_embeddings(root_url, is_product_review):
    global retriever, vectorstore, ids, global_is_product_review
    global_is_product_review = is_product_review
    
    # Determine the directory to check or save the processed documents
    persist_directory = './chroma_db'
    # Load embedding models
    embedding_function = embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text')

    # Check if data already exists
    if os.path.exists(persist_directory):
        print(f"Loading data from {persist_directory}")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            collection_name="appSignal",
            embedding_function=embedding_function,
        )
        retriever = vectorstore.as_retriever()
        return "Loaded existing data successfully."
    else:
        print(f"No existing data found for {root_url}, fetching and processing new content.")
        urls_list = fetch_all_links(root_url)
        docs_list = []

        for url in urls_list:
            if is_youtube_url(url):
                docs = YoutubeLoader.from_youtube_url(url, add_video_info=False).load()
                if isinstance(docs, list):
                    docs_list.extend(docs)
                else:
                    docs_list.append(docs)
            else:
                docs = WebBaseLoader(url).load()
                docs_list.extend(docs)
        
        # Process and save the data to disk
        try:
            vectorstore = Chroma.from_documents(
                documents=docs_list,
                collection_name="appSignal",
                embedding=embedding_function,
                persist_directory=persist_directory
            )
            vectorstore.persist()
            retriever = vectorstore.as_retriever()
            print("Processed and saved new data successfully.")
        except TypeError as e:
            print(f"An error occurred while initializing Chroma: {e}")
            # Adding more detailed error handling to pinpoint issues
            print(f"Parameters passed: documents=<{type(docs_list)}>, embedding_function=<{embedding_function}>, persist_directory=<{persist_directory}>")
            return "An error occurred while processing documents. Please check the logs for more details."

    # Proceed with using the data
    if vectorstore and retriever:
        return "The documents have been processed and are ready to be used with the model."
    else:
        return "There was an issue processing the documents. Please check the inputs or try again later."

def generate_learning_material(article_title, article_section):
    global retriever, vectorstore, model_local, global_is_product_review
    # Load LLM model
    model_local = ChatOllama(model="llama3")
    if global_is_product_review == "Yes":
        template = """
            You are tasked with creating an engaging, informative overview that introduces developers to the product explained in the context in a structured manner::
            Context:\n{context}
            - Second Summary: Craft a concise pitch that encapsulates the essence of product in the context, highlighting its core value proposition. This should be clear and compelling enough to grasp the product's purpose instantly.
            - Exploration: Elaborate on what developers can build with product in the context. Provide examples of projects or applications to spark inspiration and showcase the platform's versatility and capabilities.
            - "Hello World" Tutorial: Guide the user through setting up a simple "hello world" project with product in the context. The instructions should be straightforward, ensuring a beginner can follow along and achieve a quick win within 30 minutes. Include necessary code snippets, setup steps, and any relevant configuration details.
            Ensure the material is structured to be clear, concise, and practical, often focusing on real-world applications. Use markdown styling to make the content visually appealing and easy to follow, with titles for each section, highlighted key points, and well-separated sections for readability.
        """ 
    else:
        template = f"""
            You are the best software engineering instructor, technical writer with excellent capability on explaining complex topics in simple way with examples an din a way that everyone can understand.
            You are also Open edX expert with speciality on integrating Open edX with third part systems.
            I need your help in creating an easy to understand, engaging and detailed technical article only based on the following context:
            Context:\n{{context}}
            Help me out to write this technical article, we should explain to the users step by step how to do this implementation with Open edX so they can do it by themselves.
            We should provide detailed code as well.
            The article should be in markdown format.
            The article title is {article_title}.
            Lets start with {article_section}
        """
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever}
        | prompt
        | model_local
        | StrOutputParser()
    )
    response = chain.invoke("")
    return response

def clean_db():
    global ids
    global vectorstore
    try:
        if ids and vectorstore:
            # Assuming `delete` is the correct method and `ids` contains the correct document IDs
            vectorstore.delete(ids)
            ids = []  # Reset the IDs list after successful deletion
            gr.Warning("Removed all processed content from the database.")
        else:
            gr.Warning("Database is empty or not initialized.")
    except Exception as e:
        return f"An error occurred: {str(e)}"

def review_code(code):
    global retriever
    global vectorstore

    # Load LLM model
    model_local = ChatOllama(model="llama3")
    template = """
        You are expert software engineer and code reviewer. Use the context provided and your experties and review my code and provide the following
        this is the Context:\n{context}
        - the result/output of the code
        - any possible bug or issue
        - any possible security volunribility
        - any possible improvments
        The following is the code I need the review for:
        Code:\n{code}
    """ 
    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "code": RunnablePassthrough()}
    )
    prompt = ChatPromptTemplate.from_template(template)
    chain = setup_and_retrieval | prompt | model_local | StrOutputParser()
    response = chain.invoke(code)
    return response

# Define the main function that will create the tabs
def main():
    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="emerald", text_size=gr.themes.sizes.text_lg, font=["Arial", "sans-serif"]),
        title="The Learning Algorithm",
        ) as interface:
        with gr.Tab("Step 1"):
            gr.Markdown("""
            ## Step 1: Prepare Your Learning Content
            This tool is designed to help you absorb and practice programming concepts more effectively. 
            Just provide the URL of an online programming resource, like the latest Python documentation, and the system will analyze the content and embed it into a language model. 
            This process ensures that the responses and practice questions you receive are directly based on the material, providing an accurate and insightful learning experience without any fabricated information. 

            ### Here's how to get started:
            1. **Enter URLs**: Place each URL on a new line in the text box below. These should link to online programming content you want to learn from, such as Python's official documentation for a new release.
            1. **Process URLs**: After entering the URLs, click the 'Process URLs' button. Our system will fetch the content from the URLs, analyze it, and store it in our database for further use.

            ### After processing, proceed to Step 2 where you'll interact with the model to enhance your learning experience.
            ---
            """)
            url_input = gr.Textbox(label="Enter URLs separated by new lines", lines=4)
            embedding_status_output = gr.Textbox(label="Status")
            is_product_review = gr.Radio(choices=["Yes", "No"], label="Do you need Product Overview ?")
            with gr.Row():
                clean_db_button = gr.Button(
                    value="Truncate Database",
                    elem_id="clean_db_button", 
                    variant="stop"
                ).click(clean_db)
                gr.ClearButton(
                    components=[url_input, embedding_status_output],
                    elem_id="clear_button", 
                    variant="secondary"
                )
                prepare_button = gr.Button(
                    value="Process URLs",
                    elem_id="process_button", 
                    variant="primary"
                )
                prepare_button.click(
                    prepare_embeddings,
                    inputs=[url_input, is_product_review],
                    outputs=embedding_status_output
                )
            gr.Examples(
                examples=[
                    ["https://docs.python.org/3/whatsnew/3.12.html"], 
                    ["https://www.youtube.com/watch?v=-BOBedcjySI"],
                ],
                inputs=url_input
            )
        with gr.Tab("Step 2"):
            # Markdown instructions for Step 2
            gr.Markdown("""
            ## Step 2: Interact with the Model
            ---
            """)
            article_title = gr.Textbox(placeholder="What is the article's title ?")
            article_section = gr.Textbox(placeholder="Which section you are working on ?", lines=10)
            learning_material = gr.Markdown(
                label="Learning Content",
                value="# Learning Material"
            )
            clear_button = gr.ClearButton(components=[learning_material])
            submit_button = gr.Button(
                value="Generate Learning Material",
                variant="primary"
            ).click(generate_learning_material, inputs=[article_title, article_section], outputs=learning_material)
    interface.queue(max_size=10)
    return interface

# Launch the interface
if __name__ == "__main__":
    main().launch(show_api=True, root_path="/", share=True)