import gradio as gr
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
model_local = ChatOllama(model="mistral")

def is_youtube_url(url):
    """Check if the given URL is a YouTube URL."""
    return "youtube.com" in url or "youtu.be" in url

def prepare_embeddings(urls, is_product_review):    
    global retriever, vectorstore, ids, global_is_product_review
    global_is_product_review = is_product_review
    # Load embedding models
    embedding = embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text')

    urls_list = urls.split("\n")
    docs_list = []  # This will aggregate all documents from both web and YouTube sources

    for url in urls_list:
        try:
            if is_youtube_url(url):
                docs = YoutubeLoader.from_youtube_url(url, add_video_info=False).load()
                # Assuming docs is a list of transcripts or a single transcript string
                if isinstance(docs, list):
                    docs_list.extend(docs)
                else:
                    docs_list.append(docs)
            else:
                docs = WebBaseLoader(url).load()
                # Assuming docs is a list of document contents
                docs_list.extend(docs)
        except Exception as e:
            gr.Error(f"Failed to load content at {url}: {e}")

    # Process each document content directly
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs_list)

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="KnowledgeVectors",
        embedding=embedding,
    )
    retriever = vectorstore.as_retriever()

    # Check if the vectorstore and retriever have been successfully created
    if vectorstore and retriever:
        gr.Info('The documents have been processed and are ready to be used with the model.')
        return "The documents and videos have been processed and are ready for your queries."
    else:
        gr.Error("There was an error. Please check the inputs or try again later.")
        return "There was an issue processing the documents and videos. Please check the inputs or try again later."

def generate_learning_material():
    global retriever, vectorstore, model_local, global_is_product_review
    # Load LLM model
    model_local = ChatOllama(model="mistral")
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
        template = """
            Assist in learning from a technical document by performing the following tasks in markdown format, only based on the following context:
            Context:\n{context}
            - Summarize: Summarize the main topics covered in the context. Aim for simple explanation, engaging and clarity and brevity to capture the essence of the content.
            - Explain with Code: Select key concepts from the document and demonstrate each with a relevant code snippet. Provide a brief explanation for how each snippet exemplifies the concept, ensuring the explanation enhances understanding.
            - Verify Understanding:
                - Craft a few multiple-choice questions that quiz on the document's main points.
                - Include true/false questions that address key insights, ensuring to back up answers with concise explanations.
            - Engage with Hands-on Coding:
                - Propose coding challenges that require applying knowledge from the document. Challenges could range from debugging exercises to developing new code based on the concepts learned.
                - For each challenge, include context and specify what success looks like. If appropriate, provide starter code and a solution for self-verification.
            The ultimate aim is to create a learning tool that ensures comprehension of the context content through a mix of summarization, theoretical questions, and practical, hands-on coding exercises. This approach should be being clear, concise, explain as simple as possible, and practical, often focusing on real-world applications of programming concepts.
            Important: Make sure you provide answers to all the questions, answers should be collapsable and default should be hidden so users uncollapse it to see the answer.
            Make sure use markdown styling to make the content visually appealing and easy to follow. make titles bigger, highlight important parts. have good amount of separation and spacing between sections.
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
    model_local = ChatOllama(model="mistral")
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
                    ["https://www.youtube.com/watch?v=NkVSSMt-h3Y"],
                    ["https://strapi.io"]
                ],
                inputs=url_input
            )
        with gr.Tab("Step 2"):
            # Markdown instructions for Step 2
            gr.Markdown("""
            ## Step 2: Interact with the Model
            Now that your data is ready, let's generate our learning material:
                - This is a **Retrieval-Augmented Generation (RAG)** application. It does not make up answers, and strives to provide accurate and reliable information.

            ---
            """)

            with gr.Row():
                with gr.Column():
                    learning_material = gr.Markdown(
                        label="Learning Content",
                        value="# Learning Material"
                    )
                    clear_button = gr.ClearButton(components=[learning_material])
                    submit_button = gr.Button(
                        value="Generate Learning Material",
                        variant="primary"
                    ).click(generate_learning_material, outputs=learning_material)
                with gr.Column():
                    playground = gr.Code(
                        label="Playground",
                        language="python",
                        interactive=True,
                        show_label=True,
                    )
                    chat_box = gr.Markdown(
                        label="Chat History",
                        show_label=True,
                    )
                    review_code_button = gr.Button(
                        value="Review My Code",
                        variant="primary",
                        elem_id="review_code_button"
                    ).click(review_code, inputs=playground, outputs=chat_box)
    interface.queue(max_size=10)
    return interface

# Launch the interface
if __name__ == "__main__":
    main().launch(show_api=False)