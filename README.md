# Retrieval-Augmented-Generation-LLM-app
RAG LLM app using Chainlit to run a Medical Chatbot using Llama2 on CPU. 
The chatbot is made 

# Instructions

1. Create a HuggingFace account 
2. Download the following quantized models by running the script `scripts/download_models.py`
3. Create a virtual environment and run `pip install -r requirements.txt`
4. Build the app with `docker compose build`
5. Run the app with `docker compose up -d`

Note: Set the environment variable `RESET_DB` to `true` to automatically load medical documents from the Gale Encyclopedia.
You can check the upload progress through the little Streamlit app showing the Chroma DB documents with the command `streamlit run chroma-viewer/viewer.py`


# Medium link
Find the related Medium article of the repo here -> 
