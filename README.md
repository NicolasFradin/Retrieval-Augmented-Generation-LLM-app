# Retrieval-Augmented-Generation-LLM-app
Build a RAG LLM app using Chainlit to run a Medical Chatbot using Llama2 

# Steps 

- Create a HuggingFace account 
- Download the quantized models with the script
- pip install -r requirements.txt
- Create vector database with the script 
- Run Chainlit with `chainlit run src/model.py -w`






# Custom the chatbot

This app can be adapted to your own corpus by changing pdf files in the `data/` folder. Please choose the appropriate 
`chunk_size` and `chunk_overlap` vectorstore parameters to fit your needs.  