import chromadb
import pandas as pd
import streamlit as st
import logging
import click

pd.set_option('display.max_columns', 4)


@click.command()
@click.option(
    "--host",
    envvar="DB_HOST",
    help="Chroma database host",
    default="localhost",
)
@click.option(
    "--port",
    envvar="DB_PORT",
    help="Chroma database port",
    default="localhost",
)
def view_collections(host, port):

    logging.info("Opening database: %s on port: %s", host, port)

    st.markdown("### DB Path: " + str(host) + ':' + str(port))

    chroma_client = chromadb.HttpClient(host=host,port=port)

    # This might take a while in the first execution if Chroma wants to download the embedding transformer

    st.header("Collections")

    for collection in chroma_client.list_collections():
        data = collection.get(include=['embeddings', 'documents', 'metadatas'])

        df = pd.DataFrame.from_dict(data)
        st.markdown("### Collection: **%s**" % collection.name)
        st.dataframe(df)

        st.markdown("Total documents in the collection: %s" % len(df))

if __name__ == "__main__":
    view_collections()
