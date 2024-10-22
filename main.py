from src.rag_application import RAGQABot

# initiate the class
Bot = RAGQABot(
    configuration_path="./resources/config.json"
    )

# Invoke the RAG pipeline

Bot.upload_file()
Bot.create_vector_db()
Bot.qa_on_vector_store()