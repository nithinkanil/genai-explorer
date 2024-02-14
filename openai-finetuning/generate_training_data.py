from pathlib import Path
from llama_index.core.extractors import (
    QuestionsAnsweredExtractor,
)
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import download_loader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import VectorStoreIndex
import json


def read_data_from_pdf(file_path):
    """
    Read PDF from the given path and generate llama index object
    :param file_path: Path of the PDF
    :return: Llama index document object
    """
    PDFReader = download_loader("PDFReader")
    loader = PDFReader()
    documents = loader.load_data(file=Path(file_path))
    return documents


def store_generated_question_answer(question, answer, path):
    """
    Store the generated question and answers to a text file
    :param question: Generated Question
    :param answer: Generated Answer
    :param path: Path of the output file
    :return: None
    """
    with open(path, "a") as f:
        f.write(
            json.dumps(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": "Dolly is a chatbot who can answer questions about Indian national policy",
                        },
                        {"role": "user", "content": question},
                        {
                            "role": "assistant",
                            "content": answer,
                        },
                    ]
                }
            )+"\n"
        )


def generate_question_and_answers(documents, path):
    """
    Genarate question and answer from the given document
    :param documents: Text data document
    :param path: Path of the output file
    :return: None
    """
    # Logic for splitting the text to small nodes
    text_splitter = TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=128)
    # Extracting 10 question from each node
    qa_extractor = QuestionsAnsweredExtractor(questions=10)
    pipeline = IngestionPipeline(transformations=[text_splitter, qa_extractor])

    nodes = pipeline.run(
        documents=documents,
        in_place=True,
        show_progress=True,
    )
    # Creating a vector index for generating the answer
    index = VectorStoreIndex(documents)
    query_engine = index.as_query_engine(similarity_top_k=2)
    for node in nodes:
        for question in node.metadata["questions_this_excerpt_can_answer"].split("\n"):
            # Quering the index and generating the answer
            response = query_engine.query(question)
            store_generated_question_answer(question, response.response, path)


if __name__ == "__main__":
    # Read data from pdf and generate llama index document object
    data = read_data_from_pdf("files/India - national health policy.pdf")
    # Generating question and answers from the text
    generate_question_and_answers(data, "files/q_a.txt")
