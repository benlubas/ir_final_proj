from typing import List, Tuple
import os
import tantivy

from document_parser import DocumentParser


class TantivySearch:
    index: tantivy.Index
    schema: tantivy.Schema
    id: str = "default"

    def __init__(self, id) -> None:
        self.id = id

        # Declaring our schema.
        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("title", stored=True)
        schema_builder.add_text_field("content", stored=True)
        schema_builder.add_text_field("id", stored=True)
        schema_builder.add_text_field("bias_text", stored=True)
        schema_builder.add_text_field("authors", stored=True)
        schema_builder.add_text_field("date", stored=True)
        schema_builder.add_text_field("source", stored=True)
        schema_builder.add_text_field("topic", stored=True)
        schema_builder.add_text_field("url", stored=True, tokenizer_name="raw")

        self.schema = schema_builder.build()

        # Creating our index
        os.makedirs(f"./data/tantivy/{self.id}", exist_ok=True)
        self.index = tantivy.Index(self.schema, path=f"./data/tantivy/{self.id}")

    def add_documents(self, doc_parser: DocumentParser):
        """Adds all documents from the given DocumentParser to the index, this has to be done once,
        as the index is persisted to disk."""
        doc_parser.add_tanivity_documents(self.index.writer())

    def query(self, query: str) -> List[Tuple[str, float]]:
        """Returns a tantivy.Searcher object for the given query
        args:
            query: string query to search for
        returns: a list of Document IDs
        """
        self.index.reload()
        searcher = self.index.searcher()

        query = self.index.parse_query(query, ["title", "topic", "content"])
        return [
            (searcher.doc(address), score)
            for score, address in searcher.search(query, 1000).hits[:1000]
        ]
