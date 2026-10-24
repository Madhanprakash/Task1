import json
import unittest

from rag import (
    document_id,
    extract_document_text,
    normalize_entity,
    parse_graph_facts,
    split_text,
)


class DocumentProcessingTests(unittest.TestCase):
    def test_text_upload_decodes_utf8_bom(self) -> None:
        self.assertEqual(extract_document_text("notes.txt", b"\xef\xbb\xbfhello"), "hello")

    def test_unsupported_file_type_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported file type"):
            extract_document_text("notes.csv", b"a,b")

    def test_split_text_preserves_content_and_bounds_chunks(self) -> None:
        text = " ".join(f"word{i}" for i in range(100))
        chunks = split_text(text, chunk_size=100, overlap=20)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))
        self.assertTrue(all(chunk in text for chunk in chunks))

    def test_split_text_rejects_invalid_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "Chunk size"):
            split_text("some text", chunk_size=10, overlap=10)

    def test_document_id_is_stable(self) -> None:
        self.assertEqual(document_id("same text"), document_id("same text"))
        self.assertNotEqual(document_id("same text"), document_id("different text"))


class KnowledgeGraphParsingTests(unittest.TestCase):
    def test_graph_facts_are_normalized_and_relations_validated(self) -> None:
        response = json.dumps(
            {
                "entities": [{"name": " Ada  Lovelace "}, {"name": "Charles Babbage"}],
                "relations": [
                    {
                        "source": "ada lovelace",
                        "predicate": "worked with",
                        "target": "Charles Babbage",
                    },
                    {
                        "source": "unknown",
                        "predicate": "invented",
                        "target": "Ada Lovelace",
                    },
                ],
            }
        )

        facts = parse_graph_facts(response)

        self.assertEqual(normalize_entity(" Ada  Lovelace "), "ada lovelace")
        self.assertEqual(len(facts["entities"]), 2)
        self.assertEqual(
            facts["relations"],
            [{"source": "ada lovelace", "predicate": "worked with", "target": "charles babbage"}],
        )

    def test_graph_response_requires_list_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid knowledge-graph structure"):
            parse_graph_facts('{"entities": {}, "relations": []}')


if __name__ == "__main__":
    unittest.main()
