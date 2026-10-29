import unittest

from backend.ingestion.chunker import chunk_pages
from backend.ingestion.parsers import parse_document


class IngestionTests(unittest.TestCase):
    def test_csv_becomes_searchable_text(self):
        pages = parse_document("people.csv", b"name,team\nAda,Research\n")
        self.assertEqual(pages[0]["section"], "CSV")
        self.assertIn("Ada | Research", pages[0]["text"])

    def test_chunk_provenance_is_retained(self):
        chunks = chunk_pages([{"text": "hello world " * 30, "page": 3, "section": "Intro"}], chunk_size=100, overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(item["page"] == 3 and item["section"] == "Intro" for item in chunks))

    def test_empty_content_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "No text"):
            parse_document("empty.txt", b" \n")


if __name__ == "__main__":
    unittest.main()
