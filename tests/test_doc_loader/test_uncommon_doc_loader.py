import pytest
import tempfile
import json
from pathlib import Path
from rag_src.doc_loader import UncommonDocLoader

def write_file(path: Path, content: str):
    path.write_text(content, encoding="utf-8")

def test_load_csv_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        file = tmp_path / "sample.csv"
        file.write_text("name,age\nAlice,30\nBob,25", encoding="utf-8")

        loader = UncommonDocLoader(file)
        result = loader.load()

        assert "Alice" in result[0]
        assert "Bob" in result[0]

def test_load_json_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        file = tmp_path / "data.json"
        json.dump({"key": "value", "list": [1, 2]}, file.open("w", encoding="utf-8"))

        loader = UncommonDocLoader(file)
        result = loader.load()

        assert '"key": "value"' in result[0]
        assert '"list": [' in result[0]

def test_load_xml_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        file = tmp_path / "doc.xml"
        xml_content = "<root><item>Hello</item><item>World</item></root>"
        write_file(file, xml_content)

        loader = UncommonDocLoader(file)
        result = loader.load()

        assert "Hello" in result[0]
        assert "World" in result[0]

def test_load_rst_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        file = tmp_path / "readme.rst"
        content = "This is a reStructuredText file."
        write_file(file, content)

        loader = UncommonDocLoader(file)
        result = loader.load()

        assert "reStructuredText" in result[0]

def test_load_tex_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        file = tmp_path / "notes.tex"
        content = r"\section{Intro} This is LaTeX content."
        write_file(file, content)

        loader = UncommonDocLoader(file)
        result = loader.load()

        assert "LaTeX content" in result[0]

@pytest.mark.skipif("ebooklib" not in globals(), reason="ebooklib not installed")
def test_load_epub_file():
    import ebooklib
    from ebooklib import epub
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        file = tmp_path / "book.epub"

        # Create a basic EPUB
        book = epub.EpubBook()
        book.set_identifier("id123")
        book.set_title("Test Book")
        book.set_language("en")
        c1 = epub.EpubHtml(title="Intro", file_name="intro.xhtml", lang="en")
        c1.content = "<h1>Hello</h1><p>This is EPUB</p>"
        book.add_item(c1)
        book.toc = (epub.Link("intro.xhtml", "Intro", "intro"),)
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.spine = ["nav", c1]
        epub.write_epub(str(file), book)

        loader = UncommonDocLoader(file)
        result = loader.load()

        assert "This is EPUB" in result[0]
