import pytest
import tempfile
from pathlib import Path
from rag_src.doc_loader import DefaultDocLoader

def create_temp_txt_file(dir_path: Path, filename: str, content: str) -> Path:
    file_path = dir_path / filename
    file_path.write_text(content, encoding="utf-8")
    return file_path

def test_load_single_txt_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        file_path = create_temp_txt_file(tmp_path, "test.txt", "Hello world!")

        loader = DefaultDocLoader(file_path)
        contents = loader.load()

        assert len(contents) == 1
        assert contents[0] == "Hello world!"

def test_load_multiple_txt_files_from_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        create_temp_txt_file(tmp_path, "file1.txt", "File 1 content.")
        create_temp_txt_file(tmp_path, "file2.txt", "File 2 content.")
        create_temp_txt_file(tmp_path, "not_included.md", "# Markdown")

        loader = DefaultDocLoader(tmp_path)
        contents = loader.load()

        assert len(contents) == 2
        assert any("File 1 content." in doc for doc in contents)
        assert any("File 2 content." in doc for doc in contents)
        assert all(".md" not in doc for doc in contents)

def test_invalid_path_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        invalid_file = tmp_path / "not_text.md"
        invalid_file.write_text("Should not load", encoding="utf-8")

        loader = DefaultDocLoader(invalid_file)
        with pytest.raises(ValueError, match="Invalid path"):
            loader.load()

def test_empty_txt_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        file_path = create_temp_txt_file(tmp_path, "empty.txt", "")

        loader = DefaultDocLoader(file_path)
        contents = loader.load()

        assert len(contents) == 1
        assert contents[0] == ""
