import fitz
import pytest

from app.services.pdf_service import PDFExtractionError, PDFService


def test_extract_text_by_page_valid_pdf(tmp_path):
    pdf_path = tmp_path / "valid.pdf"

    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "Page one content")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Page two content")
    doc.save(pdf_path)
    doc.close()

    service = PDFService()
    pages = service.extract_text_by_page(str(pdf_path))

    assert len(pages) == 2
    assert pages[0]["page_number"] == 1
    assert "Page one content" in pages[0]["text"]
    assert pages[1]["page_number"] == 2
    assert "Page two content" in pages[1]["text"]


def test_extract_text_by_page_invalid_pdf(tmp_path):
    invalid_pdf_path = tmp_path / "invalid.pdf"
    invalid_pdf_path.write_bytes(b"this is not a valid pdf binary")

    service = PDFService()

    with pytest.raises(PDFExtractionError):
        service.extract_text_by_page(str(invalid_pdf_path))
