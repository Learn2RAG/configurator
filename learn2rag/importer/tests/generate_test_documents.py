"""
generate_test_documents.py

Helper script to generate sample documents for testing all supported file types.
This script creates test documents in the tests/data directory.
"""

import io
from pathlib import Path


def create_pdf() -> bytes:
    """Create a minimal valid PDF file."""
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 100 >>
stream
BT
/F1 12 Tf
50 750 Td
(PDF Document Example) Tj
0 -20 Td
(This is a sample PDF file for testing.) Tj
ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000244 00000 n 
0000000395 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
483
%%EOF"""
    return pdf_content


def create_xlsx() -> bytes:
    """Create a minimal valid XLSX file using built-in zip."""
    import zipfile

    # XLSX is a ZIP archive with XML files
    xlsx_buffer = io.BytesIO()
    with zipfile.ZipFile(xlsx_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # [Content_Types].xml
        content_types = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
<Override PartName="/xl/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
</Types>"""
        zf.writestr("[Content_Types].xml", content_types)

        # _rels/.rels
        rels = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
</Relationships>"""
        zf.writestr("_rels/.rels", rels)

        # xl/_rels/workbook.xml.rels
        wb_rels = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>
</Relationships>"""
        zf.writestr("xl/_rels/workbook.xml.rels", wb_rels)

        # xl/workbook.xml
        workbook = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<sheets>
<sheet name="Sheet1" sheetId="1" r:id="rId1" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/>
</sheets>
</workbook>"""
        zf.writestr("xl/workbook.xml", workbook)

        # xl/worksheets/sheet1.xml
        sheet = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<sheetData>
<row r="1"><c r="A1" t="inlineStr"><is><t>Name</t></is></c><c r="B1" t="inlineStr"><is><t>Age</t></is></c></row>
<row r="2"><c r="A2" t="inlineStr"><is><t>Alice</t></is></c><c r="B2" t="n"><v>28</v></c></row>
<row r="3"><c r="A3" t="inlineStr"><is><t>Bob</t></is></c><c r="B3" t="n"><v>34</v></c></row>
</sheetData>
</worksheet>"""
        zf.writestr("xl/worksheets/sheet1.xml", sheet)

        # xl/styles.xml (minimal)
        styles = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<cellXfs count="1"><xf numFmtId="0"/></cellXfs>
</styleSheet>"""
        zf.writestr("xl/styles.xml", styles)

        # xl/theme/theme1.xml (minimal)
        theme = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Office Theme"/>"""
        zf.writestr("xl/theme/theme1.xml", theme)

        # docProps/core.xml
        core_props = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/officeDocument/2006/custom-properties"/>"""
        zf.writestr("docProps/core.xml", core_props)

    return xlsx_buffer.getvalue()


def create_docx() -> bytes:
    """Create a minimal valid DOCX file using built-in zip."""
    import zipfile

    docx_buffer = io.BytesIO()
    with zipfile.ZipFile(docx_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # [Content_Types].xml
        content_types = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
</Types>"""
        zf.writestr("[Content_Types].xml", content_types)

        # _rels/.rels
        rels = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
</Relationships>"""
        zf.writestr("_rels/.rels", rels)

        # word/document.xml
        document = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>
<w:p><w:r><w:t>DOCX Document Example</w:t></w:r></w:p>
<w:p><w:r><w:t>This is a sample DOCX file for testing the loaders.</w:t></w:r></w:p>
<w:p><w:r><w:t>It contains multiple paragraphs and formatted text.</w:t></w:r></w:p>
</w:body>
</w:document>"""
        zf.writestr("word/document.xml", document)

        # docProps/core.xml
        core_props = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/officeDocument/2006/custom-properties"/>"""
        zf.writestr("docProps/core.xml", core_props)

    return docx_buffer.getvalue()


def create_pptx() -> bytes:
    """Create a minimal valid PPTX file using built-in zip."""
    import zipfile

    pptx_buffer = io.BytesIO()
    with zipfile.ZipFile(pptx_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # [Content_Types].xml
        content_types = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
<Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
</Types>"""
        zf.writestr("[Content_Types].xml", content_types)

        # _rels/.rels
        rels = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
</Relationships>"""
        zf.writestr("_rels/.rels", rels)

        # ppt/presentation.xml
        presentation = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<p:sldIdLst>
<p:sldId id="256" r:id="rId2"/>
</p:sldIdLst>
</p:presentation>"""
        zf.writestr("ppt/presentation.xml", presentation)

        # ppt/slides/slide1.xml
        slide = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
<p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name="Title"/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="9144000" cy="6858000"/><a:chOff x="0" y="0"/><a:chExt cx="9144000" cy="6858000"/></a:xfrm></p:grpSpPr>
<p:sp><p:nvSpPr><p:cNvPr id="2" name="Title 1"/></p:nvSpPr><p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:t>PPTX Presentation Example</a:t></a:r></a:p></p:txBody></p:sp>
</p:spTree></p:cSld>
</p:sld>"""
        zf.writestr("ppt/slides/slide1.xml", slide)

        # ppt/_rels/presentation.xml.rels
        pres_rels = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>
</Relationships>"""
        zf.writestr("ppt/_rels/presentation.xml.rels", pres_rels)

        # docProps/core.xml
        core_props = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/officeDocument/2006/custom-properties"/>"""
        zf.writestr("docProps/core.xml", core_props)

    return pptx_buffer.getvalue()


def create_test_documents(data_dir: Path) -> None:
    """Generate all test documents."""
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create PDF
    pdf_path = data_dir / "sample.pdf"
    pdf_path.write_bytes(create_pdf())
    print(f"Created: {pdf_path}")

    # Create XLSX
    xlsx_path = data_dir / "sample.xlsx"
    xlsx_path.write_bytes(create_xlsx())
    print(f"Created: {xlsx_path}")

    # Create DOCX
    docx_path = data_dir / "sample.docx"
    docx_path.write_bytes(create_docx())
    print(f"Created: {docx_path}")

    # Create PPTX
    pptx_path = data_dir / "sample.pptx"
    pptx_path.write_bytes(create_pptx())
    print(f"Created: {pptx_path}")


if __name__ == "__main__":
    data_dir = Path(__file__).parent / "data"
    create_test_documents(data_dir)
    print(f"\nAll test documents created in: {data_dir}")
