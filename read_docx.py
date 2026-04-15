import zipfile
import xml.etree.ElementTree as ET
import sys

def get_docx_text(path):
    document = zipfile.ZipFile(path)
    xml_content = document.read('word/document.xml')
    document.close()
    
    tree = ET.XML(xml_content)
    # The namespace prefix used by MS Word for OOXML
    WORD_NAMESPACE = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    
    # We will simply extract all the text nodes
    paragraphs = []
    
    for paragraph in tree.iter(WORD_NAMESPACE + 'p'):
        texts = [node.text
                 for node in paragraph.iter(WORD_NAMESPACE + 't')
                 if node.text]
        if texts:
            paragraphs.append(''.join(texts))

    return '\n'.join(paragraphs)

if __name__ == "__main__":
    text = get_docx_text(sys.argv[1])
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(text)
