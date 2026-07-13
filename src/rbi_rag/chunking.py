from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_pages(pages, *, chunk_size: int, chunk_overlap: int):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    ).split_documents(pages)

