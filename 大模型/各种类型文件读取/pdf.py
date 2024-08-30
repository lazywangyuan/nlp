import pdfplumber


def read_pdf(flag):
    # 打开PDF文件
    with pdfplumber.open(path) as pdf:
        # 遍历每一页
        docx_span = '\n'.join([page.extract_text() for page in pdf.pages])
        flag = 1
    return docx_span, flag

