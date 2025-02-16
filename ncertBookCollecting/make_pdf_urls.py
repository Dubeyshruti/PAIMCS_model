from pandas import read_csv, DataFrame
booklist = read_csv('ncertBooklist.csv')
pdf_urls = {'pdfName': list(), 'downloadLink': list()}
for index, book in booklist.iterrows():
    bookName = '_'.join([str(book['class']), book.subject, book.lang_char, book.title]).strip()
    for chapter in range(1, book.chapters+1):
        pdfName = "{}{}.pdf".format(bookName, chapter)
        if chapter < 10:
            downloadLink = f"https://ncert.nic.in/textbook/pdf/{book.preName}{chapter}.pdf"
        else:
            downloadLink = f"https://ncert.nic.in/textbook/pdf/{book.preName[:-1]}{chapter}.pdf"
        pdf_urls['pdfName'].append(pdfName); pdf_urls["downloadLink"].append(downloadLink)
DataFrame(pdf_urls).to_csv("pdf_urls.csv", index=False)
