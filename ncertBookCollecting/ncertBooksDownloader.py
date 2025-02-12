from requests import get
from pandas import read_csv
from os import makedirs
save_dir = "ncertBooks"
makedirs(save_dir, exist_ok=True)
preUrl = 'https://ncert.nic.in/textbook/pdf/'
postUrl = '.pdf'
booklist = read_csv("ncertBooklist.csv")
for index, book  in booklist.iterrows():
    bookName = "_".join([str(book['class']), book.subject, book.lang_char, book.title]).strip()
    for chapter in range(1, book.chapters+1):
        pdfName = "{}{}.pdf".format(bookName, chapter)
        if chapter < 10:
            downloadLink = f'{preUrl}{book.preName}{chapter}{postUrl}'
        else:
            downloadLink = f'{preUrl}{book.preName[:-1]}{chapter}{postUrl}'
        response = get(downloadLink, headers = {'Referrer': 'https://www.duckduckgo.com'})
        if response.status_code == 200:
            with open(f"{save_dir}/{pdfName}", 'wb') as f:
                f.write(response.content)
        else:
            print("\n{}\t not worked\n{}]\t not downloaded".format(downloadLink, pdfName))
