import json
import wiktextract
import sys
import ijson




fh = open("output.json", "w")
def word_cb(data):
    json.dump(data,fh)
    
    
ctx = wiktextract.parse_wiktionary(
    "enwiktionary-latest-pages-articles.xml", word_cb,
    capture_cb=None,
    languages=["English", "Translingual"],
    translations=False,
    pronunciations=False,
    redirects=False)

print("{} English entries processed.".format(ctx.language_counts["English"]))
print("{} bytes written to output.json".format(fh.tell()))

fh.close()
