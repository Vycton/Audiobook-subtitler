import os
import re

import ebooklib
from ebooklib import epub

import stable_whisper

from bs4 import BeautifulSoup

def get_lines(content):
    soup = BeautifulSoup(content, 'html.parser')
    lines = soup.get_text().split("\n")
    lines = [x for x in lines if x]
    lines = [x.replace('\u3000', "") for x in lines]
    text = "\n".join(lines)
    text = text.replace("。","。\n")
    return text

def align(text, audio_file):
    model = stable_whisper.load_model('base')
    result = model.align(audio_file, text, language='ja', original_split=True)
    
    print(result.to_srt_vtt(word_level=False))
    print(result.to_txt())

def read_ebook():
    book = epub.read_epub('./input/Book 1.epub')
    for i in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        print(i.get_name())
        if i.get_name() == "text/part0007.html":
            return get_lines(i.get_content())

if __name__ == "__main__":
    input_dir = "./input/Book 1 mp3/"
    mp3_files = []
    for file in os.listdir(input_dir):
        if file.endswith(".mp3"):
            mp3_files.append(input_dir+file)
    print("\n".join([(str(x)+": "+i) for (x,i) in enumerate(mp3_files)]))
    
    text = read_ebook()
    align(text, mp3_files[3])
