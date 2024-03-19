import os
from multiprocessing import Pool

import json
from pathlib import Path

import ebooklib
from ebooklib import epub

import stable_whisper

from thefuzz import fuzz

from bs4 import BeautifulSoup

LANG = 'ja'
SENTENCE_DELINEATORS = "。？！…〜"
TRANSCR_POOL_SIZE = 8
TRANSCR_MODEL = 'tiny'
ALIGN_MODEL = 'large-v3'

def read_chapter(chapter):
    soup = BeautifulSoup(chapter, 'html.parser')
    lines = soup.get_text().split("\n")
    lines = [x for x in lines if x]
    lines = [x.replace('\u3000', "") for x in lines]
    text = "\n".join(lines)
    for c in SENTENCE_DELINEATORS:
        text = text.replace(c, c+"\n")
    return text

def read_ebook(book_file):
    book = epub.read_epub(book_file)
    chapters = []
    for ch in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        chapters.append(read_chapter(ch.get_content()))
    return chapters

def transcribe_chapter(audio_file):
    model = stable_whisper.load_model(TRANSCR_MODEL)
    result = model.transcribe(audio_file, language=LANG)
    return (audio_file, result.to_txt())

def transcribe_audiobook(audio_dir):
    cache_file = Path(audio_dir+"transcription-cache.json")
    if cache_file.is_file():
        print("reading transcripts from cache at "+str(cache_file))
        with open(cache_file, 'r') as f:
            transcriptions = json.load(f)
        return transcriptions

    files = []
    for file in os.listdir(audio_dir):
        if file.endswith(".mp3"):
            files.append(audio_dir+file)

    with Pool(TRANSCR_POOL_SIZE) as p:
        transcriptions = p.map(transcribe_chapter,files)

    transcriptions = {
            t[0]:{'transcript':t[1], 'chapter':"", 'best':0, 'second':0}
            for t in transcriptions
            }

    with open(cache_file, 'w+') as f:
        print("writing transcripts to cache at "+str(cache_file))
        json.dump(transcriptions, f, ensure_ascii=False)
    return transcriptions

def match_files(transcriptions, chapters):
    for audio_chapter in transcriptions.values():
        for book_chapter in chapters:
            ratio = fuzz.ratio(audio_chapter['transcript'], book_chapter)
            if ratio > audio_chapter['best']:
                audio_chapter['second'] = audio_chapter['best']
                audio_chapter['best'] = ratio
                audio_chapter['chapter'] = book_chapter
            elif ratio > audio_chapter['second']:
                audio_chapter['second'] = ratio
    return transcriptions


def align_chapter(text, audio_file, model, output_dir):
    print(f"Aligning {audio_file}...\n\n\n")
    result = model.align(audio_file, text, language=LANG, original_split=True)
    
    output_file = output_dir + str(Path(audio_file).stem) + ".srt"
    print(f"Writing subtitles to {output_file}")
    with open(output_file, 'w+') as f:
        f.write(result.to_srt_vtt(word_level=False))


def align_book(matched_files, output_dir):
    print("loading model...")
    model = stable_whisper.load_model(ALIGN_MODEL)
    for file, info in matched_files.items():
        align_chapter(info['chapter'], file, model, output_dir)

def main():
    input_dir = "./input/Book 1 mp3/"
    output_dir = "./output/subtitles/"
    transcriptions = transcribe_audiobook(input_dir)
    chapters = read_ebook('./input/Book 1.epub')
    matched_files = match_files(transcriptions, chapters)

    print("Audio files and their match scores for ebook chapters. Best < 70 or second best close to best means something might have gone wrong:")
    for file in matched_files.keys():
        info = matched_files[file]
        print(f"Best, second best: {info['best']}, {info['second']} for {file}")

    print("\n\nAligning matched ebook chapters to the audio files:")
    align_book(matched_files, output_dir)

if __name__ == "__main__":
    main()
