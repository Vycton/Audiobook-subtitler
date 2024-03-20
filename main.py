import os
from multiprocessing import Pool
import argparse

import json
from pathlib import Path

import ebooklib
from ebooklib import epub

import stable_whisper

from thefuzz import fuzz, process

from bs4 import BeautifulSoup

LANG = 'ja'
SENTENCE_DELINEATORS = "。"
TRANSCR_POOL_SIZE = 8
TRANSCR_MODEL = 'tiny'
ALIGN_MODEL = 'large-v3'


def read_chapter(chapter):
    """
    Takes the html of a single ebook chapter and produces the cleaned up text,
    adding a line break after each sentence.
    """
    soup = BeautifulSoup(chapter, 'html.parser')
    lines = soup.get_text().split("\n")
    lines = [x for x in lines if x]
    lines = [x.replace('\u3000', "") for x in lines]
    text = "\n".join(lines)
    for c in SENTENCE_DELINEATORS:
        text = text.replace(c, c+"\n")
    return text

def read_ebook_plain(book_file):
    """
    Reads and ebook file and returns a list of strings containing the text of 
    each chapter.
    """
    book = epub.read_epub(book_file, {'ignore_ncx': True})
    chapters = []
    for ch in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        chapters.append(read_chapter(ch.get_content()))
    return chapters

def read_ebook_toc(book_file, audio_files):
    """
    Reads and ebook file and tries to match chapters to the provided audio
    files by searching for links with link text inside the book.
    Prompts the user whether the matches are correct, 
    and returns None if they are not.
    """
    book = epub.read_epub(book_file, {'ignore_ncx': True})
    chapters = [x for x in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)]
    link_tags = []
    for ch in chapters:
        linksoup = BeautifulSoup(ch.get_content(),"html.parser")
        link_tags += list(linksoup.find_all('a'))

    table_links = []
    links = []
    links_text = []
    chapter_names = [ch.get_name() for ch in book.get_items()]
    for l in link_tags:
        table_link = l.get('href') 
        matched_name = process.extractOne(table_link, chapter_names)[0] 

        table_links.append(table_link)
        links.append(matched_name)
        links_text.append(l.get_text())

    match_list = []
    for path in audio_files:
        match_i = links_text.index(process.extractOne(str(path.stem), links_text)[0])
        match_list.append((path, links[match_i], table_links[match_i], links_text[match_i]))

    print("\n".join(f"{p.name} -> {ltext} ({table_link} -> {link})" for p, link, table_link, ltext in match_list))
    answer = input("\nIf the [audio file -> ebook chapter] and [table of contents link -> chapter link] pairs above are correct "+
            "alignment can begin. If not, matches will have to be made by transcribing " +
            "the audio files, which will take some time (5-10 minutes). Do the above matches seem correct? [Y/n]")

    if answer.lower() == "n":
        return None

    out_list = []
    for path, link, _, _ in match_list:
        chapter = book.get_item_with_href(link)
        chaptertext = read_chapter(chapter.get_content())
        out_list.append((path, chaptertext))

    return out_list

def get_audio_files(audio_dir):
    """
    Returns a list of Path objects for each .mp3 file in the given directory.
    """
    files = []
    for file in os.listdir(audio_dir):
        if file.endswith(".mp3"):
            files.append(Path(audio_dir+file))
    return files

def transcribe_chapter(audio_file):
    """
    Creates a transcript of the provided audio file using the TRANSCR_MODEL 
    model. Transcripts are only used to match audio files to the correct ebook
    chapter, so they do not need to be very accurate.
    """
    model = stable_whisper.load_model(TRANSCR_MODEL)
    result = model.transcribe(str(audio_file), language=LANG)
    return (audio_file, result.to_txt())

def transcribe_audiobook(audio_dir):
    """
    Creates a transcript for every provided audio file. This happens in 
    parallel with a pool size of TRANSCR_POOL_SIZE.
    Also creates a cache of transcripts in the audio directory, and uses that
    cache if it exists, skipping transcription.
    """
    cache_file = Path(audio_dir+".transcription-cache.json")
    if cache_file.is_file():
        print("reading transcripts from cache at "+str(cache_file))
        with open(cache_file, 'r') as f:
            transcriptions = json.load(f)
        return transcriptions
        
    files = [str(f) for f in get_audio_files(audio_dir)] #so it works with json
    print("loading models and transcribing (ignore jumping progress bar)...")
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

def match_files_transcriptions(transcriptions, chapters):
    """
    Matches audio files to the correct ebook chapter using their transcript.
    Also prints an overview of how good the matches are to give the user an 
    idea of whether the matching went correctly.
    """
    for audio_chapter in transcriptions.values():
        for book_chapter in chapters:
            ratio = fuzz.ratio(audio_chapter['transcript'], book_chapter)
            if ratio > audio_chapter['best']:
                audio_chapter['second'] = audio_chapter['best']
                audio_chapter['best'] = ratio
                audio_chapter['chapter'] = book_chapter
            elif ratio > audio_chapter['second']:
                audio_chapter['second'] = ratio

    print("\n")
    for file in transcriptions.keys():
        info = transcriptions[file]
        print(f"Best, second best: {info['best']}, {info['second']} for {Path(file).name}")

    print("Audio files and their match scores for ebook chapters. Best < 70 or second best close to best means something might have gone wrong.")
    return [(Path(fn), info['chapter']) for fn, info in transcriptions.items()] 


def align_chapter(text, audio_file, model, output_dir):
    """
    Aligns a single ebook chapter to its corresponding audio file, and saves
    the resulting .srt file. If the .srt already exists the chapter is skipped.
    """
    output_file = (output_dir / Path(audio_file).stem).with_suffix(".srt")
    if output_file.exists():
        print(f"Subtitle file {output_file.name} already exists! Skipping...")
        return

    print(f"\nAligning {audio_file.stem}...")
    result = model.align(str(audio_file), text, language=LANG, original_split=True)
    
    print(f"Writing subtitles to {output_file}")
    with open(output_file, 'w+') as f:
        f.write(result.to_srt_vtt(word_level=False))


def align_book(matched_files, output_dir):
    """
    Aligns each chapter in the ebook to its corresponding audio file using a 
    model of size ALIGN_MODEL.
    """
    print("loading model...")
    model = stable_whisper.load_model(ALIGN_MODEL)
    for path, text in matched_files:
        align_chapter(text, path, model, output_dir)

def main():
    parser = argparse.ArgumentParser( prog='audiobook-subtitler',
        description='Combines an .epub and directory of .mp3 files to create .srt subtitles for the .mp3s')

    parser.add_argument("epub_filename")
    parser.add_argument("audiobook_directory")
    args = parser.parse_args()

    ebook_file = args.epub_filename
    audio_dir = args.audiobook_directory
    audio_files = get_audio_files(audio_dir)


    matched_files = read_ebook_toc(ebook_file, audio_files)
    if not matched_files:
        print("\nMatching audio files to ebook chapters with transcription:")
        transcriptions = transcribe_audiobook(audio_dir)
        chapters = read_ebook_plain(ebook_file)
        matched_files = match_files_transcriptions(transcriptions, chapters)

    subtitle_dir = Path.cwd() / 'subtitles'
    subtitle_dir.mkdir(parents=True, exist_ok=True)
    print("\n\nAligning matched ebook chapters to the audio files:")
    align_book(matched_files, subtitle_dir)

if __name__ == "__main__":
    main()
