# SmartAss OnBoard
# Using LexFridman Podcase Transcript specifically
import requests
from bs4 import BeautifulSoup

def get_lex_links():
    url = "https://lexfridman.com/podcast"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    transcripts = {}
    vid_titles = soup.find_all('div', class_='vid-title')
    vid_persons = soup.find_all('div', class_='vid-person')
    vid_materials = soup.find_all('div', class_='vid-materials')
    for title, person, materials in zip(vid_titles, vid_persons, vid_materials):
        name = person.get_text(strip=True)
        transcript_link = None
        links = materials.find_all('a')
        for link in links:
            if 'transcript' in link.get('href'):
                transcript_link = link.get('href')
                break
        if transcript_link:
            transcripts[name] = transcript_link
    return transcripts


def parse_lex_transcript_with_soup(soup, name):
    transcript_segments = soup.find_all('div', class_='ts-segment')
    transcript = ""
    curr_speaker = ""
    curr_response = ""

    for segment in transcript_segments:
        speaker = segment.find('span', class_='ts-name').get_text(strip=True)
        timestamp = segment.find('span', class_='ts-timestamp').get_text(strip=True)
        text = segment.find('span', class_='ts-text').get_text(strip=True)

        if curr_speaker == "":
            curr_speaker = speaker
            curr_response = text 
        elif curr_speaker != speaker:
            transcript += f"\n\n{curr_speaker}: {curr_response}"
            curr_speaker = speaker
            curr_response = text
        elif curr_speaker == speaker:
            curr_response = curr_response.strip() + f" {text}"  

    # Save the trnascript into txt files 
    import os
    os.makedirs("genius", exist_ok=True)
    with open(f"genius/transcript_{name}.txt", "w") as file:
        file.write(transcript.strip())


def parse_lex_transcript(name, url):
    response = requests.get(url)
    # Check if the request was successful
    try:
        soup = BeautifulSoup(response.content, 'html.parser')
        parse_lex_transcript_with_soup(soup, name)
    except:
        print("Issue parsing transcript for ", name)


to_lower_name = lambda name: name.lower().replace(' ', '-')
to_cap_name = lambda name_lower: name_lower.replace('-', ' ').capitalize()


def scape_genius_transcript():
    import os
    os.makedirs("genius", exist_ok=True)
    transcripts = get_lex_links()
    for name, url in transcripts.items():
        try:
            parse_lex_transcript(name, url)
        except:
            print("Issue parsing transcript for ", name)

