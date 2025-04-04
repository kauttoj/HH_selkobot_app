import os
import pickle
import io
import time
import re
import random
import ast
import numpy as np
import pandas as pd
import base64
import markdown2
from datetime import datetime

import tiktoken
from bs4 import BeautifulSoup, NavigableString
from xhtml2pdf import pisa
from PIL import Image, ImageDraw
from dotenv import load_dotenv
import openai
from litellm import completion
from bs4 import BeautifulSoup, NavigableString, Tag
from html import escape
from pydantic import BaseModel
import anthropic

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv('.env')
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

client_openai = openai.OpenAI()
client_anthropic = anthropic.Anthropic()

FACTCORRECT_model = "gpt-4o"
PARSER_model =  "gpt-4o"
WRITER_models = ['claude-3-7-sonnet-20250219','gpt-4o-2024-08-06']

PROMPT_writer = {
    'prompt_22':
'''Olet uutistoimittaja ja tehtäväsi on yksinkertaistaa uutistekstejä seuraavien ohjeiden mukaan:

Tekstin rakenne:
-Säilytä alkuperäisen tekstin otsikko (<title>), mutta yksinkertaista sitä tarvittaessa
-Kirjoita ingressi (<lead>), joka kertoo uutisen kiinnostavimman asian
-Jaa teksti sopiviin osiin väliotsikoilla (<subtitle>)
-Säilytä suorat lainaukset (<quote>), mutta muuta niiden sisältö selkokielelle

Kielen yksinkertaistaminen:
-Säilytä tekstin asiasisältö
-Käytä lyhyitä ja selkeitä lauseita
-Vältä vaikeita, harvinaisia tai vieraskielisiä sanoja
-Vältä erikoistermejä

Tekstin järjestys:
-Etene loogisesti ja kronologisesti
-Säilytä uutisen pääasia ja tärkeimmät yksityiskohdat
-Esitä yksi asia kerrallaan
-Karsi epäolennaiset asiat 

Muotoilu:
-Käytä lyhyitä kappaleita
-Käytä väliotsikoita tekstin jäsentämiseen
-Säilytä suorat lainaukset, mutta yksinkertaista niitä tarvittaessa

Tärkeää:
-Älä muuta faktoja tai lisää omia tulkintoja
-Säilytä tekstin luotettavuus ja asiallisuus
-Pidä teksti informatiivisena mutta yksinkertaisena
-Huomioi lukijan mahdollinen rajallinen suomen kielen taito

Muokkaa teksti niin, että se on helppolukuinen. Tekstin tulee olla selkeä, kiinnostava ja ymmärrettävä, mutta säilyttää alkuperäisen uutisen asiasisältö ja luotettavuus.
Älä koskaan keksi uutta asiasisältöä tai faktoja, joita alkuperäisessä tekstissä ei mainita.

Nyt, mukauta annettu uutisteksti noudattaen näitä ohjeita.{TOKENS}Merkitse kirjoittamasi yksinkertaistettu teksti <simplified_article> tagien sisään.''',
    'prompt_29':
    '''
Olet selkokielisen uutispalvelun toimittaja. Tehtäväsi on mukauttaa annettuja uutistekstejä selkokielelle seuraavien ohjeiden mukaan:

** Tekstin rakenne **
- Säilytä alkuperäisen tekstin otsikko (<title>), mutta yksinkertaista sitä tarvittaessa
- Kirjoita ytimekäs ingressi (<lead>), joka kertoo uutisen tärkeimmän asian
- Jaa teksti sopiviin osiin väliotsikoilla (<subtitle>), jotka tiivistävät tärkeimmät asiat
- Säilytä tärkeimmät suorat lainaukset (<quote>) ja lyhennä niitä tarvittaessa

** Kielen yksinkertaistaminen **
- Käytä lyhyitä ja selkeitä lauseita
- Vältä vaikeita sanoja ja ilmaisuja
- Selitä vaikeat käsitteet
- Käytä konkreettista ja suoraa kieltä
- Säilytä tekstin tyyli ja tärkein asiasisältö muuttumattomana

** Tekstin järjestys **
- Etene loogisesti ja kronologisesti
- Esitä yksi asia kerrallaan
- Vältä viittauksia tekstin eri osien välillä
- Kerro asiat suoraan ja konkreettisesti

** Tekstin muotoilu **
- Käytä lyhyitä kappaleita
- Käytä ydinasiat tiivistäviä väliotsikoita <subtitle> pitkän tekstin jäsentämiseen
- Säilytä tärkeimmät suorat lainaukset <quote>, mutta yksinkertaista niitä tarvittaessa

** Tärkeää **
- Sopiva selkokielisen tekstin pituus on noin 80% alkuperäisestä uutisesta.
- Älä koskaan muuta alkuperäisen uutisen faktoja tai lisää omia tulkintoja
- Säilytä alkuperäisen tekstin tyylilaji
- Pidä teksti informatiivisena ja viihdyttävänä, vaikka se on yksinkertaista
- Huomioi lukijan mahdollinen rajallinen suomen kielen taito

Muokkaa annettu uutisteksti niin, että se on helppolukuinen, selkeä ja ymmärrettävä, mutta ei lapsellinen.
Älä koskaan keksi uutta asiasisältöä tai faktoja, joita alkuperäisessä tekstissä ei mainita.
Nyt, mukauta annettu uutisteksti selkokielelle noudattaen näitä ohjeita.{TOKENS}Merkitse kirjoittamasi selkokielinen teksti <simplified_article> tagien sisään.    
    ''',
    'prompt_21':
    '''
** TEHTÄVÄNANTO **:
Olet selkokielisen uutismedian toimittaja. Tehtäväsi on muuttaa uutisteksti selkokielelle säilyttäen sen tyyli ja keskeisin asiasisältö. 

** SELKOKIELEN MÄÄRITELMÄ **:
Selkokieli on suomen kielen muoto, joka on muutettu sisällöltään, sanastoltaan ja rakenteeltaan yleiskieltä helpommin ymmärrettäväksi. Se on suunnattu henkilöille, joilla on vaikeuksia lukea tai ymmärtää yleiskieltä. Tyypillinen selkokielinen uutisteksti on n. 80% alkuperäisen uutistekstin pituudesta.

** TEKSTIN SISÄLTÖ **:
-Poimi lähtötekstistä sen tärkein sisältö
-Muokkaa tekstistä selkokielinen
-Tee uutiselle tiivis otsikko ja informatiivinen ingressi
-Käytä väliotsikoita tekstin jäsentämiseksi
-Säilytä suorat lainaukset, mutta yksinkertaista niiden kieli

** TEKSTIN KIELIASU **
-Kirjoita lyhyitä päälauseita
-Käytä aktiivisia verbejä
-Käytä yleisiä, arkisia sanoja
-Älä toista samoja sanoja useita kertoja
-Vältä toistamasta peräkkäin useita lyhyitä päälauseita
-Kirjoita yksinkertaisia sivulauseita
-Selitä vaikeat sanat ja käsitteet heti niiden jälkeen
-Vältä erikoistermejä
-Kirjoita numerot ja päivämäärät selkeästi auki
-Avaa lyhenteet ja vieraskieliset sanat
-Selitä viranomaisten ja organisaatioiden toiminta yksinkertaisesti
-Säilytä henkilöiden ja paikkojen nimet alkuperäisinä

** RAJOITUKSET **
- Älä keksi uutta sisältöä tai faktoja
- Älä muuta alkuperäisen uutisen merkitystä
- Älä lisää tekstiin omia tulkintojasi tai mielipiteitäsi
- Älä käytä kielikuvia tai sanontoja
- Älä käytä muita kuin määriteltyjä HTML-tageja

** PAKOLLISET HTML-TAGIT **
- `<title>` - Otsikko (pakollinen)
- `<lead>` - Ingressi (pakollinen)
- `<subtitle>` - Väliotsikot (tarvittaessa)
- `<quote>` - Suorat lainaukset (säilytä alkuperäiset, mutta yksinkertaista kieli)

** TEKSTIN VIIMEISTELY **
Varmista, että:
-Kaikki tärkeät faktat on säilytetty
-Teksti etenee loogisesti
-Kieli on yhtenäistä ja selkeää
-Vaikeat sanat on selitetty
-HTML-tagit ovat oikein
-Et ole lisännyt uutta asiasisältöä

Nyt, mukauta annettu uutisteksti selkokielelle noudattaen näitä ohjeita.{TOKENS}Merkitse kirjoittamasi selkokielinen teksti <simplified_article> tagien sisään.    
''',
    'prompt_20':
    '''
Olet selkokielisen uutismedian toimittaja. Tehtäväsi on muuttaa uutistekstejä selkokielelle. Noudatat seuraavia periaatteita:

** TEKSTIN RAKENNE **:
- Säilytä HTML-tagit <title>, <lead>, <subtitle> ja <quote>
- Aloita teksti aina tiiviillä otsikolla (<title>) ja otsikkoa taustoittavalla ingressillä (<lead>)
- Käytä väliotsikoita (<subtitle>) jakaessasi tekstiä osiin 
- Säilytä sitaatit (<quote>) mutta yksinkertaista niiden kieli
- Pidä kappaleet lyhyinä, korkeintaan 3–4 virkettä

** KIELEN SÄÄNNÖT **:
- Kirjoita lyhyitä päälauseita
- Käytä aktiivisia verbejä
- Käytä yleisiä, arkisia sanoja
- Vältä samojen sanojen toistamista
- Selitä vaikeat sanat ja käsitteet heti niiden jälkeen
- Vältä erikoistermejä
- Kirjoita numerot ja päivämäärät selkeästi auki

** SISÄLLÖN KÄSITTELY **:
- Kiteytä artikkelin kiinnostavin asia tiiviiseen otsikkoon
- Säilytä uutisen pääasia ja tärkeimmät yksityiskohdat
- Etene loogisessa järjestyksessä
- Kerro väliotsikossa jokin mielenkiintoinen yksityiskohta
- Anna tarvittava taustatieto
- Karsi epäolennainen pois
- Säilytä alkuperäisen uutisen tyyli

** ERITYISOHJEITA **:
- Avaa lyhenteet ja vieraskieliset sanat
- Selitä viranomaisten ja organisaatioiden toiminta yksinkertaisesti
- Säilytä henkilöiden ja paikkojen nimet alkuperäisinä

Tuota selkokielinen versio, joka on helposti ymmärrettävä mutta säilyttää alkuperäisen uutisen tärkeimmän sisällön. Älä lisää tekstiin omia tulkintojasi tai mielipiteitäsi.
Älä koskaan keksi uutta asiasisältöä tai faktoja, joita alkuperäisessä tekstissä ei mainita.
Nyt, mukauta annettu uutisteksti selkokielelle noudattaen näitä ohjeita.{TOKENS}Merkitse kirjoittamasi selkokielinen teksti <simplified_article> tagien sisään.    
    ''',
    'prompt_25':
    '''
Olet uutistoimittaja ja tehtäväsi on muokata uutistekstejä selkokielelle seuraavien ohjeiden mukaan:

Tekstin rakenne:
-Säilytä alkuperäisen tekstin otsikko (<title>), mutta yksinkertaista sitä tarvittaessa
-Kirjoita ytimekäs ingressi (<lead>), joka kertoo uutisen tärkeimmän asian
-Jaa teksti sopiviin osiin informatiivisilla väliotsikoilla (<subtitle>)
-Säilytä suorat lainaukset (<quote>) ja merkitse ne selkeästi

Kielen yksinkertaistaminen:
-Käytä lyhyitä ja selkeitä lauseita
-Vältä vaikeita, harvinaisia tai vieraskielisiä sanoja ja ilmaisuja
-Vältä tarinallisuutta, kielikuvia, sanontoja ja erikoistermejä
-Säilytä tekstin asiasisältö muuttumattomana

Tekstin järjestys:
-Etene loogisesti ja kronologisesti
-Säilytä uutisen pääasia ja tärkeimmät yksityiskohdat
-Esitä yksi asia kerrallaan
-Karsi epäolennaiset asiat 

Muotoilu:
-Käytä lyhyitä kappaleita
-Käytä väliotsikoita tekstin jäsentämiseen
-Säilytä suorat lainaukset, mutta yksinkertaista niitä tarvittaessa

Tärkeää:
-Älä muuta faktoja tai lisää omia tulkintoja
-Säilytä tekstin luotettavuus ja asiallisuus
-Pidä teksti informatiivisena, mutta yksinkertaisena
-Huomioi lukijan mahdollinen rajallinen suomen kielen taito

Muokkaa teksti niin, että se on helppolukuinen. Tekstin tulee olla selkeä, kiinnostava ja ymmärrettävä, mutta säilyttää alkuperäisen uutisen asiasisältö ja luotettavuus.Tyypillinen selkokielinen uutisteksti on mitaltaan n. 80% alkuperäisestä uutistekstistä.

Älä koskaan keksi uutta asiasisältöä tai faktoja, joita alkuperäisessä tekstissä ei mainita.
Nyt, mukauta annettu uutisteksti selkokielelle noudattaen näitä ohjeita.{TOKENS}Merkitse kirjoittamasi selkokielinen teksti <simplified_article> tagien sisään.    
    '''
}

PROMPT_error_correct = '''Olet uutisten faktantarkastaja, jonka tehtävä on tarkastaa että annetun uuden tekstin faktasisältö vastaa alkuperäistä vanhaa tekstiä. Korjaat uuden tekstin tarvittaessa. 
Keskity ainoastaan kriittisiin asiavirheisiin. Tee VAIN VÄLTTÄMÄTTÖMÄT korjaukset, jotka vakavasti vääristävät uuden tekstin asiasisältöä. Jätä teksti muilta osin täysin ennalleen. Et huomio kielellisiä ilmaisuja, tekstin muotoilua, tekstin ulkoasua tai lausejärjestystä, jotka eivät vaikuta asiasisältöön.

 Sinun tulee aina säilyttää seuraavat tekstissä olevat TAGIT sellaisenaan:
- `<title>` otsikko
- `<lead>` ingressi
- `<subtitle>` väliotsikot
- `<quote>` suorat lainaukset
 Tageja käytetään tekstin ladontaan verkkosivulle, ne eivät kuulu faktantarkastuksen piiriin. Jätä nämä tagit paikoilleen.

Seuraavassa saat alkuperäisen vanhan ja uuden tarkastettavan tekstin.

 # ALKUPERÄINEN TEKSTI (teksti A)

<alkuperäinen_teksti_A>
{old_text}
</alkuperäinen_teksti_A>

 # TARKASTETTAVA TEKSTI (teksti B)
 
<tarkastettava_teksti_B> 
{new_text}
<tarkastettava_teksti_B>

 # TEHTÄVÄ
 
 Vertaa alkuperäisessä (A) ja tarkastettavassa (B) tekstissä kerrottuja faktoja toisiinsa. Jos löydät vakavia asiavirheitä tarkastettavassa tekstissä, listaa ne kaikki ja kirjoita uusi, korjattu versio tekstistä B. Muussa tapauksessa palautat tekstin V alkuperäisessä muodossaan ilman mitään muutoksia.
  
 Tarkastettavat asiat sisältävät VAIN seuraavat:
 1. lukumäärät ja numerot
 2. erisnimet
 3. ammattinimikkeet
 4. keskeiset tapahtumat
 5. keskeiset tärkeät väitteet
 6. ajankohdat
 
 Nämä eivät saa muuttua oleellisesti.
 
 Huomioita:
 - Uusi teksti voi olla lyhennetty ja tiivistetty versio alkuperäisestä, eikä tätä lasketa virheeksi.
 - Uudessa tekstissä asiat voi olla selitetty eri tavalla ja yksinkertaisemmin, eikä tätä ei lasketa virheeksi.
 - Korjaa vain vakavat asiavirheet, älä koskaan puutu kieli tai tyyliseikkoihin.
 - Jos löydät asiavirheitä ja kirjoitat korjatun version tekstistä, tekstin pituus ei saa muuttua merkittävästi.
 - Tee aina vain välttämättömät korjaukset, jotka selvästi vääristävät uuden tekstin asiasisältöä. Jätä teksti muilta osin ennalleen.
 - Älä koskaan muuta tekstissä olevia tageja, vaan pidä ne aina ennallaan.

 # VASTAUS
 
Jos uudessa tekstissä B ei ole vakavia asiavirheitä, palauta teksti B sellaisenaan. Muussa tapauksessa listaa kaikki virheet ja kirjoita kokonainen uusi korjattu versio tekstistä B. 
'''

PROMPT_parse_text = '''
You are a text parser who reads Finnish input text articles, parses them using given 4 standard tags and returns a parsed version of the text article. You need to identify and tag the following components of the text: title, lead, subtitles and quotes.

During parsing, your will check and add these 4 types of tags into the text. These tags are the following:
- `<title>` Title of the article, always first part of the text.
- `<lead>` Lead (ingress) of the article, which always follows the title if present.
- `<subtitle>` Subtitles appearing in the middle of the article. Typically present only for long articles. Each article can contain anything between 0 and 10 subtitles. Subtitles clearly separate different parts of the article content. Not all articles contains subtitles.
- `<quote>` Direct quotes that are marked with some type of dashes, lines or quotations, such as “, ”, «, ❞ or 〞. Not all articles contain quotes, while some may contain several.

Each text MUST contain one title, while other tag types are optional and depend on text content and length. 

Below are three examples of potential text inputs and the proper output for all of them.

## Example 1: no internal structure, need to estimate all tags.

Espoossa saa kesällä puistoruokaa 16 vuoden tauon jälkeen

Lapset saavat syödä ilmaisen lounaan tänä kesänä viidessä espoolaisessa leikkipuistossa. Kyseessä on kokeilu.

Lapset saavat tänä kesänä ilmaista ruokaa myös Espoon leikkipuistoissa. Ruokaa saa heinäkuussa Soukan, Perkkaan, Olarin, Suvelan ja Tapiolan asukaspuistoissa.

Ruokailuun ei tarvitse ilmoittautua. Mukaan täytyy ottaa kotoa lautanen, lusikka, haarukka ja muki. Kaikille on tarjolla sama ruoka. Kahtena tai kolmena päivänä viikossa se on kasvisruoka.

Espoossa järjestettiin puistoruokailu viimeksi 16 vuotta sitten. Sen jälkeen kaupunki halusi säästää. Puistojen ruokailuissa syntyi hävikkiä: 16 prosenttia ruoasta meni roskikseen. Tätä pidettiin ongelmana.

Puistoruokailuista kerätään tänä vuonna tietoa

Tälle vuodelle Espoon kaupunginvaltuusto varasi lähes 150 000 euroa puistoruokailuihin.

– Keräämme tietoa siitä, miten ruokailu järjestetään ja paljonko se maksaa, kertoo aluepäällikkö Nina Konttinen.

Hän vastaa Espoossa asukaspuistoista.

Kokeilun jälkeen päätetään, jatketaanko puistoruokailua Espoossa seuraavina kesinä.

## example 2: custom markers, need to convert to standard tags.   

title: Espoossa saa kesällä puistoruokaa 16 vuoden tauon jälkeen

ingress: Lapset saavat syödä ilmaisen lounaan tänä kesänä viidessä espoolaisessa leikkipuistossa. Kyseessä on kokeilu.

Lapset saavat tänä kesänä ilmaista ruokaa myös Espoon leikkipuistoissa. Ruokaa saa heinäkuussa Soukan, Perkkaan, Olarin, Suvelan ja Tapiolan asukaspuistoissa.

Ruokailuun ei tarvitse ilmoittautua. Mukaan täytyy ottaa kotoa lautanen, lusikka, haarukka ja muki. Kaikille on tarjolla sama ruoka. Kahtena tai kolmena päivänä viikossa se on kasvisruoka.

Espoossa järjestettiin puistoruokailu viimeksi 16 vuotta sitten. Sen jälkeen kaupunki halusi säästää. Puistojen ruokailuissa syntyi hävikkiä: 16 prosenttia ruoasta meni roskikseen. Tätä pidettiin ongelmana.

subtitle: Puistoruokailuista kerätään tänä vuonna tietoa

Tälle vuodelle Espoon kaupunginvaltuusto varasi lähes 150 000 euroa puistoruokailuihin.

"Keräämme tietoa siitä, miten ruokailu järjestetään ja paljonko se maksaa, kertoo aluepäällikkö Nina Konttinen."

Hän vastaa Espoossa asukaspuistoista.

Kokeilun jälkeen päätetään, jatketaanko puistoruokailua Espoossa seuraavina kesinä.

## example 3: custom HTML tags which need to be converted into our 4 standard tags.

<h1>Espoossa saa kesällä puistoruokaa 16 vuoden tauon jälkeen</h1>

<h2>Lapset saavat syödä ilmaisen lounaan tänä kesänä viidessä espoolaisessa leikkipuistossa. Kyseessä on kokeilu.</h2>

Lapset saavat tänä kesänä ilmaista ruokaa myös Espoon leikkipuistoissa. Ruokaa saa heinäkuussa Soukan, Perkkaan, Olarin, Suvelan ja Tapiolan asukaspuistoissa.

Ruokailuun ei tarvitse ilmoittautua. Mukaan täytyy ottaa kotoa lautanen, lusikka, haarukka ja muki. Kaikille on tarjolla sama ruoka. Kahtena tai kolmena päivänä viikossa se on kasvisruoka.

Espoossa järjestettiin puistoruokailu viimeksi 16 vuotta sitten. Sen jälkeen kaupunki halusi säästää. Puistojen ruokailuissa syntyi hävikkiä: 16 prosenttia ruoasta meni roskikseen. Tätä pidettiin ongelmana.

<h3>Puistoruokailuista kerätään tänä vuonna tietoa</h3>

Tälle vuodelle Espoon kaupunginvaltuusto varasi lähes 150 000 euroa puistoruokailuihin.

<blockquote>Keräämme tietoa siitä, miten ruokailu järjestetään ja paljonko se maksaa, kertoo aluepäällikkö Nina Konttinen.</blockquote>

Hän vastaa Espoossa asukaspuistoista.

Kokeilun jälkeen päätetään, jatketaanko puistoruokailua Espoossa seuraavina kesinä.

## output: Proper parsed output for all inputs shown in examples 1-3.

<title>Espoossa saa kesällä puistoruokaa 16 vuoden tauon jälkeen</title>

<lead>Lapset saavat syödä ilmaisen lounaan tänä kesänä viidessä espoolaisessa leikkipuistossa. Kyseessä on kokeilu.</lead>

Lapset saavat tänä kesänä ilmaista ruokaa myös Espoon leikkipuistoissa. Ruokaa saa heinäkuussa Soukan, Perkkaan, Olarin, Suvelan ja Tapiolan asukaspuistoissa.

Ruokailuun ei tarvitse ilmoittautua. Mukaan täytyy ottaa kotoa lautanen, lusikka, haarukka ja muki. Kaikille on tarjolla sama ruoka. Kahtena tai kolmena päivänä viikossa se on kasvisruoka.

Espoossa järjestettiin puistoruokailu viimeksi 16 vuotta sitten. Sen jälkeen kaupunki halusi säästää. Puistojen ruokailuissa syntyi hävikkiä: 16 prosenttia ruoasta meni roskikseen. Tätä pidettiin ongelmana.

<subtitle>Puistoruokailuista kerätään tänä vuonna tietoa</subtitle>

Tälle vuodelle Espoon kaupunginvaltuusto varasi lähes 150 000 euroa puistoruokailuihin.

<quote>Keräämme tietoa siitä, miten ruokailu järjestetään ja paljonko se maksaa, kertoo aluepäällikkö Nina Konttinen.</quote>

Hän vastaa Espoossa asukaspuistoista.

Kokeilun jälkeen päätetään, jatketaanko puistoruokailua Espoossa seuraavina kesinä. 

# TASK

You need to verify that text contains tags or add them if not. If the text is already properly tagged, containing <title>, <lead>, <subtitle> or <quote> tags, you simply verify that all tags are complete, but do not add any new tags. 
If text has not tags or they are non-standard tags, you need to add standard tags yourself. When adding new tags, you need to carefully consider where to add tags. Apart from title, other tags might or might not be needed.

Instructions:
- input text could in various forms, your need to analyze the content and adapt to any input formats.
- When tagging text, do not add more tags than absolutely necessary; less is better.
- Output text must be ALWAYS clean, parsed text with standard tags that can ONLY contain tags <title>, <lead>, <subtitle> and <quote>, nothing else.
- if the input text is already tagged correctly, do not make any changes and simply pass the existing text as it is.

Important: You may NEVER change content of the article text in any way except fix and/or add given standard tags. Journalistic content and language of the article must remain exactly as it was.

# Output

Tagged version of the input text. Do NOT include any other outputs, such as explanations or comments.

tagged text:
'''

def markdown_to_pdf(md_text):
    """Converts Markdown text to PDF binary data."""
    html_text = markdown2.markdown(md_text)
    full_html = f"""
    <html>
    <head>
      <meta charset="UTF-8">
      <style>
        body {{
          font-family: 'DejaVuSans', Arial, sans-serif;
          font-size: 12pt;
          line-height: 1.6;
        }}
        h1 {{ color: darkblue; text-align: center; }}
        h2 {{ color: darkred; }}
        ul {{ margin-left: 20px; }}
        a {{ color: darkgreen; text-decoration: none; }}
      </style>
    </head>
    <body>{html_text}</body>
    </html>
    """
    pdf_buffer = io.BytesIO()
    pisa_status = pisa.CreatePDF(io.StringIO(full_html), dest=pdf_buffer)
    if pisa_status.err:
        raise Exception("Error in PDF generation")
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

def get_llm_response(system=None, input=None, model=None, temperature=None, responseformat=None):
    """Calls the LLM API using the provided prompt and configuration."""

    llm_config = {
        "model": model,
        "max_tokens": 16384,
        "temperature": temperature,
    }

    if ('claude' in llm_config['model']) and (responseformat is not None):
        raise Exception('!!! Claude does not support response_format parameter !!!')

    failed_count = 0
    response_txt = None
    while failed_count < 2:
        try:
            print(f"calling LLM {llm_config['model']}...", end='')
            if 'claude' in llm_config['model']:
                messages = [{'role': "user", 'content': input}]
                if system is None:
                    response = client_anthropic.messages.create(
                        model=llm_config['model'],
                        max_tokens=llm_config['max_tokens'],
                        temperature=llm_config['temperature'],
                        messages=messages
                    )
                else:
                    response = client_anthropic.messages.create(
                        model=llm_config['model'],
                        max_tokens=llm_config['max_tokens'],
                        system=system,
                        temperature=llm_config['temperature'],
                        messages=messages
                    )

                response_txt = response.content[0].text
            else:

                if system is None:
                    messages = [{'role': "user", 'content': input}]
                else:
                    messages = [{'role': "system", 'content': system}, {'role': "user", 'content': input}]

                if responseformat is not None:
                    response = client_openai.beta.chat.completions.parse(
                        model=llm_config['model'],
                        messages=messages,
                        response_format=responseformat,
                        temperature=llm_config['temperature'],
                        max_tokens=llm_config['max_tokens']
                    )
                    response_txt = response.choices[0].message.parsed
                else:
                    response = client_openai.chat.completions.create(
                        model=llm_config['model'],
                        messages=messages,
                        temperature=llm_config['temperature'],
                        max_tokens=llm_config['max_tokens']
                    )
                    response_txt = response.choices[0].message.content

            print("...success")
            break
        except Exception as ex:
            print(f"...FAILED: {ex}")
            failed_count += 1

    if response_txt is None:
        raise Exception("Failed to get LLM response after several attempts.")

    return response_txt

def remove_tags(input_text):
    # Define various quotation marks that might appear
    quotation_marks = ['“', '”', '«', '»', '‘', '’', '❝', '❞', '〝', '〞']

    # Function to handle <quote> tags specifically
    def handle_quote(match):
        content = match.group(1).strip()
        # Remove any existing quotation marks
        if any([content[0] in quotation_marks]):
            content = content[1:]
        if any([content[-1] in quotation_marks]):
            content = content[0:-1]
        # Check if the content does not start and end with ASCII double quotes
        content = f'«{content}»'
        return content

    # Handle <quote> tags separately, adding ASCII double quotes if needed
    output_text = re.sub(r'<quote>(.*?)</quote>', handle_quote, input_text)

    def handle_titles(match):
        content = match.group(1).strip()
        # Remove any existing quotation marks
        if any([content[0] in quotation_marks]):
            content = content[1:]
        if any([content[-1] in quotation_marks]):
            content = content[0:-1]
        # Check if the content does not start and end with ASCII double quotes
        content = f'«{content}»'
        return content

    # Remove all other tags
    tags = ['title', 'lead', 'subtitle']
    for tag in tags:
        if tag == 'subtitle':
            output_text = re.sub(f'<{tag}[^>]*>(.*?)</{tag}>',
                                 lambda m: f"{m.group(1)}",
                                 output_text)
        else:
            output_text = re.sub(f'<{tag}[^>]*>(.*?)</{tag}>',
                                 lambda m: f"{tag}: {m.group(1)}",
                                 output_text)

    return output_text

def count_tokens(input_text,model):

    if 'claude' in model:
        response = client_anthropic.messages.count_tokens(
            model=model,
            system="",
            messages=[{
                "role": "user",
                "content": input_text
            }],
        )
        tokencount = int(response.input_tokens)
    else:
        encoding = tiktoken.encoding_for_model(model)
        tokencount = int(len(encoding.encode(input_text)))

    return tokencount

def filewriter(content, filename):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        raise Exception(f"Error writing file {filename}: {str(e)}")

def filereader(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise Exception(f"Error reading file {filename}: {str(e)}")

def tagged_text_to_noncolored_html(tagged_text):
    # Instead of colors, we'll map tags to their respective HTML transformations
    tag_transform_map = {
        'title': lambda lines: '\n'.join(f'<h1>{line.strip()}</h1>' for line in lines if line.strip()),
        'lead': lambda lines: '\n'.join(f'<h3>{line.strip()}</h3>' for line in lines if line.strip()),
        'subtitle': lambda lines: '\n'.join(f'<h4>{line.strip()}</h4>' for line in lines if line.strip()),
        'quote': lambda lines: '\n'.join(f'<p><i>«{line.strip()}»</i></p>' for line in lines if line.strip()),
    }

    def process_element_children(element):
        content = ''
        for child in element.contents:
            result = process_element(child)
            if result.strip():
                # Split content into lines
                lines = result.split('\n')
                # Wrap each line with <p></p>, except we'll do this only for non-transformed text
                wrapped_lines = [f'<p>{line.strip()}</p>' for line in lines if line.strip()]
                content += '\n'.join(wrapped_lines)
        return content

    def process_element(element):
        if isinstance(element, NavigableString):
            # Escape HTML special characters in text nodes
            text = escape(str(element))
            return text
        elif isinstance(element, Tag):
            tag_name = element.name
            # Process child elements
            content = ''.join([process_element(child) for child in element.contents])
            lines = content.split('\n')

            if tag_name in tag_transform_map:
                # Apply transformation for defined tags
                transformed_content = tag_transform_map[tag_name](lines)
                return transformed_content
            else:
                # For other tags or content, process recursively
                return process_element_children(element)
        else:
            return ''

    # Parse the entire tagged_text
    soup = BeautifulSoup(tagged_text, 'html.parser')

    # Process the entire content
    html_text = process_element(soup)

    # Build final HTML (no color coding needed anymore)
    html_output = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        border: 1px solid #4CAF50;
        p {{
            margin-bottom: 1em;
            font-family: Arial, sans-serif;
        }}
        .news-url {{
        font-size: 18px;
        margin-bottom: 10px;
        }}
    </style>
</head>
<body>
{html_text}
</body>
</html>'''

    return html_output

class FactCheckedText(BaseModel):
    critical_errors_found: bool
    list_of_critical_errors: list[str]
    new_text: str

def clean_generated_text(text):
    """Clean up generated text by removing tags and extra whitespace"""
    text = text.strip()

    # Remove XML-style tags if present
    start_tag = '<simplified_article>'
    end_tag = '</simplified_article>'

    if start_tag in text:
        text = text[text.find(start_tag) + len(start_tag):]
    if end_tag in text:
        text = text[:text.find(end_tag)]

    # Normalize whitespace
    return text.strip() #.replace('\n\n', '\n')

def parse_text(raw_text: str,session_state) -> str:
    '''Call LLM to parse input text and returns parsed text'''

    model = session_state.get('model')
    agent = session_state.get('agent_type')
    prompt = session_state.get('prompt_type')
    temp = session_state.get('temperature')

    if ('<title>' in raw_text and '</title>' in raw_text) and (
                    ('<subtitle>' in raw_text and '</subtitle>' in raw_text)
                    or ('<quote>' in raw_text and '</quote>' in raw_text)
            ):
        return raw_text

    response = get_llm_response(system=PROMPT_parse_text,input='Parse the following text input:\n\n--------\n' + raw_text,model=PARSER_model,temperature=0)

    return response

def simplify_text(parsed_text: str,session_state) -> str:
    '''Call LLM to simplify input text and returns simplified text'''

    model = session_state.get('model')
    agent = session_state.get('agent_type')
    prompt_type = session_state.get('prompt_type')
    temp = session_state.get('temperature')
    use_tokencount = session_state.get('tokencount')

    if use_tokencount:
        tokencount = int(count_tokens(parsed_text,model)*0.80)
        prompt = PROMPT_writer[prompt_type].replace('{TOKENS}',f' Sopiva selkokielelle mukautetun tekstin pituus on noin 80% alkuperäisestä eli noin {tokencount} tokenia. ')
    else:
        prompt = PROMPT_writer[prompt_type].replace('{TOKENS}',f" ")

    input = f'Mukauta seuraava uutisteksti selkokielelle:\n\n{parsed_text}'
    response = get_llm_response(system=prompt,input=input,model=model,temperature=temp)

    response = clean_generated_text(response)

    if 'fakta' in agent:
        PROMPT = PROMPT_error_correct
        PROMPT = PROMPT.replace('{new_text}',response).replace('{old_text}',parsed_text)
        resp = get_llm_response(system=None,input=PROMPT,model=FACTCORRECT_model,temperature=0,responseformat=FactCheckedText)
        response = resp.model_dump()
        if response['critical_errors_found']:
            print(f'fact-checker identified following {len(response["list_of_critical_errors"])} critical errors:')
            for error in response['list_of_critical_errors']:
                print(f'...{error}\n')
        response = clean_generated_text(response['new_text'])
        response = response.replace('<tarkastettava_teksti_B>','').replace('</tarkastettava_teksti_B>','').strip()

    return response


