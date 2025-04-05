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

Keskity ainoastaan kriittisiin asiavirheisiin. Tee VAIN VÄLTTÄMÄTTÖMÄT korjaukset, jotka vakavasti vääristävät uuden tekstin asiasisältöä. Jätä teksti muilta osin täysin ennalleen. Et huomio kielellisiä ilmaisuja, tekstin muotoilua, tekstin ulkoasua tai lausejärjestystä, jotka eivät vaikuta asiasisältöön. Tietojen poisjättäminen ei ole automaattisesti virhe, kunhan se ei muuta olennaisesti tekstin ydinsisältöä tai väitteitä.

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
</tarkastettava_teksti_B>

# TEHTÄVÄ

Vertaa alkuperäisessä (A) ja tarkastettavassa (B) tekstissä kerrottuja faktoja toisiinsa. Jos löydät vakavia asiavirheitä tarkastettavassa tekstissä, listaa ne kaikki ja kirjoita uusi, korjattu versio tekstistä B. Muussa tapauksessa palautat tekstin B sellaisenaan ilman mitään muutoksia. Seuraa alla olevia ohjeita tarkasti.

**Tarkastettavat asiat sisältävät VAIN seuraavat:**
1. lukumäärät ja numerot  
2. erisnimet  
3. ammattinimikkeet  
4. keskeiset tapahtumat  
5. keskeiset tärkeät väitteet  
6. ajankohdat  

Nämä eivät saa muuttua oleellisesti.

**Ohjeita:**
- **Uusi teksti (B) on tiivistelmä eikä tarvitse sisältää kaikkia tietoja tekstistä (A).** Tietojen poisjättäminen ei ole automaattisesti virhe, kunhan se ei muuta olennaisesti tekstin ydinsisältöä tai väitteitä.  
- Jos uusi teksti sisältää tietoa, jota ei ole löydettävissä tekstistä (A), se on vakava virhe ja poistettava. 
- Uudessa tekstissä asiat voi olla selitetty eri tavalla ja yksinkertaisemmin, eikä tätä lasketa virheeksi.  
- Pieniä numeroiden tai lukumäärien pyöristyksiä tai suuntaa-antavia mainintoja (esim. "yli kaksi miljoonaa" vs. "2 120 000") voi sallia, jos ne eivät merkittävästi vääristä tekstin ydinsisältöä. Älä muuta lukumääriä tai päiväyksiä ilman syytä (esim. 200 → 210 tai 3.5.2024 → 4.5.2024).  
- Korjaa vain vakavat asiavirheet, älä koskaan puutu kieli- tai tyyliseikkoihin.  
- Jos löydät asiavirheitä ja kirjoitat korjatun version tekstistä, tekstin pituus ei saa muuttua merkittävästi (sallittu ±5 %, noin ±20 sanaa).  
- Tee aina vain välttämättömät korjaukset, jotka selvästi vääristävät uuden tekstin asiasisältöä. Jätä teksti muilta osin ennalleen.  
- Älä koskaan muuta tekstissä olevia tageja, vaan pidä ne aina ennallaan.

# VASTAUS

Jos uudessa tekstissä B ei ole vakavia asiavirheitä, palauta teksti B kokonaan täsmälleen sellaisena kuin sen sait. Muussa tapauksessa:
1. Listaa kaikki löydetyt vakavat virheet.  
2. Kirjoita kokonaan uusi korjattu versio tekstistä B (muista säilyttää tagit).
'''

PROMPT_parse_text = '''
You are a text parser designed to analyze Finnish text articles and format them using exactly four custom tags: `<title>`, `<lead>`, `<subtitle>`, and `<quote>`. Your goal is to identify parts of the article text that belong to these four special categories and tag them accordingly.

### Tag Definitions:
- `<title>`: Mandatory tag. The main title of the article, always appearing at the beginning of the text.
- `<lead>`: Optional tag. The introductory paragraph(s) directly following the title, summarizing the article’s main point. Include only if clearly identifiable.
- `<subtitle>`: Optional tag. Subheadings within the text clearly dividing sections of content. Typically appear in longer articles that have distinct sections. Do not add subtitles to short articles with no clear sections.
- `<quote>`: Optional tag. Direct quotations explicitly marked with quotation symbols or dashes such as “, ”, «, ❞, or 〞. Include only if clearly identifiable.

### Tagging Instructions:
- Every article MUST contain exactly one `<title>` tag.
- Include `<lead>`, `<subtitle>`, and `<quote>` tags ONLY if clearly identifiable. When unsure, omit these optional tags entirely.
- Minimize tagging to only essential tags. Fewer tags are preferable; do NOT tag content unnecessarily.
- Preserve the original text exactly as provided. Do NOT alter wording, punctuation, or grammar.

### Handling Input:
- If the input article is already correctly tagged using only the standard custom tags (`<title>`, `<lead>`, `<subtitle>`, `<quote>`), do NOT make any changes. Return the text as-is.
- Do NOT rewrite, rephrase, or manipulate any part of the article’s wording. You must preserve full journalistic integrity: facts, phrasing, tone, and style must remain unchanged.
- Input articles may be completely unformatted, partially formatted with random or unusual tags, or contain HTML or other formatting.
- Input articles may be completely unformatted, partially formatted with random or unusual tags, or contain HTML or other formatting.
- Your task is to carefully analyze the content and context, using any available cues to determine correct placement for tags.
- Output text must contain ONLY the custom tags listed above (`<title>`, `<lead>`, `<subtitle>`, `<quote>`), along with regular, untagged text. Remove all other formatting or tags.

### Examples:

**Example 1: Plain text article**

Input:
```
Kesän lämpöaalto saavuttaa Suomen ensi viikolla

Meteorologit ennustavat, että lämpötila nousee yli 30 asteeseen.

Ihmiset ovat varautuneet helteisiin ostamalla viilentimiä ja ilmastointilaitteita. Monet kaupat ilmoittavat, että tuulettimet on myyty loppuun.

– Tuuletinvarastomme tyhjenivät nopeasti, kertoo kauppias Mikko Virtanen.
```
Output:
```
<title>Kesän lämpöaalto saavuttaa Suomen ensi viikolla</title>

<lead>Meteorologit ennustavat, että lämpötila nousee yli 30 asteeseen.</lead>

Ihmiset ovat varautuneet helteisiin ostamalla viilentimiä ja ilmastointilaitteita. Monet kaupat ilmoittavat, että tuulettimet on myyty loppuun.

<quote>– Tuuletinvarastomme tyhjenivät nopeasti, kertoo kauppias Mikko Virtanen.</quote>
```

**Example 2: Article with existing HTML tags**

Input:
```
<h1>Suomen talouskasvu hidastuu tänä vuonna</h1>
<p>Asiantuntijoiden mukaan inflaatio ja globaalit epävarmuudet vaikuttavat talouden näkymiin.</p>
<h2>Kuluttajat varovaisia</h2>
<p>Monet kotitaloudet ovat vähentäneet kulutustaan.</p>
<blockquote>"Kuluttajien luottamus on historiallisen alhaisella tasolla", sanoo ekonomisti Anna Korhonen.</blockquote>
```
Output:
```
<title>Suomen talouskasvu hidastuu tänä vuonna</title>

<lead>Asiantuntijoiden mukaan inflaatio ja globaalit epävarmuudet vaikuttavat talouden näkymiin.</lead>

<subtitle>Kuluttajat varovaisia</subtitle>

Monet kotitaloudet ovat vähentäneet kulutustaan.

<quote>"Kuluttajien luottamus on historiallisen alhaisella tasolla", sanoo ekonomisti Anna Korhonen.</quote>
```

**Example 3: Short article without clear sections**

Input:
```
Porvoon vanha silta suljetaan huoltotöiden ajaksi

Sillan korjaustyöt alkavat maanantaina ja kestävät viikon. Kaupunki suosittelee käyttämään vaihtoehtoisia reittejä.
```
Output:
```
<title>Porvoon vanha silta suljetaan huoltotöiden ajaksi</title>

Sillan korjaustyöt alkavat maanantaina ja kestävät viikon. Kaupunki suosittelee käyttämään vaihtoehtoisia reittejä.
```

### Most Important Rules (Repeated):
- Only use the four standard tags: `<title>`, `<lead>`, `<subtitle>`, `<quote>`. Remove any other tags or formatting.
- Always include exactly one `<title>` tag.
- Do NOT include `<lead>`, `<subtitle>`, or `<quote>` unless clearly justified by the content.
- If input is already correctly tagged using ONLY the four custom tags, pass it through unchanged.
- Do NOT make any changes to the original wording, facts, tone, or style of the article. Preserve journalistic integrity.

### Output Format:
Return the tagged text clearly formatted with ONLY the allowed custom tags and regular article text. Do NOT include explanations or additional comments.
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
    tag_transform_map = {
        'title': lambda text: f'<h1>{text.strip()}</h1>',
        'lead': lambda text: f'<h3>{text.strip()}</h3>',
        'subtitle': lambda text: f'<h4>{text.strip()}</h4>',
        'quote': lambda text: f'<p><i>«{text.strip()}»</i></p>',
    }

    def process_element_children(element):
        content = ''
        for child in element.contents:
            result = process_element(child)
            if result.strip():
                # Only wrap if it's not a block-level element
                if re.match(r'^<\s*(h1|h3|h4|p|blockquote|ul|ol|div|section)\b', result.strip()):
                    content += result.strip() + '\n'
                else:
                    content += f'<p>{result.strip()}</p>\n'
        return content

    def process_element(element):
        if isinstance(element, NavigableString):
            return escape(str(element))
        elif isinstance(element, Tag):
            tag_name = element.name
            text = ''.join([process_element(child) for child in element.contents])
            if tag_name in tag_transform_map:
                return tag_transform_map[tag_name](text)
            else:
                return process_element_children(element)
        else:
            return ''

    soup = BeautifulSoup(tagged_text, 'html.parser')
    html_text = process_element(soup)

    html_output = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
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
            print(f'fact-checker identified and corrected the following {len(response["list_of_critical_errors"])} critical errors:')
            for error in response['list_of_critical_errors']:
                print(f'...{error}')
        response = clean_generated_text(response['new_text'])
        response = response.replace('<tarkastettava_teksti_B>','').replace('</tarkastettava_teksti_B>','').strip()

    return response