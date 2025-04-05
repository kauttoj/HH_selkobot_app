import html
import json
import queue
import time
import PyPDF2
import docx
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import threading
from streamlit_autorefresh import st_autorefresh
import backend # Assuming 'backend.py' exists in the same directory
import streamlit.components.v1 as components
from st_copy_to_clipboard import st_copy_to_clipboard

GUI_DEBUG = 0 # Set to 0 to use actual backend functions

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="HH Selbobot", layout="wide")

# ---------- CUSTOM STYLES ----------
# ... (Styles remain the same) ...

HTML_style = '''<style>
    .mybox {
        box-sizing: border-box;
        padding: 10px;
        background-color: #f0f0f0;
        border: 1px solid black;
        border-radius: 4px;
        font-family: sans-serif;
        overflow-y: auto;
        height: 100%;
        width: 100%;
        white-space: normal;
    }
</style>'''

st.markdown("""
    <style>
    .title-font {
        font-size: 42px;
        font-weight: 700;
        font-family: 'Segoe UI', sans-serif;
        color: #4B8BBE;
        margin-bottom: 17px;
        margin-top: 0px;
    }
        .parsed-disabled { color: gray; font-style: italic; }
        textarea[aria-label="Ask something about the text..."] { min-height: 3em !important; }
        button:disabled { background-color: #ddd !important; color: #888 !important; cursor: not-allowed !important; }
    textarea { border: 1px solid black !important; background-color: #f0f0f0 !important; border-radius: 4px; }    
    </style>
""", unsafe_allow_html=True)

# ---------- TITLE & INSTRUCTIONS ----------
# ... (Title and Instructions remain the same) ...
st.markdown('<div class="title-font">HH Selbobot</div>', unsafe_allow_html=True)
st.markdown("""
<div style='font-size: 0.85em; line-height: 1.4; padding: 10px 15px; background-color: #f9f9f9; border: 1px solid #ddd; border-radius: 6px;'>
<b>📘 Usage Instructions</b><br><br>
<b>1. Input text:</b> Insert raw or parsed text by writing, copy-pasting, or reading from a file.<br>
<b>2. Parse raw text:</b> If not parsing yourself, use automatic parsing. Manually fix any mistakes.<br>
<b>3. Set parameters:</b> Try different <i>models</i>, <i>prompts</i>, and <i>agent types</i>. Keep temperature close to <b>0.40</b>.<br>
<b>4. Simplify:</b> Simplify parsed text. This takes <b>5–30 seconds</b> (depending on backend). You can generate and view multiple versions.<br><br>
<b>Agents:</b> Muuntaja = writer only, Muuntaja & faktantarkastaja = writer followed by fact correction.<br>
<b>80% pituusprompti:</b> Include prompt instruction to aim for ~80% of original text length.<br><br>
<span style='color: #555;'>💡 "Clean view" shows cleaned HTML rendered output without parsing tags.</span>
</div>
""", unsafe_allow_html=True)

example_text = '''<title>Syyrialainen huippukokki avasi ravintolan Helsinkiin – nyt Tawook Lab kerää ylistäviä arvioita</title>
<lead>Helsinkiin avattu syyrialainen ravintola on saanut erinomaisen vastaanoton. Ravintolaa pyörittävä pariskunta kertoo, mikä Tawook Labista tekee poikkeuksellisen.</lead>
Google-arvosteluista on tullut tärkeä tekijä ravintoloiden erottautumiselle. Asiakas jättää käynnistään merkinnän, ja arvosteluiden keskiarvo komeilee Googlen hakutuloksissa ravintolan nimen yhteydessä.
Harva ravintola on kerännyt niin yksimielisen kiittäviä arvioita kuin Helsingin keskustaan joulukuussa avattu Tawook Lab. Pieni syyrialainen ravintola oli saanut puolessa vuodessa Googlessa runsaat 300 arviota, lähes kaikissa täydet viisi tähteä.
<subtitle>Vahvuuksia</subtitle>
Google-arvosteluissa hehkutetaan Tawook Labin ruokaa, palvelua ja kotoisaa tunnelmaa.
Jututetaanpa yhtä asiakkaista paikan päällä. Lounaskeittoa lusikoiva Stephen Webb kertoo löytäneensä Tawook Labin ystävänsä suosituksesta. Ensimmäisellä käyntikerralla ravintolassa syntyi luonteva keskustelu, johon osallistuivat sekä paikan pitäjät että toisiaan ennalta tuntemattomat asiakkaat.
<quote>”Juttelimme elämästä, niitä näitä. Nyt syön täällä kolmatta kertaa. Tykkään paikan yhteisöllisyydestä. Ravintoloissa hyvin harvoin törmää tällaiseen olohuonemaisuuteen”, Webb luonnehtii.</quote>
Palaute lämmittää ravintolaa pyörittävää syyrialaista avioparia. Kasem Al-Hallak vastaa ruoasta ja Waed Hejazi palvelusta.
<quote>”Eräs asiakas kertoi, että olemme vaimon kanssa jo julkkiksia hänen työpaikallaan. Puolet ravintolan menestyksestä muodostuu ruoasta ja puolet vieraanvaraisuudesta. Vaimoni ansiosta asiakkaat palaavat uudestaan ja uudestaan”, Kasem Al-Hallak sanoo.</quote>
Pariskunta huomioi haastattelun aikana jokaisen ravintolaan saapuvan asiakkaan ja vaihtaa heidän kanssaan vähintään muutaman sanan, kenen kanssa mistäkin.
Asiakaspalvelu on niin välitöntä, että on vaikea uskoa, että Tawook Lab on Hejazin ensimmäinen työpaikka. Pariskunta asui aikaisemmin Saudi-Arabiassa, jossa Al-Hallak työskenteli ravintoloissa ja Hejazi hoiti maan tavan mukaisesti perhettä kotona.
Hejazin lause keskeytyy liikutuksen kyyneliin, kun hän kertoo ravintolassa saamastaan palautteesta.
<quote>”Asiakas halasi ja kertoi, että hänelle tuli tunne kuin olisi palannut äitinsä ruokapöytään. Olemme järjestäneet kotona paljon juhlia, rakastan uusien ihmisten kohtaamista ja olen aina tykännyt kestitsemisestä. Ravintola on meille kuin toinen koti, ja aviomiehen kanssa on ihanaa tehdä töitä – vaikka välillä otammekin vähän yhteen. Työn ainoa raskas puoli on työpäivien pituus”, Hejazi kertoo.</quote>
<subtitle>Uhkia</subtitle>
Pariskunnan kolme lasta auttavat tarpeen mukaan ravintolassa. 17-vuotias tytär hoitaa somen. Al-Hallakin ja Hejazin työpäivät venyvät helposti silti aamusta iltamyöhään.
Al-Hallakille armoton työtahti ei ole uutta. Hän laskee työskennelleensä ravintola- ja hotellialalla kaikkiaan 20 vuotta. Hän nousi pääkokiksi armenialaista ruokaa tarjoavan Lusin-ketjun ravintolassa, joka piti monta vuotta kärkipaikkaa Saudi-Arabian pääkaupungin Riadin Tripadvisor-listauksissa. Lisäksi Al-Hallak konseptoi Mira Foods -konsernin muita ravintoloita. Kiireisimpinä aikoina vapaapäivät jäivät niin vähiin, että Hejazi soitti jo miehensä pomolle.
<quote>”Se oli rankkaa aikaa, olin aivan poikki. Samaan aikaan vaimo hoiti kaiken kotona. Muutimme Suomeen, ja nyt yhteistä aikaa kertyy lähes 24 tuntia vuorokaudessa. Menestys on minun juttuni. Teen töitä kellon ympäri, vaikka sitten terveyteni uhalla. Kun kotona nukkumaan mennessä selailemme kiittäviä Google-arvioita, päivä on pelastettu”, Al-Hallak kertoo.</quote>
Taskuun saisi silti jäädä enemmänkin.
<quote>”Liiketilan vuokra on tosi korkea ja raaka-aineet kalliita. Ravintolan tuotto on ihan hyvä, mutta ei niin hyvä kuin sen näillä työtunneilla kuuluisi olla”, Al-Hallak sanoo.</quote>
<subtitle>Haasteita</subtitle>
Perhe muutti Suomeen vuoden 2020 taitteessa, juuri ennen koronapandemian puhkeamista.
Al-Hallak kertoo käyttäneensä paljon aikaa maan tapojen ja sääntöjen opetteluun. Hän oli mukana startup-yrityksessä ja opiskeli sekä yrityksen perustamista että tietotekniikkaa, mistä on ollut hyötyä ravintolan järjestelmien kanssa. Suomen kieli ei vielä taivu, mutta Al-Hallak sanoo osaavansa sitä riittävästi tullakseen toimeen.
Tawook Lab sijaitsee Citykäytävässä Helsingin ydinkeskustassa, vilkkaan kulkuväylän varrella. Haastattelu tehdään hiljaisempana hetkenä iltapäivällä, lounasajan jälkeen. Moni katselee ruokalistaa ravintolan ulkopuolella mutta kävelee lopulta ohi. Syyrialaisen ruoan eksoottisuus on sekä vahvuus että haaste.
<quote>”Uuden maistaminen voi olla pelottavaa, ja ehkä on helpompi valita pizza”, Al-Hallak miettii.</quote>
<quote>”Mutta kun suomalaisen saa kerran houkuteltua tänne, niin hän alkaa käydä koko ajan. Lounasaikaan meillä käy keskimäärin 40–50 asiakasta. Perjantai on hiljaisempi, koska se on monessa työpaikassa etäpäivä. Keskiviikot ovat vilkkaita.”</quote>
<subtitle>Mahdollisuuksia</subtitle>
Al-Hallakin työtausta fine dining -kokkina maistuu ja näkyy Tawook Labin ruoassa. Perinteistä ja modernia yhdistävät annokset ovat huoliteltuja. Ote on kokeileva ja leikkisä. Annosten hinnat asettuvat kymmenen ja kahdenkymmenen euron väliin.
<quote>”Teemme lujasti töitä sen eteen, että pystymme tarjoamaan suomalaisille oikean syyrialaisen ruokakokemuksen, pienellä fuusio-otteella”, Al-Hallak kuvailee.</quote>
<quote>”Helsinki on hieno kaupunki, mutta tänne tarvitaan enemmän muitakin kuin sushi-, pizza- ja hampurilaisravintoloita. Pari tuttua ravintoloitsijaa suositteli minulle lounasbuffetia, mutta siihen en lähde. Haluamme muuttaa suomalaisten suhtautumista ruokaan, ja laadukkaat lautasannokset vahvistavat myös ravintolan brändiä. Valmistamme omin käsin kaiken hummuksesta kastikkeisiin, ja silloin maut oikeasti tuntee.”</quote>
Tawook Labin taustalla on yksityinen rahoittaja. Al-Hallakin ja Hejazin suunnitelmissa on perustaa ravintoloita ympäri Suomea, ehkä maailmallekin. Tawook Labissa käy asiakkaita ympäri maailmaa, ja heidän kanssaan tulee juteltua monenlaista.
<quote>”Mumbaista kävi yksi porukka, ja he kyselivät, olisimmeko kiinnostuneita laajentamaan sinne. Kuala Lumpurista Malesiasta tiedusteltiin 7/11-tyyppiseen kioskiketjuun, mutta sellaiseen emme ole valmiita.”</quote>'''

# ---------- SESSION STATE INIT ----------
defaults = {
    "status": "Ready",
    "input_text": '',
    "parsed_text": "",
    "output_text": "",
    "parsed_available": False,
    "last_saved_raw": "",
    "tokencount": False,
    "output_is_html": True,
    "active_text": 'RAW',
    "html_input_text": '',
    'additional_status_text':'',
    "html_parsed_text": '',
    "view_only_mode": False,
    "output_versions": [],
    "output_metadata": [],
    "selected_version_index": 0,
    "compute_thread": None,
    "result_queue": queue.Queue(),
    "parse_result": None,
    "simplify_result": None,
    "model": backend.WRITER_models[0] if hasattr(backend, 'WRITER_models') and backend.WRITER_models else None,
    "agent_type": "Muuntaja",
    "prompt_type": list(backend.PROMPT_writer.keys())[0] if hasattr(backend, 'PROMPT_writer') and backend.PROMPT_writer else None,
    #"temperature": 0.40,
}
defaults["html_input_text"] = backend.tagged_text_to_noncolored_html(defaults["input_text"])
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# ---------- FUNCTION DEFINITIONS ----------

# --- Thread target functions ---
# (parse_text_thread and simplify_text_thread remain the same as the previous version)
def parse_text_thread(raw_text_arg: str,result_queue):
    print("Parsing started in background thread.")
    raw_text = raw_text_arg
    parsed_result,clean,error_message = None,None,None
    try:
        if GUI_DEBUG:
            print('Running dummy parsing...'); time.sleep(5)
            parsed_result = f"<title>{raw_text_arg}</title>"
            clean = backend.tagged_text_to_noncolored_html(parsed_result)
        else:
            print('Running actual backend parsing...');
            parsed_result = backend.parse_text(raw_text, st.session_state);
            clean = backend.tagged_text_to_noncolored_html(parsed_result)
            print('Backend parsing complete.')
    except Exception as e: error_message = f"Error during parsing thread: {e}"; print(f"⚠️ {error_message}")
    finally:
        print("Parsing finished in background thread.")
        result_queue.put({'raw':parsed_result,'html':clean})

def simplify_text_thread(parsed_text_arg: str, settings: dict,result_queue):
    print("Simplification started in background thread.")
    parsed_text_in = parsed_text_arg
    raw,clean,error_message = None,None,None
    try:
        if GUI_DEBUG:
            print('Running dummy simplification...'); time.sleep(5)
            raw = "[SIMPLIFIED]\n" + parsed_text_in
            clean = backend.tagged_text_to_noncolored_html(raw)
            print('Dummy simplification complete.')
        else:
            print('Running actual backend simplification...');
            raw = backend.simplify_text(parsed_text_in, settings)
            clean = backend.tagged_text_to_noncolored_html(raw)
            if not raw:
                error_message = "Backend simplification returned no result."; print(f'⚠️ {error_message}')
                print('Backend simplification complete.')
    except Exception as e: error_message = f"Error during simplification thread: {e}"; print(f"⚠️ {error_message}")
    finally:
        print("Simplification finished in background thread.")
        # Store settings used along with the result
        result_queue.put({'raw':raw,'html':clean,'settings':settings})

# --- Other helper functions ---
# (get_status_color, show_prompt_dialog, process_file, reset_all_fields remain the same)
def get_status_color(status):
    return {
        "Ready": "lightgreen",
        "parsing": "khaki", "parsing_done": "lightgreen", "simplifying": "lightblue", "simplifying_done": "lightgreen", }.get(
        status, "lightgray")

@st.dialog("Prompt Description")
def show_prompt_dialog():
    agent = st.session_state.get('agent_type')
    prompt_type = st.session_state.get('prompt_type')

    PROMPT = backend.PROMPT_writer[prompt_type]

    st.markdown(
        """
        <style>
        div[data-testid="stDialog"] div[role="dialog"]:has(.big-dialog) {
            width: 80vw;
            height: 80vh;
            display: flex;
            flex-direction: column;
        }

        div[data-testid="stDialog"] div[role="dialog"]:has(.big-dialog) > div {
            overflow: auto;
            max-height: calc(80vh - 3rem); /* adjust to allow header spacing */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## Writer prompt")
    st.code(PROMPT, language="text")

    if 'fakta' in agent:
        st.markdown("## Fact-checker prompt")
        st.code(backend.PROMPT_error_correct, language="text")

    st.markdown("## Parsing prompt")
    st.code(backend.PROMPT_parse_text, language="text")

    st.html("<span class='big-dialog'></span>")


def process_file():
    uploaded_file = st.session_state["uploaded_file"]

    file_type = uploaded_file.name.split('.')[-1].lower()

    if file_type == "txt":
        # Read text file
        new_text = uploaded_file.read().decode("utf-8").strip()

    elif file_type == "pdf":
        # Read PDF file
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ''
        new_text = text.strip()

    elif file_type == "docx":
        # Read Word file
        doc = docx.Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        new_text = text.strip()

    else:
        st.error(f'⚠️ Unsupported file type uploaded, only .txt, .pdf and .docx are supported')
        return

    if st.session_state.active_text == 'RAW':
        st.session_state.input_text = new_text
        st.session_state.html_input_text = backend.tagged_text_to_noncolored_html(new_text)
    else:
        st.session_state.parsed_text = new_text
        st.session_state.html_input_text = backend.tagged_text_to_noncolored_html(new_text)

    st.toast(f"Loaded content from {uploaded_file.name}", icon="📄")

def reset_all_fields():
    st.session_state.status = "Ready";
    st.session_state.input_text = ""
    st.session_state.parsed_text = ""
    st.session_state.html_input_text = ""
    st.session_state.html_parsed_text = ""
    st.session_state.output_versions = []
    st.session_state.output_metadata = []
    st.session_state.selected_version_index = 0
    st.session_state.parsed_available = False
    st.toast("All fields reset.", icon="🔄")

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown('<div style="position: relative; top: -20px; left: 0; font-size: 10px; color: gray;"><strong>v1.0 JanneK</strong></div>', unsafe_allow_html=True)
    st.markdown("### Status")
    status_text = st.session_state.status.replace("_", " ").upper()

    # Add animated spinner if status is actively running
    is_loading = st.session_state.status in ["parsing", "simplifying"]
    spinner_html = '<div class="dot-flashing"></div>' if is_loading else ''

    st.markdown(
        f"""
        <style>
        .status-box {{
            padding: 10px;
            background-color: {get_status_color(st.session_state.status)};
            border-radius: 5px;
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: bold;
            font-family: sans-serif;
        }}
        .dot-flashing {{
            position: relative;
            width: 16px;
            height: 16px;
            border-radius: 6px;
            background-color: #333;
            animation: dotFlashing 1s infinite linear alternate;
        }}
        @keyframes dotFlashing {{
            0% {{ background-color: #333; }}
            50%, 100% {{ background-color: #ddd; }}
        }}
        </style>
        <div class="status-box">{spinner_html}{status_text}</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(f'<div style="font-size: 12px; color: black;"><strong>{st.session_state.additional_status_text}</strong></div>', unsafe_allow_html=True)

    #st.markdown( f"<div style='padding: 10px; background-color: {get_status_color(st.session_state.status)}; border-radius: 5px;'><strong>{status_text}</strong></div>", unsafe_allow_html=True )
    st.markdown("---")
    st.header("⚙️ Settings")

    # Fixed: Settings are NOT disabled during computation
    settings_disabled = False # Keep settings enabled

    model_list = backend.WRITER_models if hasattr(backend, 'WRITER_models') else []
    model_index = model_list.index(st.session_state.model) if st.session_state.model in model_list else 0
    agent_list = ["Muuntaja", "Muuntaja & faktantarkastaja"]
    agent_index = agent_list.index(st.session_state.agent_type) if st.session_state.agent_type in agent_list else 0
    prompt_list = list(backend.PROMPT_writer.keys()) if hasattr(backend, 'PROMPT_writer') else []
    prompt_index = prompt_list.index(st.session_state.prompt_type) if st.session_state.prompt_type in prompt_list else 0

    st.selectbox("Kielimalli", model_list, key="model", index=model_index, disabled=settings_disabled)
    st.radio("Bottityyppi", agent_list, key="agent_type", index=agent_index, disabled=settings_disabled)
    col_prompt, col_button = st.columns([2, 1])
    with col_prompt: st.radio("Prompti", prompt_list, key="prompt_type", index=prompt_index, disabled=settings_disabled)
    with col_button:
        # Disable Show button only if dialog needs settings that might change mid-compute? Unlikely. Keep enabled.
        if st.button("View", disabled=False):
            show_prompt_dialog()
    st.checkbox("80% pituusprompti", key="tokencount", value=st.session_state.tokencount, disabled=settings_disabled)
    st.number_input("Lämpötila", min_value=0.0, max_value=1.0, step=0.01, key="temperature", value=0.4, disabled=settings_disabled)

# ---------- MAIN LAYOUT ----------
col_input, col_output = st.columns(2)

# --- View Toggle Buttons ---
def activate_input_text(): st.session_state.active_text = 'RAW'
def activate_parsed_text(): st.session_state.active_text = 'PARSED'
col1, col2,_ = st.columns([1,1,2])

# Fixed: View toggles are NOT disabled during computation
toggle_disabled = False # Keep enabled

with col1: # Raw view button
    with stylable_container( "toggle_btn1", css_styles=f""" button {{ background-color: {"#e57373" if st.session_state.active_text == 'RAW' else "lightgray"} !important; border: 1px solid black !important; color: black !important; }} """ ):
        st.button("Show Raw", on_click=activate_input_text, use_container_width=True, disabled=toggle_disabled)
with col2: # Parsed view button
    with stylable_container( "toggle_btn2", css_styles=f""" button {{ background-color: {"#e57373" if st.session_state.active_text == 'PARSED' else "lightgray"} !important; border: 1px solid black !important; color: black !important; }} """ ):
        st.button("Show Parsed", on_click=activate_parsed_text, use_container_width=True, disabled=toggle_disabled)

# --- Input Box ---
with col_input:
    col_input_header, col_input_toggle,col_copytext = st.columns([3,1,1])
    with col_input_header:
        st.subheader("📝 Input texts")
    with col_input_toggle:
        # Fixed: Clean view toggle NOT disabled during computation
        st.checkbox( "Clean view", key="view_only_mode", value=st.session_state.view_only_mode, disabled=False )

    new_text = ''
    #--------------------------------
    if st.session_state.active_text == 'RAW':
        if not st.session_state.view_only_mode: # Editable Raw
            new_text = st.session_state.input_text
        else: # View Raw HTML (Never disabled)
            new_text = st.session_state.html_input_text
    else: # PARSED view active
        if not st.session_state.view_only_mode: # Editable Parsed
            new_text = st.session_state.parsed_text
        else: # View Parsed HTML (Never disabled)
            # Ensure html_parsed_text is up-to-date before displaying
            new_text = st.session_state.html_parsed_text

    #------------------------------
    with col_copytext:
        pass
        #print(f'col_input copy-paste obtained with length {len(new_text)}')
        #st_copy_to_clipboard(new_text, r"🗐") # ,key='input_copypaste'

    # Fixed: Text area is disabled ONLY if it's editable AND computing
    # We disable editing to prevent conflicts with background threads reading the data.
    input_disabled = (st.session_state.status in ["parsing", "simplifying"]) and not st.session_state.view_only_mode

    if st.session_state.active_text == 'RAW':
        if not st.session_state.view_only_mode: # Editable Raw
            new_text = st.text_area("Raw text", value=st.session_state.input_text, height=600, key="text_area_input", label_visibility="collapsed", disabled=input_disabled)
            # Only update state if editing is allowed (not disabled) and text actually changed
            if not input_disabled and new_text != st.session_state.input_text:
                 st.session_state.input_text = new_text
                 st.session_state.html_input_text = backend.tagged_text_to_noncolored_html(new_text)
                 st.session_state.parsed_text = ""; st.session_state.html_parsed_text = ""; st.session_state.parsed_available = False
                 st.rerun()
        else: # View Raw HTML (Never disabled)
            components.html(f"{HTML_style}<div class='mybox'>{st.session_state.html_input_text}</div>", height=600, scrolling=True)
        #st.session_state.input_current_text = st.session_state.html_input_text
    else: # PARSED view active
        if not st.session_state.view_only_mode: # Editable Parsed
            new_text = st.text_area("Parsed text", value=st.session_state.parsed_text, height=600, key="text_area_parsed", label_visibility="collapsed", disabled=input_disabled)
            # Only update state if editing is allowed (not disabled) and text actually changed
            if not input_disabled and new_text != st.session_state.parsed_text:
                st.session_state.parsed_text = new_text
                st.session_state.html_parsed_text = backend.tagged_text_to_noncolored_html(new_text)
                st.session_state.parsed_available = len(new_text) > 10
                st.rerun()
        else: # View Parsed HTML (Never disabled)
            # Ensure html_parsed_text is up-to-date before displaying
            if not st.session_state.html_parsed_text and st.session_state.parsed_text:
                 st.session_state.html_parsed_text = backend.tagged_text_to_noncolored_html(st.session_state.parsed_text)
            components.html(f"{HTML_style}<div class='mybox'>{st.session_state.html_parsed_text}</div>", height=600, scrolling=True)
            #st.html(f"{st.session_state.html_parsed_text}")
        #st.session_state.input_current_text = st.session_state.html_parsed_text

# --- Output Box ---
with col_output:
    col_output_header, col_output_toggle,col_copytext = st.columns([3,1,1])
    with col_output_header:
        st.subheader("📄 Simplified text")
    with col_output_toggle:
        # Fixed: Clean view toggle NOT disabled during computation
        st.checkbox("Clean view", key="output_is_html",disabled=False) # st.session_state.output_is_html value=True

    #----------------------------
    # # Fixed: Output selection NOT disabled during computation
    # output_disabled = False
    #
    # new_text=''
    # if st.session_state.output_versions:
    #     # Regenerate labels each time in case metadata changes (though it shouldn't here)
    #     version_labels = [
    #         f"ver. {i+1} ({meta['model']}, {meta['prompt_type']}, {meta['agent_type']}, {meta['temperature']}, tokencount={meta['tokencount']})"
    #         for i, meta in enumerate(st.session_state.output_metadata)
    #     ]
    #     if version_labels: # Only show if versions exist
    #         # Display selected version (check index validity again)
    #         if 0 <= st.session_state.selected_version_index < len(st.session_state.output_versions):
    #             selected_output = st.session_state.output_versions[st.session_state.selected_version_index]
    #             if not st.session_state.output_is_html: # Raw view
    #                 new_text= selected_output['raw']
    #             else: # HTML view
    #                 new_text = selected_output['html']
    #     else: # No versions to select (shouldn't happen if output_versions check passed)
    #         raise Exception('BAD OUTPUT DATA')

    #----------------------------
    output_disabled = False
    copypaste_text = ''
    if st.session_state.output_versions:
        # Regenerate labels each time in case metadata changes (though it shouldn't here)
        version_labels = [
            f"ver. {i+1} ({meta['model']}, {meta['prompt_type']}, {meta['agent_type']}, {meta['temperature']}, tokencount={meta['tokencount']})"
            for i, meta in enumerate(st.session_state.output_metadata)
        ]
        current_index = st.session_state.selected_version_index
        # Validate index
        if not (0 <= current_index < len(version_labels)):
            current_index = max(0, len(version_labels) - 1)

        if version_labels: # Only show if versions exist
            selected_label = st.selectbox( "Output version", version_labels, index=current_index, key="selected_version_label", disabled=output_disabled )
            # Update selected index if changed and not disabled
            if not output_disabled:
                new_index = version_labels.index(selected_label)
                if new_index != st.session_state.selected_version_index: st.session_state.selected_version_index = new_index; st.rerun()

            # Display selected version (check index validity again)
            if 0 <= st.session_state.selected_version_index < len(st.session_state.output_versions):
                selected_output = st.session_state.output_versions[st.session_state.selected_version_index]
                if not st.session_state.output_is_html: # Raw view
                    st.text_area("Output Text Raw", value=selected_output['raw'], height=600, label_visibility="collapsed", disabled=False, key=f"output_raw_{st.session_state.selected_version_index}")
                    #st.session_state.output_current_text = selected_output['raw']
                    copypaste_text = selected_output['raw']
                else: # HTML view
                    components.html(f"{HTML_style}<div class='mybox'>{selected_output['html']}</div>", height=600, scrolling=True)
                    #st.session_state.output_current_text = selected_output['html']
                    copypaste_text = selected_output['html']
            else: # Handle edge case where index becomes invalid after selection somehow
                 raise Exception('BAD INDEX')
        else: # No versions to select (shouldn't happen if output_versions check passed)
             raise Exception('BAD OUTPUT DATA')
    else: # No output generated yet
        st.markdown(f"""<div style="height:600px; overflow:auto; padding:10px; background-color:#f5f5f5; border:1px solid #ddd; border-radius:4px; color:#999;">No output generated yet</div> """, unsafe_allow_html=True)

# ---------- ACTION BUTTONS ----------
col_readfile, col_parse,col_sample,col_dummy1, col_simplify,col_copypaste,col_reset = st.columns([1,1,0.75,2,1,1,1])

with col_readfile:
    st.file_uploader("Upload file",on_change=process_file,type=["txt", "pdf", "docx"],label_visibility="collapsed",key="uploaded_file")

# Fixed: Parse/Simplify buttons are the ONLY ones disabled by status
parse_disabled = ( st.session_state.status != "Ready" or not st.session_state.input_text )
with col_parse:
    if st.button("🔍 Parse Input", disabled=parse_disabled, use_container_width=True):
        if st.session_state.compute_thread is None:
            if len(st.session_state.input_text) < 10:
                st.error(f'⚠️ Raw text too short {len(st.session_state.input_text)}, nothing to parse')
            else:
                print("Starting parsing thread...");
                st.session_state.status = "parsing";
                st.session_state.additional_status_text = 'calling LLM...'
                st.session_state.parse_result = None
                thread_args = (st.session_state.input_text,st.session_state.result_queue);
                #thread = threading.Thread(target=parse_text_thread, args=thread_args, daemon=True)
                thread = threading.Thread(
                    target=parse_text_thread,
                    args=thread_args,
                    daemon=True
                )
                st.session_state.compute_thread = thread
                thread.start()
                print("Parsing thread started")
                st.rerun()

with col_sample:
    if st.button("Load example", disabled=False, use_container_width=True):
        st.session_state.parsed_text = example_text
        st.session_state.html_parsed_text = backend.tagged_text_to_noncolored_html(example_text)
        st.session_state.parsed_available=True
        st.session_state.active_text = 'PARSED'
        st.rerun()

with col_simplify:
    simplify_disabled = ( st.session_state.status != "Ready" or not st.session_state.parsed_available )
    if st.button("🧠 Simplify Text", disabled=simplify_disabled, use_container_width=True):
        if st.session_state.compute_thread is None:
            if len(st.session_state.parsed_text) < 10:
                st.error(f'⚠️ Parsed text too short {len(st.session_state.parsed_text)}, nothing to simplify')
            elif not (('<title>' in st.session_state.parsed_text) and ('</title>' in st.session_state.parsed_text)):
                st.error(f'⚠️ Input text does not appear to be parsed. Parse it first with proper tags.')
            else:
                print("Starting simplification thread...")
                st.session_state.status = "simplifying";
                st.session_state.additional_status_text='calling LLM...'
                st.session_state.simplify_result = None
                current_settings = {"model": st.session_state.model, "agent_type": st.session_state.agent_type, "prompt_type": st.session_state.prompt_type, "temperature": st.session_state.temperature, "tokencount": st.session_state.tokencount };
                thread_args = (st.session_state.parsed_text, current_settings,st.session_state.result_queue)
                #thread = threading.Thread(target=simplify_text_thread, args=thread_args, daemon=True)
                thread = threading.Thread(
                    target=simplify_text_thread,
                    args=thread_args,
                    daemon=True
                )
                st.session_state.compute_thread = thread;
                thread.start();
                print("Simplification thread started. Triggering rerun.");
                st.rerun()

# Fixed: Reset button only disabled if actively computing
with col_copypaste:
    st_copy_to_clipboard(copypaste_text,r"📋 Copy text") #key='output_copypaste')

with col_reset:
    reset_disabled = st.session_state.status in ["parsing", "simplifying"]
    st.button( "🔄 Reset texts", on_click=reset_all_fields, disabled=reset_disabled, use_container_width=True )

# ---------- AUTO-REFRESH and THREAD COMPLETION HANDLING ----------
refresh_interval_ms = 500

if not st.session_state.result_queue.empty():
    result = st.session_state.result_queue.get()
    st.session_state.status = "parsing_done" if st.session_state.status=='parsing' else "simplifying_done"
    st.session_state.thread_data = result
    st.session_state.compute_thread = None
    st.session_state.additional_status_text = ''
    st.rerun()  # Rerun immediately to show the updated status

if st.session_state.status == "parsing":
    st_autorefresh(interval=refresh_interval_ms, key="parse_refresher")
    if st.session_state.compute_thread and not st.session_state.compute_thread.is_alive():
        print("Parse thread finished. Updating state.");
        st.session_state.status = "parsing_done";
        st.session_state.compute_thread = None;
        st.rerun()
elif st.session_state.status == "parsing_done":
    print("Processing parsing results...") # Debug print
    result_info = st.session_state.thread_data
    if result_info:

        parsed_data = result_info
        print(f"Parsed data received") # Debug print
        st.session_state.parsed_text = parsed_data['raw']
        # Fixed: Ensure HTML version is generated *before* potential display
        st.session_state.html_parsed_text = parsed_data['html']
        st.session_state.parsed_available = True
        # Fixed: Explicitly set active_text to trigger view switch *before* rerun
        if st.session_state.active_text != 'PARSED':
             st.session_state.active_text = 'PARSED'
        #st.toast("Parsing completed ✅", icon="🔍")
        print("Parsing state updated successfully.") # Debug print

    # Reset status and clear result *before* final rerun
    st.session_state.status = "Ready";
    st.session_state.thread_data = None;
    print("Resetting status to idle and triggering final rerun for parsing.") # Debug print
    st.rerun() # Rerun to clear status message and potentially show new view/data

elif st.session_state.status == "simplifying":

    st_autorefresh(interval=refresh_interval_ms, key="simplify_refresher")
    if st.session_state.compute_thread and not st.session_state.compute_thread.is_alive():
        print("Simplify thread finished. Updating state.");
        st.session_state.status = "simplifying_done";
        st.session_state.compute_thread = None;
        st.rerun()

elif st.session_state.status == "simplifying_done":
    print("Processing simplification results...") # Debug print
    result_info = st.session_state.thread_data
    if result_info:

        print("Simplification data received.") # Debug print
        new_output_data = {'raw':result_info['raw'],'html':result_info['html']}
        # Get settings used for this run from the result dict
        metadata = result_info["settings"]

        st.session_state.output_versions.append(new_output_data)
        st.session_state.output_metadata.append(metadata)

        # Fixed: Ensure index points to the newly added item
        st.session_state.selected_version_index = len(st.session_state.output_versions) - 1
        #st.toast(f"Simplification version {len(st.session_state.output_versions)} created ✅", icon="✨")

        print("Simplification state updated successfully.") # Debug print

    # Reset status and clear result *before* final rerun
    st.session_state.status = "Ready";
    st.session_state.simplify_result = None;
    print("Resetting status to idle and triggering final rerun for simplification.") # Debug print
    st.rerun() # Rerun to clear status message and show new output/selection
