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

GUI_DEBUG = 1 # Set to 0 to use actual backend functions

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="HH Selbobot", layout="wide")

# ---------- CUSTOM STYLES ----------
# ... (Styles remain the same) ...
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
    .html-text-box { padding: 10px; background-color: #f0f0f0; border: 1px solid black; border-radius: 5px; font-family: sans-serif; overflow-y: auto; height: 600px; }
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
<b>3. Set parameters:</b> Try different <i>models</i>, <i>prompts</i>, and <i>agent types</i>. Recommended temperature is close to <b>0.40</b>.<br>
<b>4. Simplify:</b> Simplify parsed text. This takes <b>5–30 seconds</b> (depending on backend). You can generate and view multiple versions.<br><br>
<b>Agents:</b> Muuntaja = writer only, Muuntaja & faktantarkastaja = writer followed by fact correction.<br>
<b>80% pituusprompti:</b> Include prompt instruction to aim for ~80% of original text length.<br><br>
<span style='color: #555;'>💡 "Clean view" shows cleaned HTML rendered output without parsing tags.</span>
</div>
""", unsafe_allow_html=True)

# ---------- SESSION STATE INIT ----------
defaults = {
    "status": "Ready",
    "input_text": '''<title>Tehdas käynnistää mittavat rekrytoinnit Vantaalla – tarve 100 uudelle työntekijälle lähitulevaisuudessa</title>
<lead>Kansainvälinen energianhallintayhtiö Eaton siirsi toimintonsa Espoosta Vantaalle viime vuonna.</lead>
Energianhallintayhtiö Eaton aloittaa mittavat rekrytoinnit Vantaalla.
 Yhtiö tarvitsee tänä vuonna 70 uutta työntekijää viime vuonna valmistuneeseen kriittisten sähkönjakelujärjestelmien tehtaaseensa Tuupakkaan.
 Uusia ammattilaisia tarvitaan varsinkin tuotantoon laitteiden kokoonpanotöihin.
 Asennus- ja kokoonpanopuolen lisäksi tarvetta on testauksen ja varastologistiikan puolella.
 Myös työnjohtajille ja toimihenkilöille avautuu töitä vuoden sisällä etenkin hankinnan ja laadunvalvonnan puolella.
 <quote>Juhannukseen mennessä tavoite on rekrytoida 40 uutta työntekijää ja vuoden loppuun mennessä 70. Kasvamme lähitulevaisuudessa yhteensä noin 100 työntekijällä, ennakoi tehtaanjohtaja Petri Koskinen tiedotteessa.</quote>
Yhdysvaltalainen pörssiyhtiö Eaton on maailman suurimpia kolmivaiheisten varavoimajärjestelmien (UPS) valmistajia.
 Yhtiön tehdas valmistui Kehä III:n varrelle Tuupakkaan loppusyksystä 2023. Yritys siirsi toimintonsa Espoosta Vantaalle entistä suurempiin tiloihin.
 Tehtaassa kehitetään ja valmistetaan varavoimajärjestelmiä (UPS), jotka suojaavat elektronisia laitteita yleisimmiltä virtaongelmilta, kuten sähkökatkoksilta ja sähköverkon häiriöiltä.
 Varavoimajärjestelmiä tarvitaan datakeskuksissa, liike- ja teollisuusrakennuksissa sekä terveydenhuollon ja meriteollisuuden kohteissa.
 Eatonin tuotantolaitos työllistää tällä hetkellä yli 300 työntekijää. Petri Koskisen mukaan lisätyövoimalle tulee edelleen tarvetta lähivuosina yhtiön liiketoiminnan kasvaessa.''',
    "parsed_text": "",
    "output_text": "",
    "parsed_available": False,
    "chat_history": [],
    "last_saved_raw": "",
    "tokencount": False,
    "output_is_html": True,
    "active_text": 'RAW',
    "html_input_text": '',
    "html_parsed_text": '',
    "view_only_mode": False,
    "output_versions": [],
    "output_metadata": [],
    "selected_version_index": 0,
    "compute_thread": None,
    "result_queue": queue.Queue(),
    "input_current_text": 'no input text',
    "output_current_text": 'no output text',
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
        if len(raw_text) < 10: error_message = f'Input length {len(raw_text)}, nothing to parse'; print(f'⚠️ {error_message}')
        elif GUI_DEBUG:
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
    st.markdown('<div style="position: relative; top: -20px; left: 0; font-size: 10px; color: gray;"><strong>v1.3 (UI Fixes)</strong></div>', unsafe_allow_html=True)
    st.markdown("### Status")
    status_text = st.session_state.status.replace("_", " ").upper()
    st.markdown( f"<div style='padding: 10px; background-color: {get_status_color(st.session_state.status)}; border-radius: 5px;'><strong>{status_text}</strong></div>", unsafe_allow_html=True )
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
        if st.button("Show", disabled=False): show_prompt_dialog()
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
            st.html(f"<div class='html-text-box'>{st.session_state.html_input_text}</div>")
        st.session_state.input_current_text = st.session_state.html_input_text
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
            st.html(f"<div class='html-text-box'>{st.session_state.html_parsed_text}</div>")
        st.session_state.input_current_text = st.session_state.html_parsed_text

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
                    st.session_state.output_current_text = selected_output['raw']
                    copypaste_text = selected_output['raw']
                else: # HTML view
                    st.html(f"<div class='html-text-box'>{selected_output['html']}</div>")
                    st.session_state.output_current_text = selected_output['html']
                    copypaste_text = selected_output['html']
            else: # Handle edge case where index becomes invalid after selection somehow
                 raise Exception('BAD INDEX')
        else: # No versions to select (shouldn't happen if output_versions check passed)
             raise Exception('BAD OUTPUT DATA')
    else: # No output generated yet
        st.markdown(f""" <div style="height:600px; overflow:auto; padding:10px; background-color:#f5f5f5; border:1px solid #ddd; border-radius:5px; color:#999;"> No output generated yet. Press 'Simplify Text' after parsing. </div> """, unsafe_allow_html=True)

# ---------- ACTION BUTTONS ----------
col_readfile, col_parse, col_dummy1, col_simplify,col_copypaste,col_reset = st.columns([1, 1,2,1,1,1])

with col_readfile:
    st.file_uploader("Upload file",on_change=process_file,type=["txt", "pdf", "docx"],label_visibility="collapsed",key="uploaded_file")

# Fixed: Parse/Simplify buttons are the ONLY ones disabled by status
with col_parse:
    parse_disabled = ( st.session_state.status != "Ready" or not st.session_state.input_text )
    if st.button("🔍 Parse Input", disabled=parse_disabled, use_container_width=True):
        if st.session_state.compute_thread is None:
            print("Starting parsing thread...");
            st.session_state.status = "parsing";
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

with col_simplify:
    simplify_disabled = ( st.session_state.status != "Ready" or not st.session_state.parsed_available )
    if st.button("🧠 Simplify Text", disabled=simplify_disabled, use_container_width=True):
        if st.session_state.compute_thread is None:
            if len(st.session_state.parsed_text) < 10:
                st.error(f'⚠️ Parsed text too short {len(st.session_state.parsed_text)}, nothing to parse')
            elif not (('<title>' in st.session_state.parsed_text) and ('</title>' in st.session_state.parsed_text)):
                st.error(f'⚠️ Input text does not appear to be parsed. Parse it first with proper tags.')
            else:
                print("Starting simplification thread...");
                st.session_state.status = "simplifying";
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
    st.button( "🔄 Reset", on_click=reset_all_fields, disabled=reset_disabled, use_container_width=True )

# ---------- AUTO-REFRESH and THREAD COMPLETION HANDLING ----------
refresh_interval_ms = 500

if not st.session_state.result_queue.empty():
    result = st.session_state.result_queue.get()
    st.session_state.status = "parsing_done" if st.session_state.status=='parsing' else "simplifying_done"
    st.session_state.thread_data = result
    st.session_state.compute_thread = None
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
        st.toast("Parsing completed ✅", icon="🔍")
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
        st.toast(f"Simplification version {len(st.session_state.output_versions)} created ✅", icon="✨")

        print("Simplification state updated successfully.") # Debug print

    # Reset status and clear result *before* final rerun
    st.session_state.status = "Ready";
    st.session_state.simplify_result = None;
    print("Resetting status to idle and triggering final rerun for simplification.") # Debug print
    st.rerun() # Rerun to clear status message and show new output/selection
