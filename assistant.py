import streamlit as st
import requests
from PIL import Image
import io
import pytesseract
import torch
import whisper
import tempfile
import os
from streamlit_option_menu import option_menu


st.set_page_config(
    page_title="MAiA",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🧠 MAiA ")
st.subheader("Skriv, spela in eller skanna")

tab1, tab2, tab3 = st.tabs(["Huvudsidan", "Skriv anteckning", "Spela in möte"])
with tab1:
    st.subheader("Välkommen till MAiA")
    st.write("Detta är en app som sammanställer anteckningar för BUP")
    
    
with tab2:

    selected = option_menu(
    menu_title=None,
    options=["Skriv", "Spela in", "Skanna"],
    icons=["pencil", "mic", "camera"],
    orientation="horizontal",
)
    # Ljuduppladdning
    uploaded_audio = st.file_uploader("🎤 Ladda upp en ljudinspelning (.mp3/.wav/.m4a):", type=["mp3", "wav", "m4a"])
    transcribed_text = ""

    if uploaded_audio is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_audio.name)[1]) as tmp_file:
            tmp_file.write(uploaded_audio.read())
            tmp_path = tmp_file.name

        st.audio(tmp_path)
        with st.spinner("🔍 Transkriberar ljud..."):
            whisper_model = whisper.load_model("base")
            result = whisper_model.transcribe(tmp_path, language="sv")
            transcribed_text = result["text"]

    transcribed_text = st.text_area("🎧 Transkriberad text (redigerbar):", value=transcribed_text, height=150)

    # Två kolumner: text och bilder
    col1, col2 = st.columns(2)
    with col1:
        user_text = st.text_area("✏️ Din egen text:", height=150)
    with col2:
        uploaded_files = st.file_uploader("📷 Ladda upp bilder:", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    image_caption = ""

    # Bildanalys via Tesseract OCR
    if uploaded_files:
        with st.expander("📎 Extraherad text från bilder"):
            for i, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption=f"Bild {i+1}", use_container_width=True)
                extracted = pytesseract.image_to_string(
                    image,
                    lang="swe",
                    config="--tessdata-dir /opt/homebrew/share/tessdata"
                )
                st.markdown(f"**Text från Bild {i+1}:**\n\n{extracted}")
                image_caption += f"\nBild {i+1}:\n{extracted}\n"

    # Välj mall
    st.markdown("---")
    st.subheader("📄 Välj mall och generera anteckning")
    template_choice = st.selectbox("Mall:", [
        "NP-konferens",
        "ACE",
        "Teamanteckning"
    ])

    # Sammanställningsknapp
    if st.button("✨ Sammanställ anteckning", type="primary"):
        # Kombinera textkällor
        full_text = ""
        if user_text:
            full_text += f"Användartext:\n{user_text}\n"
        if transcribed_text:
            full_text += f"Transkriberat samtal:\n{transcribed_text}\n"

        # Välj prompt baserat på mall
        if template_choice == "NP-konferens":
            full_prompt = f"""
            Du är en professionell skrivassistent. Skriv en sammanhängande, flytande text enligt följande NP-konferensmall:

            Närvarande, Aktuellt, Beslut.

            Använd löpande text – inga punktlistor. Strukturera tydligt efter rubrikerna.

            Text:
            {full_text}

            Extraherad text från bild:
            {image_caption}
            """
        elif template_choice == "ACE":
            full_prompt = f"""
            Du är en professionell skrivassistent. Läs följande text (kladdig anamnes) och extrahera relevant information enligt ACE-mallen.

För varje kriterium (A1a–i och A2a–i) ska du:
- Skriva en **kort, flytande mening** som sammanfattar vad som framkommit om just det kriteriet.
- **Varje rad** ska börja med t.ex. "A1a –", följt av texten.
- Hoppa över kriterier där ingen information finns.
- Använd inga punktlistor, inga rubriker, inga sammanfattningar.
- Skriv sakligt i journalstil och referera till personen som "patienten" och skriv endast på svenska.

**Förtydligande av kriterier**:
A1a – Slarvfel: Svårigheter att hålla uppe noggrannhet, ofta misstag i skolarbete eller andra uppgifter.  
A1b – Bibehålla fokus: Svårt att hålla uppmärksamheten, lättdistraherad.  
A1c – Lyssna: Verkar inte lyssna när man pratar direkt till hen.  
A1d – Följa instruktioner: Svårigheter att följa igenom uppgifter.  
A1e – Organisering: Svårt att organisera uppgifter och aktiviteter.  
A1f – Undviker mental ansträngning: Undviker uppgifter som kräver ihållande mental koncentration.  
A1g – Tappa bort saker: Tappar ofta bort sådant som är nödvändigt i vardagen.  
A1h – Lättdistraherad av yttre stimuli.  
A1i – Glömska: Glömmer dagliga aktiviteter, tider, överenskommelser.  

A2a – Sitter stilla: Svårighet att sitta still.  
A2b – Lämnar plats: Lämnar plats i situationer där det förväntas att man sitter kvar.  
A2c – Inre/yttre rastlöshet: Springer runt, rör sig mycket, eller visar tecken på stark inre rastlöshet.  
A2d – Leka lugnt: Svårt att leka eller umgås lugnt och tyst.  
A2e – På språng: Ofta ”på språng”, som om driven av en motor.  
A2f – Avbryter andra: Avbryter samtal eller handlingar hos andra.  
A2g – Inkräktar: T.ex. tränger sig i samtal eller lekar.  
A2h – Svårt att vänta på sin tur.  
A2i – Pratar överdrivet mycket.

Text:
{full_text}


            Extraherad text från bild:
            {image_caption}
            """
        elif template_choice == "Teamanteckning":
            full_prompt = f"""
            Du är en professionell skrivassistent. Skriv en sammanhängande anteckning enligt följande mall:

            Närvarande, Bedömning, Bakgrund, Anamnes.

            Texten ska ha flyt, vara lättläst och tydlig. Undvik punktlistor och skriv under varje rubrik som ett stycke.

            Text:
            {full_text}

            Extraherad text från bild:
            {image_caption}
            """

        with st.spinner("🧠 Genererar anteckning..."):
            response = requests.post("http://localhost:11434/api/generate", json={
                "model": "llama3",
                "prompt": full_prompt,
                "stream": False
            }).json()

        st.success("✅ Klar!")
        st.markdown("### 📑 Sammanställd anteckning:")
        st.text_area("📝 Resultat:", value=response["response"], height=300)

