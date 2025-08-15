import gradio as gr
import requests

def translate_text(text, source_lang, target_lang):
    if not text.strip():
        return ""
    
    url = "https://api.mymemory.translated.net/get"
    params = {
        "q": text,
        "langpair": f"{source_lang}|{target_lang}"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data["responseStatus"] == 200:
            return data["responseData"]["translatedText"]
        else:
            return f"Translation error: {data['responseDetails']}"
    except Exception as e:
        return f"Error: {str(e)}"

# Language options
languages = {
    "Arabic": "ar",
    "Basque": "eu",
    "Belarusian": "be",
    "Bosnian": "bs",
    "Bulgarian": "bg",
    "Catalan": "ca",
    "Chinese, simplified": "zh_CN",
    "Chinese, traditional": "zh_TW",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "English": "en",
    "Estonian": "et",
    "Finnish": "fi",
    "French": "fr",
    "French, Canadian": "fr_ca",
    "Galician": "gl",
    "German": "de",
    "Greek": "el",
    "Hebrew": "he",
    "Hungarian": "hu",
    "Indonesian": "id",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Macedonian": "mk",
    "Malay": "ms",
    "Norwegian Bokmal": "nb",
    "Norwegian Nynorsk": "nn",
    "Persian": "fa",
    "Polish": "pl",
    "Portuguese": "pt",
    "Portuguese, Brazilian": "pt_BR",
    "Romanian": "ro",
    "Russian": "ru",
    "Serbian, Cyrillic": "sr_Cyrl",
    "Serbian, Latin": "sr_Latn",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Spanish": "es",
    "Swedish": "sv",
    "Thai": "th",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Valencian": "vc",
    "Vietnamese": "vn"
}

# Create Gradio interface
iface = gr.Interface(
    fn=translate_text,
    inputs=[
        gr.Textbox(label="Enter text to translate", placeholder="Type your text here..."),
        gr.Dropdown(choices=list(languages.keys()), value="English", label="From"),
        gr.Dropdown(choices=list(languages.keys()), value="Spanish", label="To")
    ],
    outputs=gr.Textbox(label="Translation"),
    title="🌍 Simple Translator 🌍",
    description="Translate text using MyMemory API",
    examples=[
        ["Hello, how are you?", "English", "Spanish"],
        ["Bonjour, comment allez-vous?", "French", "English"],
        ["Hola, ¿cómo estás?", "Spanish", "German"]
    ],
    allow_flagging="never",
    theme=gr.themes.Citrus()
)

if __name__ == "__main__":
    iface.launch()
