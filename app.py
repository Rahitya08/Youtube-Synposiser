import re
import os
from flask import Flask, render_template, request
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import yt_dlp
import whisper
from googletrans import Translator

app = Flask(__name__)


supported_languages = {
    'en': 'English',
    'fr': 'French',
    'es': 'Spanish',
    'hi': 'Hindi',
    'te': 'Telugu',
    'mr': 'Marathi'
}

class TextSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.summarizer = pipeline("summarization", model=self.model, tokenizer=self.tokenizer)

    def chunk_text(self, text, max_length=4096):
        
        sentences = text.split('. ')
        current_chunk = []
        total_length = 0
        chunks = []

        for sentence in sentences:
            sentence_length = len(self.tokenizer.encode(sentence))
            if total_length + sentence_length <= max_length:
                current_chunk.append(sentence)
                total_length += sentence_length
            else:
                chunks.append(". ".join(current_chunk) + '.')
                current_chunk = [sentence]
                total_length = sentence_length

        if current_chunk:
            chunks.append(". ".join(current_chunk) + '.')

        return chunks

    def summarize(self, text, max_length=5000, min_length=1000):
        
        chunks = self.chunk_text(text)
        if not chunks:
            raise ValueError("No content to summarize.")

        summaries = []
        for chunk in chunks:
            inputs = self.tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True, padding=True)
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                length_penalty=1.0,
                num_beams=4,
                early_stopping=True
            )
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)

        return " ".join(summaries)

    def multi_round_summarize(self, text, rounds=2):
        
        for _ in range(rounds):
            text = self.summarize(text, max_length=5000, min_length=1000)
        return text

class TextTranslator:
    def __init__(self):
        
        self.translator = Translator()

    def translate(self, text, dest_language):
        
        translation = self.translator.translate(text, dest=dest_language)
        return translation.text

def extract_video_id(url):
    
    video_id_pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11})'
    match = re.search(video_id_pattern, url)
    return match.group(1) if match else None

def download_audio(video_id):
    
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'{video_id}.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            audio_file = ydl.prepare_filename(info_dict).replace('.webm', '.mp3').replace('.m4a', '.mp3')
            return audio_file
    except Exception as e:
        print(f"Error downloading audio: {e}")
        raise

def transcribe_audio_whisper(audio_file):
    
    try:
        model = whisper.load_model("tiny")
        result = model.transcribe(audio_file)
        return result["text"]
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        raise

@app.route("/", methods=["GET", "POST"])
def index():
    
    if request.method == "POST":
        try:
            url = request.form.get("video_url")
            language_choice = request.form.get("language")

            video_id = extract_video_id(url)
            if not video_id:
                return render_template("index.html", error="Invalid YouTube URL.", languages=supported_languages)

            summarizer = TextSummarizer()

            try:
                
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                description = ''.join([x['text'] + '\n' for x in transcript])
            except (TranscriptsDisabled, NoTranscriptFound):
                
                audio_file = download_audio(video_id)
                description = transcribe_audio_whisper(audio_file)
                os.remove(audio_file)

            
            summary = summarizer.summarize(description)

            
            if language_choice != 'en':
                translator = TextTranslator()
                summary = translator.translate(summary, language_choice)

            return render_template("index.html", summary=summary, languages=supported_languages)

        except Exception as e:
            return render_template("index.html", error=str(e), languages=supported_languages)

    return render_template("index.html", languages=supported_languages)

if __name__ == "__main__":
    app.run(debug=True)
