import whisper  
import sounddevice as sd
from scipy.io.wavfile import write
import spacy
import re
from transformers import pipeline

nlp = spacy.load("en_core_web_sm")
sentiment_pipeline = pipeline("sentiment-analysis") 





def record_audio(filename, duration=10,fs=44100):
    print("Recording audio...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait() 
    write(filename, fs, audio) 
    print("Recording finished and saved as", filename)
    
#record_audio("interview_response.wav", duration=15)
def analyze_audio(filename):
    model = whisper.load_model("base")
    result = model.transcribe(filename)
    transcripted_text = result["text"]
    
    doc = nlp(transcripted_text)
    word_count = len([token.text for token in doc if token.is_alpha])
    filler_words = ["um", "uh", "like", "you know", "so", "actually", "basically", "literally", "seriously", "right", "i mean"]
    filler_count = sum(len(re.findall(r'\b' + re.escape(word) + r'\b', transcripted_text.lower())) for word in filler_words)
    
    keywords = ["teamwork", "leadership", "communication", "problem-solving", "adaptability", "creativity", "initiative", "collaboration", "innovation", "growth", "development", "experience", "skills", "passion", "opportunity", "role"]
    keyword_score = sum(transcripted_text.lower().count(keyword) for keyword in keywords)
    confidence_score = 10
    if word_count > 0:
        filler_ratio = filler_count / word_count
        confidence_score -= int(filler_ratio * 100)
    confidence_score += (keyword_score * 2)
    confidence_score = max(min(confidence_score, 100), 0)
    
    result1 = sentiment_pipeline(transcripted_text)
    label = result1[0]['label']
    score = result1[0]['score']
    final_score = round(score * 10) if label == "POSITIVE" else round((1 - score) * 10) 
    final_score1 = (final_score * 0.4 + confidence_score * 0.3 + keyword_score * 0.3)
    
    analysis = {
        "transcription": transcripted_text,
        "word_count": word_count,
        "filler_count": filler_count,
        "keyword_score": keyword_score,
        "confidence_score": confidence_score,
        "sentiment":label,
        "sentiment_score": score,
        "final_score": final_score1     
        
    }
    return analysis
    
    

    
'''chunk = 1024  # Number of audio frames per buffer
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1  # Single channel for microphone
    fs = 44100  # Sampling frequency

    p = pyaudio.PyAudio()

    # Start recording
    stream = p.open(format=sample_format, channels=channels,
                    rate=fs, input=True,
                    frames_per_buffer=chunk)

    print("Recording...")

    frames = []

    for i in range(0, int(fs / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Finished recording.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded data as a WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))

# Call the function to record audio
record_audio()
'''
# Transcribe the recorded audio

'''
print("Transcipt Analysis:")
print(f"Word Count: {word_count}")
print(f"Filler Words Count: {filler_count}")
print(f"Keyword Score: {keyword_score}")
print(f"Confidence Score: {confidence_score}")

result1 = sentiment_pipeline(transcripted_text)
print("Sentiment Analysis:", result1)

def interpret_sentiment(result1):
    label = result1[0]['label']
    score = result1[0]['score']
    final_score = round(score * 10) if label == "POSITIVE" else round((1 - score) * 10)
    final_score1 = (final_score * 0.4 + confidence_score * 0.3 + keyword_score * 0.3)
    print("Final Score:", final_score1)
    return label, final_score
label, final_score = interpret_sentiment(result1)
'''
'''transcribed_text = result["text"]
blob = TextBlob(transcribed_text)
sentiment = blob.sentiment
print("Sentiment Polarity(emotion):", sentiment.polarity)
print("Subjective (Confidence in tone):", sentiment.subjectivity)

if sentiment.polarity > 0.2:
    mood = "positive"
elif sentiment.polarity < -0.2:
    mood = "negative"   
else:
    mood = "neutral"

if sentiment.subjectivity > 0.5:
    confidence = "Confident/Opinionated"
else:
    confidence = "Neutral/Fact-based"
print(f"Mood: {mood} | Confidence: {confidence}")
'''
