import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
import speech_recognition as sr
import pyttsx3
import random
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer

# Initialize speech recognition and text-to-speech engines
nltk.download('punkt')
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Function to speak a given text
def talk(text):
    st.write("Jarvis:", text)
    engine.say(text)
    engine.runAndWait()

# Function to transcribe speech input
def transcribe_speech():
    with sr.Microphone() as source:
        st.write("Listening...")
        try:
            audio = recognizer.listen(source, timeout=8)  # Listen for 5 seconds
            st.write("Transcribing...")
            text = recognizer.recognize_google(audio, language="en-US")
            return text.lower()
        except sr.WaitTimeoutError:
            return ""
        except sr.UnknownValueError:
            return ""

# Sentiment Analysis Function
def sentiment_analysis():
    with open('tokenizer(2).pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    model = load_model('model2.h5')
    label_to_emotion = {0: 'sadness', 1: 'anger', 2: 'joy', 3: 'love', 4: 'surprise', 5: 'fear'}

    st.title("Sentiment Analysis")
    talk("Welcome to Sentiment Analysis. Please speak your sentence.")
    user_input =transcribe_speech()
    st.text_area("Your sentence will appear here after speaking", user_input)
    if not user_input:
        talk("No input detected. Please speak your sentence.")
    else:
        # add code here for sentimental analysis
        sequences = tokenizer.texts_to_sequences([user_input])
        padded_sequences = pad_sequences(sequences, maxlen=313, truncating='pre')
        predictions = model.predict(padded_sequences)
        predicted_label = np.argmax(predictions, axis=1)[0]
        predicted_emotion = label_to_emotion[predicted_label]
        # result = "This is a positive sentence."
        talk(predicted_emotion)
    talk("do you want to predict emotion of one more")
    choice = transcribe_speech()
    if "yes" in choice:
        talk("you chose sentiment analysis")
        sentiment_analysis()

    st.write("Now proceeding to the Movie Guessing Game...")
    movie_guessing_game()

def movie_guessing_game():
    st.title("Movie Guessing Game")

    movie_clues = {
        "passengers": [
            "A spacecraft traveling to a distant colony planet and transporting thousands of people has a malfunction in its sleep chambers. As a result, two passengers are awakened 90 years early.",
            "The lead actors of this movie are Jennifer Lawrence and Chris Pratt.",
            "It was directed by Morten Tyldum."
        ],
        "thor ragnarok": [
            "A superhero film based on Asgardian who was without his Mjolnir.",
            "The lead actor of this movie is Chris Hemsworth.",
            "It was directed by Taika Waititi."
        ],
        "maleficent": [
            "A dark fantasy film based on the Disney Sleeping Beauty story. A curse to the daughter of a king.",
            "The lead actress of this movie is Angelina Jolie.",
            "It was directed by Robert Stromberg."
        ],
        "night at the museum": [
            "A fantasy-comedy film series security guard at the American Museum of Natural History in New York City.",
            "The lead actor of this movie is Ben Stiller.",
            "It was directed by Shawn Levy."
        ],
        "fantastic four": [
            "Four astronauts on an experimental spacecraft who are bombarded with a comet's cosmic rays, whereby they acquire extraordinary abilities.",
            "The lead actors of this movie are Ioan Gruffudd, Jessica Alba, Chris Evans, and Michael Chiklis.",
            "It was directed by Josh Trank."
        ],
    }

    talk("Welcome to the Movie Guessing Game!")
    for movie, clues in movie_clues.items():

        movie = random.choice(list(movie_clues.keys()))
        clues = movie_clues[movie]
        # for movie, clues in movie_clues.items():
        talk("Here's the first clue for a movie title:")
        talk(clues[0])  # First clue without revealing lead actors
        player_input = transcribe_speech()
        st.write("User:", player_input)
        user_guess = player_input.lower().strip()
        correct_answer = movie.lower().strip()
        if user_guess == movie:
            talk("Correct! You guessed the movie title.")
        else:
            talk("Incorrect. Try again!")
            talk("Here's another hint:")
            talk(clues[1])
            player_input = transcribe_speech()
            st.write("User:", player_input)
            user_guess = player_input.lower().strip()
            if user_guess == movie:
                talk("Correct! You guessed the movie title.")
            else:
                talk("Incorrect. Try again!")
                talk("Here's the final hint:")
                talk(clues[2])
                player_input = transcribe_speech()
                st.write("User:", player_input)
                user_guess = player_input.lower().strip()
                if user_guess == movie:
                    talk("Correct! You guessed the movie title.")
                else:
                    talk(f"Sorry, the movie was '{movie}'. Let's move on to the next one.")

    talk("Thank you for playing the Movie Guessing Game!")

# if __name__ == "__main__":
#     st.title("Fun Voice Games")
#
#     sentiment_analysis()

def main():
    st.title("Choose Your Game")
    talk("Welcome! Do you want to play the movie guessing game or predict sentiment?")
    option = transcribe_speech()

    if "movie guessing" in option:
        talk("You chose the movie guessing game!")
        movie_guessing_game()
    elif "sentiment" in option or "prediction" in option:
        talk("You chose sentiment prediction!")
        sentiment_analysis()
    else:
        talk("Sorry, I couldn't understand your choice. Please try again.")
        main()


if __name__ == "__main__":
    main()
