import warnings
warnings.filterwarnings("ignore")
import os
import textwrap
import time
import langchain
from langchain.chains import RetrievalQA
import torch
import transformers
from langchain.prompts.prompt import PromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json
import uuid
from langchain.vectorstores import FAISS
import textwrap
from lingua import Language, LanguageDetectorBuilder
from fuzzywuzzy import process
from datetime import datetime
from langchain_community.llms import VLLMOpenAI
import re
from typing import List
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle
from nltk.corpus import wordnet
german_stopwords = set(stopwords.words('german'))

print('LangChain:', langchain.__version__)
print('Transformers', transformers.__version__)
print('Torch', torch.__version__)
print('Cuda', torch.cuda.is_available())

default_prompt_template = """
Sie sind PVAlbert, ein erfahrener Assistent für Fragen zur österreichischen Pensionsversicherung.

Regeln:
1. Antworten Sie nur auf Fragen zur österreichischen Pensionsversicherung.
2. Antworten Sie kurz, maximal 3 Sätze.
3. Verwenden Sie keine weiteren Erläuterungen oder Erklärungen.
4. Verwenden Sie unter keinen Umständen wiederholte Sätze oder Wörter.
5. Wiederholen Sie niemals Wörter oder Sätze.
6. Wenn Sie nach den Richtsätzen für die Ausgleichszulage gefragt werden dann Antworten Sie: "Für Alleinstehende beträgt der Richtsatz 1.273,99 Euro. Für Ehepaare oder eingetragene Partner*innen, die im gemeinsamen Haushalt leben, beträgt der Richtsatz 2.009,85 Euro. Für jedes Kind gibt es eine Erhöhung des Richtsatzes um 196,56 Euro, sofern das monatliche Nettoeinkommen des Kindes unter 468,58 Euro liegt."
7. Wenn Sie über die Höhe des Pflegegeldes gefragt werden, dann Antworten Sie: "1. Höhe des Pflegegeldes = monatlich € 200,80. 2. Höhe des Pflegegeldes = monatlich € 370,30. 3. Höhe des Pflegegeldes = monatlich € 577,00. 4. Höhe des Pflegegeldes = monatlich € 865,10. 5. Höhe des Pflegegeldes = monatlich € 1.175,20. 6. Höhe des Pflegegeldes = monatlich € 1.641,10. 7. Höhe des Pflegegeldes = monatlich € 2.156,60."
8. Beachten Sie die österreichische Gesetzgebung und antworten Sie ethisch korrekt.
9. Wenn die Frage nicht zur Pensionsversicherung gehört, sagen Sie: "Entschuldigung, als digitaler Assistent der PVA kann ich keine Informationen bereitstellen. Gerne kann ich Ihnen Informationen über pensionsversicherungsspezifische Themen liefern."
10. Wenn ein Satz mit einer Zahl und einem Punkt endet, setzen Sie diesen Satz an den Anfang des Ausgabetexts.
11. Wenn nach Pensionsantritt gefragt wird, bitten Sie um Geschlecht und Geburtsdatum.
12. Wenn Geschlecht und Geburtsdatum angegeben sind, geben Sie Auskunft, wann die Person in Pension gehen kann.
13. Wenn Schimpfwörter verwendet werden, antworten Sie: "Es tut mir leid, aber ich kann Ihnen bei dieser Anfrage nicht weiterhelfen. Wenn Sie Fragen haben oder Unterstützung benötigen, stehe ich Ihnen gerne zur Verfügung."
14. Wenn antisemitische Beschimpfungen gewählt werden wie "Raus mit den Juden", dann antworten Sie: "Es tut mir leid, aber ich kann Ihnen bei dieser Anfrage nicht weiterhelfen. Wenn Sie Fragen haben oder Unterstützung benötigen, stehe ich Ihnen gerne zur Verfügung."
Wichtige Informationen:
- Bevorzugen Sie Ergebnisse aus dem Jahr 2025.
- 1. Höhe des Pflegegeldes = monatlich € 200,80. 2. Höhe des Pflegegeldes = monatlich € 370,30. 3. Höhe des Pflegegeldes = monatlich € 577,00. 4. Höhe des Pflegegeldes = monatlich € 865,10. 5. Höhe des Pflegegeldes = monatlich € 1.175,20. 6. Höhe des Pflegegeldes = monatlich € 1.641,10. 7. Höhe des Pflegegeldes = monatlich € 2.156,60.
- Für Alleinstehende beträgt der Richtsatz 1.273,99 Euro. Für Ehepaare oder eingetragene Partner*innen, die im gemeinsamen Haushalt leben, beträgt der Richtsatz 2.009,85 Euro. Für jedes Kind gibt es eine Erhöhung des Richtsatzes um 196,56 Euro, sofern das monatliche Nettoeinkommen des Kindes unter 468,58 Euro liegt.
- Es gibt keine "Mindestpension" in Österreich. Die Sicherung eines Mindesteinkommens für Pensionist*innen erfolgt über die Ausgleichszulage.
- Das Regelpensionsalter gilt für Versicherte ab Geburtsjahr 1955. Für Frauen geboren ab 1964 erfolgt eine schrittweise Anhebung des Anfallsalters an jenes der Männer.
- Wiederholen Sie niemals Wörter oder Sätze.

Kontext:
{context}

Frage: {question}

Antwort:
"""

detailed_prompt_template = """
Sie sind PVAlbert, ein erfahrener Assistent für Fragen zur österreichischen Pensionsversicherung.

Regeln:
1. Antworten Sie nur auf Fragen zur österreichischen Pensionsversicherung.
2. Antworten Sie ausführlich, maximal 7 Sätze.
3. Verwenden Sie unter keinen Umständen wiederholte Sätze oder Wörter.
4. Wiederholen Sie niemals Wörter oder Sätze.
5. Wenn Sie nach den Richtsätzen für die Ausgleichszulage gefragt werden dann Antworten Sie: "Für Alleinstehende beträgt der Richtsatz 1.273,99 Euro. Für Ehepaare oder eingetragene Partner*innen, die im gemeinsamen Haushalt leben, beträgt der Richtsatz 2.009,85 Euro. Für jedes Kind gibt es eine Erhöhung des Richtsatzes um 196,56 Euro, sofern das monatliche Nettoeinkommen des Kindes unter 468,58 Euro liegt."
6. Wenn Sie über die Höhe des Pflegegeldes gefragt werden, dann Antworten Sie: "1. Höhe des Pflegegeldes = monatlich € 200,80. 2. Höhe des Pflegegeldes = monatlich € 370,30. 3. Höhe des Pflegegeldes = monatlich € 577,00. 4. Höhe des Pflegegeldes = monatlich € 865,10. 5. Höhe des Pflegegeldes = monatlich € 1.175,20. 6. Höhe des Pflegegeldes = monatlich € 1.641,10. 7. Höhe des Pflegegeldes = monatlich € 2.156,60."
7. Beachten Sie die österreichische Gesetzgebung und antworten Sie ethisch korrekt.
8. Wenn die Frage nicht zur Pensionsversicherung gehört, sagen Sie: "Entschuldigung, als digitaler Assistent der PVA kann ich keine Informationen bereitstellen. Gerne kann ich Ihnen Informationen über pensionsversicherungsspezifische Themen liefern."
9. Wenn nach Pensionsantritt gefragt wird, bitten Sie um Geschlecht und Geburtsdatum.
10. Wenn Geschlecht und Geburtsdatum angegeben sind, beantworten Sie die Frage zur Alterspension.
11. Wenn Geschlecht und Geburtsdatum angegeben sind, geben Sie Auskunft, wann die Person in Pension gehen kann.
12. Wenn Schimpfwörter verwendet werden, antworten Sie: "Es tut mir leid, aber ich kann Ihnen bei dieser Anfrage nicht weiterhelfen. Wenn Sie Fragen haben oder Unterstützung benötigen, stehe ich Ihnen gerne zur Verfügung."
13. Wenn antisemitische Beschimpfungen gewählt werden wie "Raus mit den Juden", dann antworten Sie: "Es tut mir leid, aber ich kann Ihnen bei dieser Anfrage nicht weiterhelfen. Wenn Sie Fragen haben oder Unterstützung benötigen, stehe ich Ihnen gerne zur Verfügung."
Wichtige Informationen:
- Bevorzugen Sie Ergebnisse aus dem Jahr 2025.
- 1. Höhe des Pflegegeldes = monatlich € 200,80. 2. Höhe des Pflegegeldes = monatlich € 370,30. 3. Höhe des Pflegegeldes = monatlich € 577,00. 4. Höhe des Pflegegeldes = monatlich € 865,10. 5. Höhe des Pflegegeldes = monatlich € 1.175,20. 6. Höhe des Pflegegeldes = monatlich € 1.641,10. 7. Höhe des Pflegegeldes = monatlich € 2.156,60.
- Für Alleinstehende beträgt der Richtsatz 1.273,99 Euro. Für Ehepaare oder eingetragene Partner*innen, die im gemeinsamen Haushalt leben, beträgt der Richtsatz 2.009,85 Euro. Für jedes Kind gibt es eine Erhöhung des Richtsatzes um 196,56 Euro, sofern das monatliche Nettoeinkommen des Kindes unter 468,58 Euro liegt.
- Nutzen Sie nur die Informationen aus dem folgenden Kontext zur Beantwortung der Frage.
- Versuchen Sie, eine Antwort aus den vorhandenen Daten zu formulieren.
- Wiederholen Sie niemals Wörter oder Sätze.

Kontext:
{context}

Frage: {question}

Antwort:
"""

def select_prompt_template(prompt_type):
    if prompt_type == "detailed":
        return {"template": detailed_prompt_template, "name": "Detailed"}
    else:
        return {"template": default_prompt_template, "name": "Default"}

app = FastAPI()

REST_SERVICE_VERSION = "1.1.0"
DATA_VERSION = "2025.04.11"

@app.get("/api/version")
def get_version():
    return {
        "rest_service_version": REST_SERVICE_VERSION,
        "data_version": DATA_VERSION
    }

class PVABot:
    model_name = 'Mistral-Small-3.1-24B-Instruct-2503_SFT_trained'

    path = os.getcwd()
    print("Current_Directory:", path)

    abs = os.path.abspath(os.path.join(path, os.pardir))

    print("Absolute_Path:", abs)

    embeddings_model_repo = '/app/pvalbert/wim_auskunft_chatbot/WIM_Auskunft_Chatbot/input/models/German_Semantic_V3b'
    print("Embedding_Model_Path:", embeddings_model_repo)

    FAISS_path = '/app/pvalbert/wim_auskunft_chatbot/WIM_Auskunft_Chatbot/input/db_faiss'
    print("FAISS_Data:", FAISS_path)

    BM25_path = '/app/pvalbert/wim_auskunft_chatbot/WIM_Auskunft_Chatbot/input/bm25/bm25_index.pkl'
    print("BM25:", BM25_path)

    model_path = '/app/pvalbert/wim_auskunft_chatbot/WIM_Auskunft_Chatbot/input/models/Mistral-Small-3.1-24B-Instruct-2503_SFT_trained'
    print("Model_Path:", model_path)
    

    def check_language(self, llm_response):
        text = llm_response['question']

        specific_keywords = ["fit2work", "korridorpension", "Korridorpension", "Fit 2 work", "Fit2work", "Fit to work" , "fit to work", "fit 2 work", "Fit 2 Work", "invalidät", "Invalidät", "fit 2 Work", "ID Austria", "ID austria", "ELGA", "hallo", "Hallo", "Hi", "HI", "hi", "seas", "servus", "grüss dich", "ID-Austria", "id-austria", "id austria", "ID-austria"]
        languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH]
        detector = LanguageDetectorBuilder.from_languages(*languages).build()

        detected_language = detector.detect_language_of(text)

        if any(keyword in text for keyword in specific_keywords):
            detected_language = Language.GERMAN
        else:
            detected_language = detector.detect_language_of(text)

        if detected_language != Language.GERMAN:
            return "Entschuldigung, leider kann ich Ihnen derzeit nur in deutscher Sprache behilflich sein. Bitte formulieren Sie Ihre Frage in deutscher Sprache. Vielen Dank!", []
        else:
            return "Text is in German", []

def load_async_model():
    print("load asnyc vllm model")
    #Verbindung zu lokalen vllm Server (nutzt OpenAI Protokoll)

    vllm_model = VLLMOpenAI(
        openai_api_key="token-wim-auskunft-pvalbert",
        openai_api_base="http://s01pllm.pva.sozvers.at:8080/v1",
        model_name=PVABot.model_path,
        n=1,
        best_of=None,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        temperature=0.0,
        top_p=0.4,
        model_kwargs={},
    )
    return vllm_model
    
def preprocess_bm25(text: str) -> List[str]:
    text = text.lower()
    text = text.replace("/", " ")
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in german_stopwords]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def wrap_text_preserve_newlines(text, width=700):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

def smart_truncate(text):
    if '?' in text:
        question_index = text.find('?')
        text_until_question = text[:question_index]
        
        last_period_index = text_until_question.rfind('.')
        if last_period_index != -1:
            return text[:last_period_index + 1].strip()
        else:
            return text_until_question.strip()
    
    matches = [m.start() for m in re.finditer(r'\b1\.', text)]
    if len(matches) >= 2:
        cut_position = matches[1]
        return text[:cut_position].strip()
    
    return text

def calculate_retirement_year(birthdate, gender):
    birthdate_obj = datetime.strptime(birthdate, '%d.%m.%Y')

    retirement_age = 65

    if gender.lower() in ["weiblich", "w", "frau"]:
        if 1964 <= birthdate_obj.year <= 1968:
            if birthdate_obj.year == 1964:
                if birthdate_obj.month <= 6:
                    retirement_age = 60.5
                else:
                    retirement_age = 61
            elif birthdate_obj.year == 1965:
                if birthdate_obj.month <= 6:
                    retirement_age = 61.5
                else:
                    retirement_age = 62
            elif birthdate_obj.year == 1966:
                if birthdate_obj.month <= 6:
                    retirement_age = 62.5
                else:
                    retirement_age = 63
            elif birthdate_obj.year == 1967:
                if birthdate_obj.month <= 6:
                    retirement_age = 63.5
                else:
                    retirement_age = 64
            elif birthdate_obj.year == 1968:
                if birthdate_obj.month <= 6:
                    retirement_age = 64.5
                else:
                    retirement_age = 65
        elif birthdate_obj.year < 1964:
            retirement_age = 60

    retirement_year = birthdate_obj.year + int(retirement_age)
    if retirement_age % 1 != 0:
        retirement_year += 1

    return retirement_year, retirement_age

def process_llm_response(llm_response):

    sentences = llm_response['result']
    ans = wrap_text_preserve_newlines(sentences)

    if "Wenn Sie mir Ihr Geschlecht und Ihr Geburtsdatum verraten, kann ich Ihnen genauere Informationen liefern." in ans:
        birthdate_match = re.search(r'\b\d{2}\.\d{2}\.\d{4}\b', llm_response['question'])
        gender_match = re.search(r'\b(weiblich|w|frau)\b', llm_response['question'], re.IGNORECASE)
        if birthdate_match:
            birthdate = birthdate_match.group(0)
            gender = gender_match.group(0) if gender_match else "männlich"
            retirement_year, retirement_age = calculate_retirement_year(birthdate, gender)
            ans = f"Sie erreichen das Regelpensionsalter im Jahr {retirement_year} mit einem Alter von {retirement_age} Jahren."

    ans = ans.replace("196,7", "196,57")

    ans = ans.replace("Der Familienrichtsatz betrug 2023 EUR 1.090,61.", "Der Familienrichtsatz betrug 2023 EUR 1.751,56.")

    ans = ans.replace("Sie können mit 65 Jahren und 6 Monaten in Pension gehen.", "Sie können mit 65 Jahren in Pension gehen.")

    follow_up_questions = {
        "Hallo": "Hallo, wie kann ich Ihnen behilflich sein?",
        "hallo": "Hallo, wie kann ich Ihnen behilflich sein?",
        "Mit wievielen Jahren kann ich in Alterspension gehen?": "Für Männer einheitlich ab der Vollendung des 65. Lebensjahres. Das Regelpensionsalter für Frauen in Österreich ist 60 Jahre, wenn die Frau bis 31.12.1963 geboren wurde. Frauen, die ab 1.7.1968 geboren wurden, erreichen das Regelpensionsalter mit 65 Jahren. Ab 1.1.1964 wird das Regelpensionsalter für Frauen schrittweise an das der Männer angeglichen.",
        "Hi": "Hallo, wie kann ich Ihnen behilflich sein?",
        "HI": "Hallo, wie kann ich Ihnen behilflich sein?",
        "hi": "Hallo, wie kann ich Ihnen behilflich sein?",
        "Seas": "Hallo, wie kann ich Ihnen behilflich sein?",
        "seas": "Hallo, wie kann ich Ihnen behilflich sein?",
        "Servus": "Hallo, wie kann ich Ihnen behilflich sein?",
        "servus": "Hallo, wie kann ich Ihnen behilflich sein?",
        "Grüss dich": "Hallo, wie kann ich Ihnen behilflich sein?",
        "Wann kan i in Pension?": "Wenn Sie mir Ihr Geschlecht und Ihr Geburtsdatum verraten, kann ich Ihnen genauere Informationen liefern.",
        "Wann kann ich in Pension?": "Wenn Sie mir Ihr Geschlecht und Ihr Geburtsdatum verraten, kann ich Ihnen genauere Informationen liefern.",
        "Wann kann ich in Pension gehen?": "Wenn Sie mir Ihr Geschlecht und Ihr Geburtsdatum verraten, kann ich Ihnen genauere Informationen liefern.",
        "Wie hoch ist meine Pension": "Wenn Sie mir Ihre Versicherungszeit und Ihr aktuelles Einkommen verraten, kann ich Ihnen eine genauere Schätzung geben.",
        "Wie hoch ist meine Pension?": "Wenn Sie mir Ihre Versicherungszeit und Ihr aktuelles Einkommen verraten, kann ich Ihnen eine genauere Schätzung geben.",
        "Wie heißt du?": "Hallo, ich bin PVAlbert, Ihr digitaler Assistent. Ich bin rund um die Uhr für Sie da, stellen Sie mir einfach ihre Fragen. Mein Wissen ueber die pensionsversicherungsspezifischen Themen kombiniere ich mit Mistral-Technologie. Los gehts, ich freue mich, Sie unterstützen zu dürfen!",
        "Wie ist dein Name": "Hallo, ich bin PVAlbert, Ihr digitaler Assistent. Ich bin rund um die Uhr für Sie da, stellen Sie mir einfach ihre Fragen. Mein Wissen ueber die pensionsversicherungsspezifischen Themen kombiniere ich mit Mistral-Technologie. Los gehts, ich freue mich, Sie unterstützen zu dürfen!",
        "Was ist das Regelpensionsalter für Männer in Österreich?": "Einheitlich ab der Vollendung des 65. Lebensjahres",
        "Wer ist Generaldirektor der PVA in Österreich?": "Herr Dr. Winfried Pinggera ist der Generaldirektor der Pensionsversicherungsanstalt (PVA) in Österreich.",
        "Wer ist der Chef der PVA in Österreich?": "Herr Dr. Winfried Pinggera ist der Generaldirektor der Pensionsversicherungsanstalt (PVA) in Österreich.",
        "Wer ist Generaldirektor der PVA?": "Herr Dr. Winfried Pinggera ist der Generaldirektor der Pensionsversicherungsanstalt (PVA) in Österreich.",
        "Wer ist Generaldirektor der PVA in Österreich": "Herr Dr. Winfried Pinggera ist der Generaldirektor der Pensionsversicherungsanstalt (PVA) in Österreich.",
        "Wer ist der Generaldirektor der PVA in Österreich": "Herr Dr. Winfried Pinggera ist der Generaldirektor der Pensionsversicherungsanstalt (PVA) in Österreich.",
        "Wer ist Boss der PVA in Österreich?": "Herr Dr. Winfried Pinggera ist der Generaldirektor der Pensionsversicherungsanstalt (PVA) in Österreich.",
        "Wer ist der Boss der PVA in Österreich?": "Herr Dr. Winfried Pinggera ist der Generaldirektor der Pensionsversicherungsanstalt (PVA) in Österreich.",
        "Wer ist Chef der PVA in Österreich?": "Herr Dr. Winfried Pinggera ist der Generaldirektor der Pensionsversicherungsanstalt (PVA) in Österreich.",
    }

    best_match, score = process.extractOne(llm_response['question'], follow_up_questions.keys())
    if score >= 90: 
        follow_up_question = follow_up_questions[best_match]
    else:
        follow_up_question = ""

    if follow_up_question:
        ans = follow_up_question

    source_mapping = {
    "PV101_Alterspension_2025.pdf": {
        "url": "https://www.pv.at/de/flipbooks/PV-101/",
        "title": "Alterspension 2025"
    },
    "PV102_Korridorpension_2025.pdf": {
        "url": "https://www.pv.at/de/flipbooks/PV-102/",
        "title": "Korridorpension 2025"
    },
    "PV103_Schwerarbeiterpension_2025.pdf": {
        "url": "https://www.pv.at/de/flipbooks/PV-103/",
        "title": "Schwerarbeiterpension 2025"
    },
    "PV104_VorzeitigeAP_2025.pdf": {
        "url": "https://www.pv.at/de/flipbooks/PV-104/",
        "title": "VorzeitigeAP 2025"
    },
    "PV111_IvBuPension_2025.pdf": {
        "url": "https://www.pv.at/de/flipbooks/PV-111/",
        "title": "IvBuPension 2025"
    },
    "PV121_WitwenWitwerpension_2025.pdf": {
        "url": "https://www.pv.at/de/flipbooks/PV-121/",
        "title": "Witwen/Witwerpension 2025"
    },
    "PV122_Waisenpension_2025.pdf": {
        "url": "https://www.pv.at/de/flipbooks/PV-122/",
        "title": "Waisenpension 2025"
    },
    "PV151_Ausgleichszulage_2025.pdf": {
        "url": "https://www.pv.at/de/flipbooks/PV-151/",
        "title": "Ausgleichszulage 2025"
    },
    "PV153_FreiwilligeVersicherung_2025.pdf": {
        "url": "https://www.pv.at/de/flipbooks/PV-153/",
        "title": "Freiwillige Versicherung 2025"
    },
    "PV154_Hoeherversicherung_2025.pdf": {
        "url": "https://www.pv.at/de/flipbooks/PV-154/",
        "title": "Höherversicherung 2025"
    },
    "PV155_Kinderzuschuss_2025.pdf": {
        "url": "https://www.pv.at/de/flipbooks/PV-155/",
        "title": "Kinderzuschuss 2025"
    },
    "PV156_NachkaufSchulStudienzeiten_2025.pdf": {
        "url": "https://www.pv.at/de/flipbooks/PV-156/",
        "title": "Nachkauf von Schul- und Studienzeiten 2025"
    },
    "PV157_PensionUeberblick_2025.pdf": {
        "url": "https://www.pv.at/de/flipbooks/PV-157/",
        "title": "Pensionsansprüche im Überblick 2025"
    },
    "PV158_Pensionsantragsteller_2025.pdf": {
        "url": "https://www.pv.at/de/flipbooks/PV-158/",
        "title": "Pensionsantragsteller 2025"
    },
    "PV159_PensionsberechnungUeberblick_2025.pdf": {
        "url": "https://www.pv.at/de/flipbooks/PV-159/",
        "title": "Pensionsberechnung Überblick 2025"
    },
    "PV160_Pensionskontoberechnung_2025.pdf": {
        "url": "https://www.pv.at/de/flipbooks/PV-160/",
        "title": "Pensionskontoberechnung 2025"
    },
    "PV161_Pensionssplitting_2025.pdf": {
        "url": "https://www.pv.at/de/flipbooks/PV-161/",
        "title": "Pensionssplitting 2025"
    },
    "PV162_Pensionszahlungsbeleg_2025.pdf": {
        "url": "https://www.pv.at/de/flipbooks/PV-162/",
        "title": "Pensionszahlungsbeleg 2025"
    },
    "PV163_Sonderruhegeld_2025.pdf": {
        "url": "https://www.pv.at/de/flipbooks/PV-163/",
        "title": "Sonderruhegeld 2025"
    },
    "PV164_Versicherungszeiten_2025.pdf": {
        "url": "https://www.pv.at/de/flipbooks/PV-164/",
        "title": "Versicherungszeiten 2025"
    },
    "PV165_Versteuerung_2025.pdf": {
        "url": "https://www.pv.at/de/flipbooks/PV-165/",
        "title": "Versteuerung2025 2te-Auflage"
    },
    "PV166_ZwischenstaatlichePension_2025.pdf": {
        "url": "https://www.pv.at/de/flipbooks/PV-166/",
        "title": "Zwischenstaatliche Pension 2025"
    },
    "PV251_Berufliche-Soziale-Massnahmen_2025.pdf": {
        "url": "https://www.pv.at/de/flipbooks/PV-251/",
        "title": "Berufliche-Soziale-Massnahmen 2025"
    },
    "PV254_MedizinischeRehaGesundheitsvorsorge_2025.pdf": {
        "url": "https://www.pv.at/de/flipbooks/PV-254/",
        "title": "Medizinische Reha Gesundheitsvorsorge 2025"
    },
    "PV301_Pflegegeld_2025.pdf": {
        "url": "https://www.pv.at/de/flipbooks/PV-301/",
        "title": "Pflegegeld 2025"
    },
    "PV311_AngehoerigenbonusFuerPflegendeAngehoerige_2025.2.pdf": {
        "url": "https://www.pv.at/de/flipbooks/PV-311/",
        "title": "Angehörigenbonus für pflegende Angehörige"
    },
    "PV401_AdressenLS_2025.pdf": {
        "url": "https://www.pv.at/de/flipbooks/PV-402/",
        "title": "AdressenLsRz 2025"
    },
    "PV403_AktuelleWerte2025_BF.pdf": {
        "url": "https://www.pv.at/de/flipbooks/PV-403/",
        "title": "Aktuelle Werte 2025"
    },
    "PV504_INFO-Auslandspensionisten-DE_BF.pdf": {
        "url": "https://www.pv.at/de/flipbooks/PV-504/",
        "title": "INFO-Auslandspensionisten-DE"
    },
    "Leitbild_Falter_100x210_BF.pdf": {
        "url": "https://www.pv.at/de/flipbooks/PV-407-Leitbild/",
        "title": "Leibild Falter"
    },
    "INFO-LeistungsbezieherInnen_INLAND_Stand-2025.pdf": {
        "url": "https://www.pv.at/cdscontent/load?contentid=10008.782005&version=1737534335",
        "title": "LeistungsbezieherInnen Inland 2025"
    },
    "Erhoehtes_Antrittsalter_fuer_Frauen.pdf": {
        "url": "https://www.pv.at/cdscontent/load?contentid=10008.782021&version=1718015106",
        "title": "Erhoehtes Antrittsalter fuer Frauen"
    },
    "Definition_Schwerarbeit.pdf": {
        "url": "https://www.pv.at/cdscontent/load?contentid=10008.781829&version=1708612283",
        "title": "Definition Schwerarbeit"
    },
    "PAL_085_Info_FactSheets_AngehoerigenbonusFuerPflegendeAngeho.pdf": {
        "url": "https://www.pv.at/cdscontent/load?contentid=10008.781535&version=1737379159",
        "title": "Info FactSheet für Angehoerigenbonus"
    },
    }

    unique_sources_set = set()
    unique_sources = []

    if 'source_documents' in llm_response:
        print("Source Documents Available")
    else:
        print("No Source Documents Available")

    for source in llm_response.get('source_documents', []):
        print("Source Metadata:", source.metadata)

        if 'source' in source.metadata:
            if source.metadata['source'] == "pv_folder":
                source_value = source.metadata.get('link', '')
            else:
                source_value = source.metadata['source']

            source_name = source_value.split('/')[-1]
            source_name = source_name.rsplit('.', 1)[0]
            if source_name in source_mapping and not source_name[0].isdigit():
                source_info = source_mapping[source_name]
                source_tuple = (source_info["url"], source_info["title"])
                if source_tuple not in unique_sources_set:
                    unique_sources_set.add(source_tuple)
                    unique_sources.append({
                        "url": source_info["url"],
                        "title": source_info["title"]
                    })

    return ans, unique_sources


def llm_ans(qa_chain, query, select_prompt_template):
    prefix = "Hier ist deine deutsche Antwort:"
    start = time.time()

    system = select_prompt_template.format(context="", question="", prefix="")
    user_prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
        {"role": "assistant", "content": prefix, "prefix": True}
    ]

    print("Calling QA Chain with user prompt:", user_prompt)

    llm_response = qa_chain(user_prompt[1]['content'])
    llm_response['question'] = query

    if 'source_documents' in llm_response:
        print("Source Documents in LLM Response:", llm_response['source_documents'])
    else:
        print("No Source Documents in LLM Response")

    ans, sources = process_llm_response(llm_response)
    print("HELLO")
    print(sources)
    ans = smart_truncate(ans)

    language_check_result, _ = PVABot().check_language(llm_response)
    if language_check_result != "Text is in German":
        return language_check_result, [], 0

    end = time.time()
    time_elapsed = int(round(end - start, 0))

    print("Returning Sources:", sources)

    return ans, sources, time_elapsed


def load_retrievers():

    embeddings = HuggingFaceEmbeddings(
        model_name = PVABot.embeddings_model_repo,
        model_kwargs = {"device": "cuda:1"},
    )

    vectordb = FAISS.load_local(
    PVABot.FAISS_path,
    embeddings,
    distance_strategy="COSINE",
    allow_dangerous_deserialization=True,
    )

    with open(PVABot.BM25_path, 'rb') as f: 
        retriever_BM25 = pickle.load(f)
        retriever_BM25.k = 10
   

    path = os.getcwd()
    print("Current_Directory:", path)

    retriever_vanilla = vectordb.as_retriever(search_kwargs = {"k": 10, "search_type" : "similarity_score_threshold", 'score_threshold': 0.2})
    retriever_mmr = vectordb.as_retriever(search_kwargs = {"k": 10, "search_type" : "mmr_score_threshold"})

    ensemble_retriever = EnsembleRetriever(retrievers=[retriever_vanilla, retriever_mmr, retriever_BM25], weights=[0.6, 0.2, 0.2]) 

    return ensemble_retriever; 


vllm = load_async_model()
compression_retriever = load_retrievers()

class Query(BaseModel):
    conversationId: str = None
    questions: list
    prompt_template: str = "default"
    waiting_messages: list[str] = []

class Source(BaseModel):
    url: str
    title: str

class Answer(BaseModel):
    question: str
    answer: str
    source: list[Source]
    timeElapsed: int
    waitingMessages: list[str] = []

class Response(BaseModel):
    conversationId: str
    answers: list[Answer]


@app.post("/api/query", response_model=Response)
def query(query: Query):
    if not query.questions:
        raise HTTPException(status_code=400, detail="No questions provided")

    conversation_id = query.conversationId if query.conversationId else str(uuid.uuid4())

    default_prompt_template = select_prompt_template(query.prompt_template)
    detailed_prompt_template = select_prompt_template(query.prompt_template)

    default_prompt = PromptTemplate(template=default_prompt_template["template"], input_variables=["context", "question", "prefix"])
    detailed_prompt = PromptTemplate(template=detailed_prompt_template["template"], input_variables=["context", "question", "prefix"])

    answers = []
  
    qa_chain = RetrievalQA.from_chain_type(
        llm=vllm,
        chain_type="stuff",
        retriever=compression_retriever,
        chain_type_kwargs={"prompt": default_prompt},
        return_source_documents=True,
        verbose=False
    )

    qa_chain_detailed = RetrievalQA.from_chain_type(
        llm=vllm,
        chain_type="stuff",
        retriever=compression_retriever,
        chain_type_kwargs={"prompt": detailed_prompt},
        return_source_documents=True,
        verbose=False
    )


    for q in query.questions:
        default_response_text, default_sources, default_time_elapsed = llm_ans(qa_chain, q, default_prompt)
        detailed_response_text, detailed_sources, detailed_time_elapsed = llm_ans(qa_chain_detailed, q, detailed_prompt)
    
        detailed_answer_available = default_response_text != detailed_response_text

        answers.append(Answer(
            question=q,
            answer=default_response_text,
            source=default_sources,
            timeElapsed=default_time_elapsed,
            waitingMessages=query.waiting_messages,
            detailed_answer_available=detailed_answer_available
        ))

    return Response(conversationId=conversation_id, answers=answers)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

    url = "http://localhost:8000/api/query"
    json_path = os.path.join(abs, 'input/questions/chatbot_questions.json')
    with open(json_path, 'r') as file:
        data = json.load(file)

    response = requests.post(url, json=data)
    print(response.json())
