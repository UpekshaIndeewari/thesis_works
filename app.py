from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, words
import re
import enchant
import yake
import fitz 
import json
import string 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

extracted_text = None
extracted_phrases = []
http_sections = "" 
above_threshold = set()

# Extraction of text from PDF
def extract_text(pdf):
    text = ""
    reader = PdfReader(pdf)
    for page in reader.pages:
        text += page.extract_text()
    return text

# Remove headers and footers from PDF
def remove_Singlelines(text):
    pattern = r'AGILE: GIScience Series.*?https://doi\.org/[^\s]+'
    matches = re.findall(pattern, text)
    for match in matches:
        text = text.replace(match, '')
    return text.strip()

def remove_Agilelicense(text):
    pattern = r'AGILE: GIScience Series.*?This work is distributed under the Creative Commons Attribution 4.0 License'
    match = re.search(pattern, text, flags=re.DOTALL)
    removed_text = ""
    global http_sections
    if match:
        removed_text = match.group(0)
        text = text.replace(removed_text, '')
        http_pattern = re.compile(r'https?://doi\.org/[^\s]+')
        http_matches = re.findall(http_pattern, removed_text)
        http_sections += " ".join(http_matches)
    return text.strip()
     
# Convert text in to lower case
def convert_to_lower(words):
    return [word.lower() for word in words]

# Extraction of the main body from the text
def split_portion(lower_case):
    index1 = lower_case.index('abstract')
    index2 = lower_case.index('references')
    filter_portion = lower_case[index1+1:index2+1]
    return filter_portion

# Remove cases from the text
def remove_case(text_remain):
    cleaned_texts = [text.casefold() for text in text_remain]
    return cleaned_texts

def remove_stopword(cleaned_texts):
    stop_words = set(stopwords.words('english'))
    nostopword_list = [word for word in cleaned_texts if word not in stop_words]
    return nostopword_list

# Remove unicodes from the text
def remove_unicode(nostop_word):
    no_unicode = [re.sub(r'[^\x00-\x7F]+', '', word) for word in nostop_word]
    return no_unicode

# Remove numbers and digits from the text
def remove_numbers(nounicode_list):
    no_number_list = [token for token in nounicode_list if not (token.isdigit() or (token.count('.') == 1 and token.replace('.', '').isdigit()))]
    return no_number_list

# Remove special symbols from the text
def remove_special(nospecial_list):
    no_sp_words = [text for text in nospecial_list if not re.match(r'^(?=.*\d)(?=.*[A-Za-z])|(?=.*[A-Za-z])(?=.*[\W_])|(?=.*\d)(?=.*[\W_])', text)]
    return no_sp_words

# Remove letter repetition
def remove_repeat(nosp_words):
    pattern = re.compile(r'\b\w*(\w)\1{3,}\w*\b')
    filtered_word_list = [word for word in nosp_words if not pattern.search(word)]
    return filtered_word_list

# Remove single words
def remove_single_noenglish(filtered_word):
    english_words = set(words.words())
    filtered_words = [word for word in filtered_word if len(word) > 4 or word.lower() in english_words]
    return filtered_words

def remsingle_words(word_list):
    new_list = [word for word in word_list if len(word) > 1]
    return new_list

corrected_words = []

# Remove words with spelling errors
def spelling_check(word_list):
    spell_checker = enchant.Dict("en")
    for word in word_list:
        if spell_checker.check(word):
            corrected_words.append(word)
    return corrected_words

global text_final 
text_final = ''

def convert_string(lst):
    text_final = ' '.join(lst)  
    return text_final

# YAKE keyword extraction function
def yake_keyphrase_extraction(text):
    language = "en"
    max_ngram_size = 3
    deduplication_threshold = 0.4
    deduplication_algo = 'seqm'
    windowSize = 1
    numOfKeywords = 50  # Number of keywords to extract
    kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
    ex_keyphrase = kw_extractor.extract_keywords(text)
    return [phrase[0] for phrase in ex_keyphrase]

#automatic extraction of EO4GEOBOK Concepts
data = {} 

def extract_eo4geo_bok_concepts():
    with open('Data.json', 'r', encoding='utf-8') as json_file:
        return json.load(json_file)
    
def EO4GEOlist_extractor(data):
    names = []
    for key, value in data.items():
        if isinstance(value, dict):
            if 'name' in value:
                names.append(value['name'])
                names.extend(EO4GEOlist_extractor(value))
    return names

# convert concepts in to lowercase
def lower_EO4GEO(list):
    lst = []
    for x in list:
        x = x.lower()
        lst.append(x)
    return lst

global processed_list
processed_list = []

def remove_stopwords_and_process_word(text_list):
    stop_words = set(stopwords.words('english'))
    global processed_list
    
    def process_word(word):
        word = word.replace('-', ' ')
        word = word.replace(',', ' ')
        word = word.replace('/', ' ')
        word = word.replace(':', ' ')
        word = word.replace('()', ' ')
        word = word.replace("'", '')

        word_no_punctuation = ""
        for char in word:
            if char not in string.punctuation:
                word_no_punctuation += char
        return word_no_punctuation

    for element in text_list:
        words = element.split()
        filtered_words = [process_word(word) for word in words if word.lower() not in stop_words]
        processed_list.append(' '.join(filtered_words))

    return processed_list

global cleaned_list
cleaned_list = []

def clean_strings(lst):
    global cleaned_list
    for element in lst:
        cleaned_element = ' '.join(element.split())
        cleaned_list.append(cleaned_element)
    return cleaned_list


def Cosine_Similarity(list1, list2):
    print('list1',list1)
    print('list2',list2)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(list1)
    Y = vectorizer.transform(list2)
    similarity_matrix = cosine_similarity(X, Y)

    # Assuming concept_names and YAKE_keyphrases are defined somewhere
    threshold = 0.7
    

    for i, phrase1 in enumerate(list1):
        for j, phrase2 in enumerate(list2):
            score = similarity_matrix[i, j]
            if score > threshold:
                print(f"Similarity between '{phrase1}' and '{phrase2}': {score}")
                    
    return phrase1


@app.route('/', methods=['GET', 'POST'])

def index():
    global extracted_text
    global extracted_phrases
    global http_sections
    global words_corrected

    extracted_text1 = None
    http_sections = None
    lowercased_words = None
    abstract_and_references = None
    removed_cases = None
    removed_stopwords = None
    removed_unicode = None
    removed_numbers = None
    removed_special = None
    removed_repeat = None
    removed_single_noenglish = None
    removed_single = None
    words_corrected = None
    
    
    
    global eo4geo_bok_concepts
    global lower_list
    global re_stopword
    global final_list
    global similarity_results1
    global similarity_results
    
    data = None
    eo4geo_bok_concepts = None
    lower_list = None
    re_stopword = None
    final_list = None
    similarity_results1 = None
    similarity_results = None
    
    if request.method == 'POST':
        if 'extract' in request.form:
            pdf_file = request.files['file']
            if pdf_file.filename.endswith('.pdf'):
                extracted_text = extract_text(pdf_file)
        elif 'clean' in request.form and extracted_text:
            extracted_text1 = remove_Singlelines(extracted_text)
            http_sections = remove_Agilelicense(extracted_text1)
            lowercased_words = convert_to_lower(word_tokenize(http_sections))
            abstract_and_references = split_portion(lowercased_words)
            removed_cases = remove_case(abstract_and_references)
            removed_stopwords = remove_stopword(removed_cases)
            removed_unicode = remove_unicode(removed_stopwords)
            removed_numbers = remove_numbers(removed_unicode)
            removed_special = remove_special(removed_numbers)
            removed_repeat = remove_repeat(removed_special)
            removed_single_noenglish = remove_single_noenglish(removed_repeat)
            removed_single = remsingle_words(removed_single_noenglish)
            words_corrected = spelling_check(removed_single)
        
    if 'extract_bok_concepts' in request.form:
        data = extract_eo4geo_bok_concepts()
        eo4geo_bok_concepts = EO4GEOlist_extractor(data)
        print("EO4GEO Concepts:", eo4geo_bok_concepts)
        lower_list = lower_EO4GEO(eo4geo_bok_concepts)
        print("Lower List:", lower_list)
        re_stopword = remove_stopwords_and_process_word(lower_list)
        print("Processed List:", re_stopword)
        final_list = clean_strings(re_stopword)
        print("Final List:", final_list)
       

    if 'extract_keyphrases' in request.form:
        text_final = convert_string(corrected_words)
            # Using 'text_final' here which contains the cleaned text as a string
        extracted_phrases = yake_keyphrase_extraction(text_final)
        
        
    if 'calculate_similarity' in request.form and request.form['similarity_measure'] == 'cosine_similarity':
        print('calculate_similarity' in request.form)
        print(request.form['similarity_measure'] == 'cosine_similarity')
        print(request.form)
        print(final_list)
        print(extracted_phrases)
        if final_list and extracted_phrases:
            #YAKE_keyphrases = extracted_phrases
            #print(YAKE_keyphrases)
            similarity_results = Cosine_Similarity(final_list, extracted_phrases)
            print(similarity_results)
        else:
            print('final list is empty')

    return render_template('index.html', extracted_text=extracted_text, abstract_and_references=words_corrected, extracted_phrases=extracted_phrases, eo4geo_bok_concepts=final_list, similarity_results1=similarity_results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

