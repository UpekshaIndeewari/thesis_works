import json
import re
import string
import enchant
import nltk
import yake
from PyPDF2 import PdfReader
from flask import Flask, render_template, request
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
nltk.download('words')
nltk.download('stopwords')
nltk.download('punkt')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keyphrase_vectorizers import KeyphraseCountVectorizer
from keybert import KeyBERT
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import pandas as pd
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

app = Flask(__name__)
app.static_folder = 'static'

extracted_phrases = []
extracted_phrases_pattern = []
final_list = []
http_sections = ""
above_threshold = set()
title_doc = ""
message = ""
message_eo4 = ""


extracted_text = None
eo4geo_bok_concepts = None
lower_list = None
re_stopword = None
final_list = None
similarity_results1 = None
similarity_results = None
words_corrected = None
finished_message = None
title_doc = None
message = None
message_eo4 = None

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

# Extract the title ofthe document
global title
def title_Ex(word_list):
    index3 = word_list.index('abstract')
    title = word_list[0:index3]
    separator = ' ' 
    result = separator.join(title) 
    return result

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
#---------------------------------------------------------------------
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
#---------------------------------------------------------------------
global PatternRank_kp
PatternRank_kp =[]

# PatternRank keyword extraction function
def patternRank_extractor(text_document):
    text_final1 = ' '.join(text_document)
      # Tokenize the text
    words = set(nltk.corpus.words.words())
    tokens = nltk.wordpunct_tokenize(text_final1)
    filtered_tokens = [w for w in tokens if w.lower() in words or not w.isalpha()]
    x = " ".join(filtered_tokens)
      
    # Extract keyphrases
    kw_model = KeyBERT()
    key_phrases = kw_model.extract_keywords(docs=x, vectorizer=KeyphraseCountVectorizer(), top_n=50)
    print("Extracted keyphrases from PatternRank with socres: \n",key_phrases)
      
    global PatternRank_kp
    for i,j in key_phrases:
        PatternRank_kp.append(i)
        print("Extracted keyphrases from PatternRank without socres: \n",PatternRank_kp)
    return PatternRank_kp
#-------------------------------------------------------------------------------
global Keybert_kp
Keybert_kp =[]
# KeyBert keyword extraction function
def KeyBert_extrator(list):
# Convert list in to string
    listToStr = ' '.join(map(str, list))
# Extraction of key phrases using KeyBert algorithom
    kw_model = KeyBERT()
    listkp = kw_model.extract_keywords(listToStr, keyphrase_ngram_range=(1, 4), stop_words=None,top_n=100)
    global  Keybert_kp
    for item in listkp:
        Keybert_kp.append(item[0])
    print("Extracted key phrase using KetBert algorithm \n", Keybert_kp)
    return  Keybert_kp
#-------------------------------------------------------------------------------
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

global concept_names
concept_names = []
concept_names = EO4GEOlist_extractor(data)

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

#--------------------------------------------------------------------------------
#Cosine similarity measures

def Cosine_Similarity(list1, list2,threshold):
    print('list1',list1)
    print('list2',list2)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(list1)
    Y = vectorizer.transform(list2)
    similarity_matrix = cosine_similarity(X, Y)
    
    similarity_score_out = set()
    #threshold = 0.9
    for i, phrase1 in enumerate(list1):
        for j, phrase2 in enumerate(list2):
            score = similarity_matrix[i, j]
            if score > threshold:
                similarity_score_out.add(phrase1)
    return similarity_score_out
#---------------------------------------------------------------------------------
#Jaro-Winkler Similarity measures
def jarowinkler_similarity(list1, list2, threshold=0.9):
    #mapdict_Cleanedlist = dict(zip(cleaned_list, concept_names))
    # Jaro Similarity of two strings
    def jaro_distance(s1, s2):
        if s1 == s2:
            return 1.0
        len1 = len(s1)
        len2 = len(s2)

        if len1 == 0 or len2 == 0:
            return 0.0

        max_dist = (max(len1, len2) // 2) - 1

        match = 0
        hash_s1 = [0] * len1
        hash_s2 = [0] * len2

        for i in range(len1):
            for j in range(max(0, i - max_dist), min(len2, i + max_dist + 1)):
               if s1[i] == s2[j] and hash_s2[j] == 0:
                    hash_s1[i] = 1
                    hash_s2[j] = 1
                    match += 1
                    break

        if match == 0:
            return 0.0

        t = 0
        point = 0

        for i in range(len1):
            if hash_s1[i]:
                while hash_s2[point] == 0:
                    point += 1
                if s1[i] != s2[point]:
                    point += 1
                    t += 1
                else:
                    point += 1

        t /= 2

        return ((match / len1 + match / len2 + (match - t) / match) / 3.0)

    # Jaro Winkler Similarity
    def jaro_Winkler(s1, s2):
        jaro_dist = jaro_distance(s1, s2)
        if jaro_dist > 0.7:
            prefix = 0
            for i in range(min(len(s1), len(s2))):
                if s1[i] == s2[i]:
                    prefix += 1
                else:
                    break

            prefix = min(4, prefix)
            jaro_dist += 0.1 * prefix * (1 - jaro_dist)

        return jaro_dist

    similarity_matrix = [[jaro_Winkler(str1, str2) for str2 in list2] for str1 in list1]
    
    for i, row in enumerate(similarity_matrix):
        for j, similarity in enumerate(row):
            if similarity > threshold:
                #concept_name = mapdict_Cleanedlist[list1[i]]
                above_threshold.add(list1[i])
               
    return above_threshold
#--------------------------------------------------------------------------------
#LSA Similarity measures
global abovethreshold_phrases
abovethreshold_phrases = set()
   
def LSA_similarity(list1, list2,threshold=0.9):
      
    combined_elements = list1 + list2

    # Create a Document-Term Matrix (DTM) 
    vectorizer = CountVectorizer(min_df=1, stop_words='english')
    dtm_combined = vectorizer.fit_transform(combined_elements)

    # Apply TruncatedSVD (LSA) to the combined DTM
    n_components = 2
    lsa = TruncatedSVD(n_components=n_components, algorithm='arpack')
    dtmcombined_lsa = lsa.fit_transform(dtm_combined.astype(float))
    dtmcombined_lsa = Normalizer(copy=False).fit_transform(dtmcombined_lsa)

    # Split the LSA components back into two parts for each list
    dtmlist1_lsa = dtmcombined_lsa[:len(list1)]
    dtmlist2_lsa = dtmcombined_lsa[len(list1):]

    # Compute element-wise similarity using LSA components
    similarity_matrix = np.dot(dtmlist1_lsa, dtmlist2_lsa.T)

    # DataFrame to display the similarity matrix
    df_similarity = pd.DataFrame(similarity_matrix, index=list1, columns=list2)
    df_similarity.to_csv('document_similarity.csv')
    
    for i, phrase1 in enumerate(list1):
        for j, phrase2 in enumerate(list2):
            similarity = similarity_matrix[i, j]
            if similarity > threshold:
               # concept_name = mappdict_eo4geo[phrase1]
               abovethreshold_phrases.add(phrase1)
    return abovethreshold_phrases
#----------------------------------------------------------------------------------
# global abovethreshold_phrases
abovethreshold_phrases = set()
global threshold
   
def Word2Vec_similarity(list1, list2, glovemodel_path, word2vecoutput_file):
      
    # Convert the GloVe model to Word2Vec format 
    glove2word2vec(glovemodel_path, word2vecoutput_file)
    model = KeyedVectors.load_word2vec_format(word2vecoutput_file, binary=False)

    list1 = [text.lower() for text in list1]
    list2 = [text.lower() for text in list2]

    # Calculate the embeddings for each list
    list1_embeddings = [np.mean([model[token] for token in text.split() if token in model], axis=0) for text in list1]
    list2_embeddings = [np.mean([model[token] for token in text.split() if token in model], axis=0) for text in list2]

    similarity_scores = {}

    for phrase1 in list1:
        for phrase2 in list2:
            embeddings1 = list1_embeddings[list1.index(phrase1)]
            embeddings2 = list2_embeddings[list2.index(phrase2)]
            # Calculate similarity based on the dot product of embeddings
            similarity_score = np.dot(embeddings1, embeddings2) / (np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2))
            global threshold
            threshold = 0.9
            if np.any(similarity_score > threshold):
               similarity_scores[(phrase1, phrase2)] = similarity_score

    # return similarity_scores

    #glovemodel_path = "glove.6B.100d.txt"
    #word2vecoutput_file = "glove.6B.100d.word2vec"

    similarity_scores = Word2Vec_similarity(list1, list2, glovemodel_path, word2vecoutput_file)

    for (phrase1, phrase2), similarity_score in similarity_scores.items():
        if similarity_score > threshold:
        # concept_name = mappdict_eo4geo[phrase1]
            abovethreshold_phrases.add(phrase1)

    return abovethreshold_phrases

#----------------------------------------------------------------------------------
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/steps')
def steps():
    return render_template('steps.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    global extracted_text
    global extracted_phrases
    global http_sections
    global words_corrected
    global finished_message
    http_sections = None
    
    global eo4geo_bok_concepts
    global lower_list
    global re_stopword
    global final_list
    global similarity_results1
    global similarity_results
    global title_doc
    global message
    global message_eo4
    
    data = None
    
    selected_threshold = float(request.form.get('threshold', 0.7))
    selected_threshold = float(request.form.get('threshold', 0.8))
    selected_threshold = float(request.form.get('threshold', 0.9))
    
    if request.method == 'POST':
        if 'extract' in request.form:
            pdf_file = request.files['file']
            if pdf_file.filename.endswith('.pdf'):
                extracted_text = extract_text(pdf_file)
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
                title_doc = title_Ex(lowercased_words)
        
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
        message_eo4 = "You have extracted the EO4GEOBOK Concepts"
        

    if 'extract_keyphrases' in request.form:
        if  request.form['algorithm'] == 'yake':
            text_final = convert_string(corrected_words)
            extracted_phrases = yake_keyphrase_extraction(text_final)
        elif request.form['algorithm'] == 'patternrank':
            # text_final = convert_string(corrected_words)
            extracted_phrases = patternRank_extractor(words_corrected)
        elif request.form['algorithm'] == 'keybert':
            # text_final = convert_string(corrected_words)
            extracted_phrases = KeyBert_extrator(words_corrected)
        message = "You have selected the algorithm"
        
    if 'calculate_similarity' in request.form:
        if request.form['similarity_measure'] == 'cosine_similarity':
            if eo4geo_bok_concepts and extracted_phrases:
                similarity_results = Cosine_Similarity(eo4geo_bok_concepts, extracted_phrases, threshold=selected_threshold)
                print(similarity_results)
            else:
                print('final list is empty')
                
        elif request.form['similarity_measure'] == 'jaro_winkler_similarity':
            if final_list  and extracted_phrases:
                similarity_results = jarowinkler_similarity(final_list, extracted_phrases)
                print(similarity_results)
            else:
                print('final list is empty')
                
        elif request.form['similarity_measure'] == 'lsa':
            if eo4geo_bok_concepts and extracted_phrases:
                similarity_results = LSA_similarity(eo4geo_bok_concepts, extracted_phrases)
                print(similarity_results)
            else:
                print('final list is empty')

        elif request.form['similarity_measure'] == 'word2vec':
            glovemodel_path = "glove.6B.100d.txt"
            word2vecoutput_file = "glove.6B.100d.word2vec"
            if eo4geo_bok_concepts and extracted_phrases:
                similarity_results = Word2Vec_similarity(eo4geo_bok_concepts, extracted_phrases,glovemodel_path, word2vecoutput_file)
                print(similarity_results)
            else:
                print('final list is empty')
                
    return render_template('index.html',title_doc=title_doc, message=message,message_eo4=message_eo4, similarity_results=similarity_results)



if __name__ == '__main__':
    app.run(debug=True, port=5000)

