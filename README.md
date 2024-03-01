# natural-language-processing-fall23
 This repo includes the homeworks I've done for NLP course in fall 2023 semester taught by Dr Ehsaneddin Asgari.

# Home work 1
 In this homework, we were supposed to gather the data by ourselves.
 So I chose to crawl the World Health Organization website and extract the fact sheet from it.
 After that I did the pre processing on the acquired text and implemented a simple text summarizer.


# Home Work 2
Hw2 was all about regular expressions, we were given the n2c2 2018 challenge dataset (track 2).
The aforementioned dataset includes a series of documents about patients, it basically is a clinical information dataset.
The task was to extract these name entities using regex:
    Patient's Name, Age, Weight
    Doctors's Name
    Visit and Admission Date
    Diagnosed Diseases
    Location of Injured Parts
    Medications and Treatment
    History of Ilness


# Home Work 3
 This Home work was about language models and text embeddings.
 We were given DrugBank dataset which is in english, and another similar set in farsi language. 
 Utilizing FastText and BERT, I transformted the drug descriptions into embedding vectors and stored them in a pickled object.
 The goal was this: given a drug description, the program should return the name of 5 most smiliar drugs based on the input.
 This was done by comparing the input embedding to those of the pickled object.