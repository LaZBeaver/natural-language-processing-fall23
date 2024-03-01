import re
from os import scandir
from string import punctuation
import json
import matplotlib.pyplot as plt
import numpy as np

import nltk


lemmatizer = nltk.stem.WordNetLemmatizer()

PATTERNS = {"name":                     re.compile(r"(attending:|mr.|mrs.|ms.)\s*\[\*{2}(.+?)\*{2}\]", re.IGNORECASE),
            "doctor":                   re.compile(r"(dr\.{0,1}|doctor|physician|surgen)s{0,1}\s*\[\*{2}(.+?)\*{2}\]", re.IGNORECASE),
            "date":                     re.compile(r"admission date:\s*\[\*{2}(.+?)\*{2}\]", re.IGNORECASE),
            "weight":                   re.compile(r"(\d+\.{0,1}\d*)\s*(kg|kilo|kilogram|pound|lb)s{0,1}", re.IGNORECASE),
            "age":                      re.compile(r"(\d+)\s*(year old|years old|yo|y\/o)", re.IGNORECASE),
            "diagnosis":                re.compile(r"discharge diagnosis:(.*?)(\n{4,}|discharge)", re.IGNORECASE|re.DOTALL),
            "hospital_course":          re.compile(r"brief hospital course:(.*?)(\n{4,}|medications)", re.IGNORECASE|re.DOTALL),
            "treatment":                re.compile(r"discharge instructions:(.*?)(\n{4,}|followup)", re.IGNORECASE|re.DOTALL),
            "medication_discharge":     re.compile(r"discharge medications:(.*?)(\n{4,}|discharge)", re.IGNORECASE|re.DOTALL),
            "medication_admission":     re.compile(r"medications on admission:(.*?)(\n{4,}|discharge)", re.IGNORECASE|re.DOTALL),
            "history_present_illness":  re.compile(r"history of present illness:(.*?)(\n{4,}|past)", re.IGNORECASE|re.DOTALL),
            "history_overall_medical":  re.compile(r"past medical history:(.*?)(\n{4,}|social)", re.IGNORECASE|re.DOTALL),
            "more_info":                re.compile(r"social history:(.*?)(\n{4,}|physical)", re.IGNORECASE|re.DOTALL),
            "meds":                     re.compile(r"([a-z][a-z\- ]+)\s[\d\.\-]+\s*(g|mg|mgs|mcg|unit|QAM|QPM|dl)s{0,1}", re.IGNORECASE),
            "operation":                re.compile(r"major surgical or invasive procedure(.*?)(\n{4,}|history)", re.IGNORECASE|re.DOTALL)}
            

class Patient:
    def __init__(self, text) -> None:
        self.text                       = text
        self.name                       = PATTERNS["name"].search(self.text)
        self.doctor                     = PATTERNS["doctor"].search(self.text)
        self.date                       = PATTERNS["date"].search(self.text)
        self.age                        = PATTERNS["age"].search(self.text)
        self.weight                     = PATTERNS["weight"].findall(self.text)
        self.diagnosis                  = PATTERNS["diagnosis"].search(self.text)
        self.treatment                  = PATTERNS["treatment"].search(self.text)
        self.medication_admission       = PATTERNS["medication_admission"].search(self.text)
        self.medication_discharge       = PATTERNS["medication_discharge"].search(self.text)
        self.history_overall_medical    = PATTERNS["history_overall_medical"].search(self.text)
        self.history_present_illness    = PATTERNS["history_present_illness"].search(self.text)
        self.more_info                  = PATTERNS["more_info"].search(self.text)
        self.hospital_course            = PATTERNS["hospital_course"].search(self.text)
        self.meds                       = PATTERNS["meds"].findall(self.text)
        self.operation                  = PATTERNS["operation"].search(self.text)

        self.dkind  = ""
        self.locate = ""
        

    def get_patient_name(self):
        name = "Name not found"
        if self.name:
            name = self.name.group(2)
        return name


    def get_doctor_name(self):
        doctor = "Doctor not found"
        if self.doctor:
            doctor = self.doctor.group(2)
        return doctor


    def get_date(self):
        date = "Date not found"
        if self.date:
            date = self.date.group(1)
        return date


    def get_med_names(self):
        # re.findall(r"([A-Z][a-zA-Z\- ]+)\s[\d\.\-]+\s*(g|mg|mgs|mcg|unit|QAM|QPM|%)s{0,1}"
        stop_words = ["for", "every", "per", "daily", "monthly", "once", "day", "month", "by", "mouth", "inhale", "unit", "g", "mg", "mcg", "qam", "qpm", "dl", "ml", "%", "tab", "tablet", "qhs", "po", "cap", "none", "qd", ""] + [x for x in punctuation]
        puncs = [x for x in punctuation]
        admission_med_names = []
        discharge_med_names = []

        admission_med_names_processed = []
        discharge_med_names_processed = []

        if self.medication_admission:
            admission_raw_meds = self.medication_admission.group(1)
            admission_med_names = re.findall(r"([a-z][a-z\- ]+)\s[\d\.\-]*\s*(g|mg|mgs|mcg|unit|U|QAM|QPM|%|prn|MDI|daily|weekly|monthly|qd|once|twice)", admission_raw_meds, re.IGNORECASE)
            admission_med_names = [x[0] for x in admission_med_names]
            if len(admission_med_names) == 0:
                admission_med_names = re.findall(r"([a-z][a-z\- ]+)\s[\d\.\-]+", admission_raw_meds, re.IGNORECASE)                
            if len(admission_med_names) == 0 and "," in admission_raw_meds:
                admission_med_names = [x.strip() for x in admission_raw_meds.split(",")]
            if len(admission_med_names) == 0:
                admission_med_names = [x.strip() for x in admission_raw_meds.split("\n")]
            # admission_med_names = [x for x in admission_med_names if lemmatizer.lemmatize(x.lower()) not in stop_words]
            
            for med in admission_med_names:
                temp = ""
                for word in med.split(" "):
                    if lemmatizer.lemmatize(word.lower()) not in stop_words:
                        temp += " "+word
                if temp != "":
                    admission_med_names_processed.append(temp)

        if self.medication_discharge:
            discharge_raw_meds = self.medication_discharge.group(1)
            discharge_med_names = re.findall(r"([a-z][a-z\- ]+)\s[\d\.\-]*\s*(g|mg|mgs|mcg|unit|U|QAM|QPM|%|prn|MDI|daily|weekly|monthly|qd|once|twice)", discharge_raw_meds, re.IGNORECASE)
            discharge_med_names = [x[0] for x in discharge_med_names]
            if len(discharge_med_names) == 0:
                discharge_med_names = re.findall(r"([a-z][a-z\- ]+)\s[\d\.\-]+", discharge_raw_meds, re.IGNORECASE)                
            if len(discharge_med_names) == 0 and "," in discharge_raw_meds:
                discharge_med_names = [x.strip() for x in discharge_raw_meds.split(",")]
            if len(discharge_med_names) == 0:
                discharge_med_names = [x.strip() for x in discharge_raw_meds.split("\n")]
            # discharge_med_names = [x for x in discharge_med_names if lemmatizer.lemmatize(x.lower()) not in stop_words]

            for med in discharge_med_names:
                temp = ""
                for word in med.split(" "):
                    if lemmatizer.lemmatize(word.lower()) not in stop_words:
                        temp += " "+word
                if temp != "":
                    discharge_med_names_processed.append(temp)


        return (admission_med_names_processed, discharge_med_names_processed)


    def get_age(self):
        age = "Date not found"
        if self.age:
            age = self.age.group(1)
        return age  


    def get_weight(self):
        weight = "No weight found"
        if len(self.weight) > 0:
            weight_list = [float(x[0]) for x in self.weight]
            weight = str(max(weight_list))
        return weight


    def get_disease_names(self):
        puncs = [x for x in punctuation]
        puncs += ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        stop_words = ["primary", "secondary", "primary diagnosis", "secondary diagnosis", ""]

        disease_names = []
        if self.diagnosis:
            disease_names_unprocessed = []
            diagnosis_raw_names = self.diagnosis.group(1)
            diagnosis_raw_names = diagnosis_raw_names.split("\n")
            for line in diagnosis_raw_names:
                if "," in line:
                    disease_names_unprocessed += line.split(",")
                else:
                    disease_names_unprocessed.append(line)
            
            for name in disease_names_unprocessed:
                disease_names.append("".join([x for x in name if x not in puncs]))
                disease_names = [x.strip() for x in disease_names]
                disease_names = [x for x in disease_names if x.lower() not in stop_words]
            
        return disease_names


    def get_disease_location(self):
        puncs = [x for x in punctuation]
        puncs += ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        body_parts = {"head" : "eye ear nose mouth forehead eyebrow lip cheek chin tongue tooth jaw deaf blind brain",
                     "upper_body" : "shoulder arm chest back elbow wrist hand finger thumb neck ribs abdomen waist",
                     "lower_body" : "hip leg knee thigh foot calf ankle toe buttocks groin heel shin",
                     "digestive_system" : "stomach liver intenstine kidney bladder pancreas spleen esophagus gallbladder bowel rectum rectal renal",
                     "respiratory_system" : "lungs throat thrachea bronchi diaphragm alveoli nostrils larynx pharynx pleura",
                     "circulatory_system" : "heart vein capillaries aorta artery hypertension strock vascular"}

        disease_names = self.get_disease_names()
        disease_locations = set()

        for disease in disease_names:
            for category, parts in body_parts.items():
                for part in parts.split():
                    if re.search(part, disease, re.IGNORECASE):
                        disease_locations.add(category)

        if len(disease_locations) == 0:
            disease_locations.add("No location found")
        return disease_locations


    def get_medical_history(self):
        history = "No history found"
        if self.history_overall_medical:
            history =  self.history_overall_medical.group(1)
            history = re.sub(r"\n", " ", history)
        return history


    def get_illness_history(self):
        illness = "No prior info on present illness found"
        if self.history_present_illness:
            illness = self.history_present_illness.group(1)
            illness = re.sub(r"\n", " ", illness)
        return illness
    

    def get_operation(self):
        operation_processsed = []

        if self.operation:
            operation = self.operation.group(1)
            operation = operation.split("\n")
            for opr in operation:
                temp = " ".join(list(filter(lambda word: True if word not in punctuation else False, opr.split(" "))))
                if temp not in ["", "none", "None"]:
                    operation_processsed.append(temp)

        return operation_processsed
    

    def get_more_info(self):
        social = "No social history"
        if self.more_info:
            social = self.more_info.group(1)
            social = re.sub(r"\n", " ", social)
        return social

files = scandir(r"./n2c2/part2")
patients = {}


for f in files:
    if ".txt" in f.name:
        with open(f.path, 'r') as file_dsc:
            patients[f.name.strip(".txt")] = Patient(file_dsc.read())


json_template = {"Patient Name": "",
                 "Doctor Name": "",
                 "Admission Date": "",
                 "Diagnosed Diseases": "",
                 "Age": "",
                 "Weight": "",
                 "Disease Type": "",
                 "Operation Type": "",
                 "Disease Location": "",
                 "Medication on Admission": "",
                 "Medication on Discharge": "",
                 "Medical History": "",
                 "Illness History": "",
                 "More Info": ""}
json_files = []

for number, data in patients.items():
    json_template["Patient Name"]                   = data.get_patient_name()
    json_template["Doctor Name"]                 = data.get_doctor_name()
    json_template["Admission Date"]         = data.get_date()
    json_template["Diagnosed Diseases"]     = data.get_disease_names()
    json_template["Age"]                    = data.get_age()
    json_template["Weight"]                 = data.get_weight()
    json_template["Disease Type"]           = ""
    json_template["Operation Type"]         = data.get_operation()
    json_template["Disease Location"]       = list(data.get_disease_location())
    json_template["Medication on Admission"]   = data.get_med_names()[0]
    json_template["Medication on Discharge"]   = data.get_med_names()[1]
    json_template["Medical History"]        = data.get_medical_history()
    json_template["Illness History"]        = data.get_illness_history()
    json_template["More Info"]              = data.get_more_info()
    
    with open(r"./track2_results/"+number+r".json", "w") as file_dsc:
        file_dsc.seek(0)
        file_dsc.write(json.dumps(json_template, indent=3))




