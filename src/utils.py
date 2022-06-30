import csv
from typing import List,Dict

def writecsv_listofdicts(filepath:str,listofdicts:List[Dict]):
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = list(listofdicts[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for d in listofdicts:
            writer.writerow(d)

def writecsv_listofstrings(filepath:str,fieldnames:str,listofstrings: List[str]):
    with open(filepath, 'w', newline='') as csvfile:
        csvfile.write(fieldnames+'\n')
        csvfile.writelines(listofstrings)

def readcsv_listofdicts(filepath:str):
    listofdicts: List[Dict] = []
    with open(filepath, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            listofdicts.append(row)
    return listofdicts

def readcsv_listofstrings(filepath:str):
    listofstrings: List[str] = []
    with open(filepath, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            listofstrings.append(str(row))
    return listofstrings