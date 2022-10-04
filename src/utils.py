import csv
from typing import List,Dict
import matplotlib.pyplot as plt
import numpy as np
import os

def writecsv_listofdicts(filepath:str,listofdicts:List[Dict]):
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = list(listofdicts[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for d in listofdicts:
            writer.writerow(d)

def writecsv_dictionary(filepath:str,dict:Dict):
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = dict.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(dict)

def writecsv_listofstrings(filepath:str,fieldnames:str,listofstrings: List[str]):
    with open(filepath, 'w', newline='') as csvfile:
        csvfile.write(fieldnames)
        csvfile.writelines(listofstrings)

def readcsv_listofstrings(filepath:str):
    with open(filepath, 'r', newline='') as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
        return lines
    return None

def readcsv_listofdicts(filepath:str):
    listofdicts: List[Dict] = []
    with open(filepath, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            listofdicts.append(row)
    return listofdicts

def print_dict(dict:Dict):
    for key,value in dict.items():
        print(key,': ',value)
def print_listofstrings(fieldnames:str,listofstrings: List[str]):
    print(fieldnames.strip())
    for line in listofstrings:
        print(line.strip())

def save_pareto_plot(folder_path:str,title:str,x:np.ndarray,y:np.ndarray,x_name:str, y_name:str):
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(x, y, '.b')
    ax.set_title('Pareto front: '+title+', {} designs'.format(x.size))
    ax.set_ylabel(y_name, loc='center')
    ax.set_xlabel(x_name, loc='center')
    #plt.legend(loc='upper left')
    ax.grid(True)
    plt.savefig(os.path.join(folder_path,title + "_pf.png"), bbox_inches="tight")
    plt.close(fig)

def save_pareto_plot_wr(folder_path: str, title: str,
                        x: np.ndarray, y: np.ndarray,
                        xref: np.ndarray, yref: np.ndarray,
                        x_name: str, y_name: str):
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(xref, yref, '+r',label = 'reference front')
    ax.plot(x, y, '.b',label = 'current front')
    ax.legend()
    ax.set_title('Pareto front: ' + title + ', {} designs'.format(x.size))
    ax.set_ylabel(y_name, loc='center')
    ax.set_xlabel(x_name, loc='center')
    # plt.legend(loc='upper left')
    ax.grid(True)
    plt.savefig(os.path.join(folder_path, title + "_pf.png"), bbox_inches="tight")
    plt.close(fig)