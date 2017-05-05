# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 15:01:35 2017

@author: George O'Donnell
"""


import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans

import pandas as pd
import numpy
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning,
module="pandas", lineno=570)
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import zeros

from scipy.linalg import svd

import re, math
from collections import Counter

wordConvertor = re.compile(r'\w+')

 
def jaro(a, b):
    string1Length = len(a)
    string2Length = len(b)
    
    #Create matching distance. Characters are considered matching only if they are not 
    #greater than this distance     
    characterMatchDistance = (max(string1Length, string2Length) // 2) - 1
    
    string1Matches = [False] * string1Length
    string2Matches = [False] * string2Length
 
    matchingCharacters = 0
    transpositions = 0
    z = 0
    
    #Find number of matching character between both strings
    for x in range(string1Length):
        start = max(0, (x - characterMatchDistance))
        end = min(string2Length, x + (characterMatchDistance +1))
        
        for y in range(start, end):
            if string2Matches[y]:
                continue
            
            if a[x] != b[y]:
                continue
            
            matchingCharacters = matchingCharacters + 1
            string1Matches[x] = True
            string2Matches[y] = True
            break
 
        
    if matchingCharacters == 0:
        return 0
 
    #Calculates the number of transpositional characters
    for x in range(string1Length):
        if not string1Matches[x]:
            continue
        
        while not string2Matches[z]:
            z = z + 1
        
        if a[x] != b[z]:
            transpositions = transpositions + 1
        
        z = z + 1
 
    
    equationUpper = (matchingCharacters / string1Length) + (matchingCharacters / string2Length) + ((matchingCharacters - transpositions/2) / matchingCharacters)
    jaroDistance = equationUpper / 3
    return jaroDistance
   
   
def lcs(a, b):
    string1Length = len(a)
    string2Length = len(b)
    
    longest = 0
    lcsResult = set()
    
    #Create list of zeros, used for determining longest common substring
    counter = [[0]*(string2Length + 1) for x in range(string1Length + 1)]
    
    for y in range(string1Length):
        for z in range(string2Length):
            #If the characters match, increment the unit position one down and one to the left
            if a[y] == b[z]:
                count = counter[y][z] + 1
                counter[y+1][z+1] = count
                
                #Keep track of the longest substring found
                if count > longest:
                    longest = count
                    lcsResult = set()
                    lcsResult.add(a[(y-count)+1 : y+1])
                
                elif count == longest:
                    lcsResult.add(a[(y-count)+1 : y+1])

    
    return lcsResult 
    
    
def damerauLevenshteinDistance(a, b):
    string1Length = len(a)
    string2Length = len(b)
    dict = {}
    
    #Populating dictionary with -1s and character positions, with character position as key
    for x in range(-1, string1Length + 1):
        dict[(x, -1)] = x + 1
             
    for x in range(-1, string2Length + 1):
        dict[(-1, x)] = x + 1
 
             
    for x in range(string1Length):
        for y in range(string2Length):
            #If the character are the same the cost to substitute or transpose one
            #for the other is 0
            if a[x] != b[y]:
                cost = 1
            else:
                cost = 0
            
            #The minimum operation, from insertion, deletion, substitution, and transposition,
            #is set as the current characters key in the dictionary
            dict[(x, y)] = min(dict[(x, y-1)] + 1, dict[(x-1, y)] + 1, dict[(x-1, y-1)] + cost)
            
            if x and y and a[x] == b[y-1] and a[x-1] == b[y]:
                dict[(x, y)] = min(dict[(x, y)], dict[x-2, y-2] + cost)

    
    return dict[string1Length-1, string2Length-1]
    
    
#This method returns a term frequency table for a given string    
def makeVector(string):
    newString = wordConvertor.findall(string)
    return Counter(newString)  
     
  
def cosine(a, b):
    #Places the common words from both strings into a set
    commonWords = set(a.keys()) & set(b.keys())
    
    #For every common word in both strings, the frequency of that word for each string is 
    #multiplied, with the resulting values being added together
    upper = sum([a[y] * b[y] for y in commonWords])
    
    #For every word in string 1, the sum of the square value for each word frequency in
    #that string is calculated. The same is done for string 2
    string1squared = sum([a[z]**2 for z in a.keys()])
    string2squared = sum([b[z]**2 for z in b.keys()])

    lower = math.sqrt(string1squared) * math.sqrt(string2squared)
     
    cosine = (upper / lower)
    
    return cosine     

        
def dicesCoefficient(a, b):
    #Add every bigram in string a into a list
    string1BigramList=[]
    for x in range(len(a)-1):
        string1BigramList.append(a[x:x+2])
   
    #Add every bigram in string b into a list
    string2BigramList=[]
    for y in range(len(b)-1):
        string2BigramList.append(b[y:y+2])
    
    string1Bigrams = set(string1BigramList)
    string2Bigrams = set(string2BigramList)
    
    #Overall length of common bigrams in both strings
    overallLength = len(string1Bigrams & string2Bigrams)
    upper = overallLength * 2
    lower = len(string1Bigrams) + len(string2Bigrams)
    dC = upper / lower

    return dC


def LSA(dict):
    #Create word frequency matrix
    vectorizer = CountVectorizer(min_df = 1, stop_words = 'english')
    termMatrix = vectorizer.fit_transform(dict)
    pd.DataFrame(termMatrix.toarray(),index=dict,columns=vectorizer.get_feature_names()).head(10)
    
    # Get words that corresponds to each column
    vectorizer.get_feature_names()
    
    #Use SVD to reconstruct the matrix with less information
    lsa = TruncatedSVD(2, algorithm = 'randomized')
    termMatrixLSA = lsa.fit_transform(termMatrix)
    termMatrixLSA = Normalizer(copy=False).fit_transform(termMatrixLSA)
    
    pd.DataFrame(lsa.components_,index = ["component_1","component_2"],columns = vectorizer.get_feature_names())
    
    pd.DataFrame(termMatrixLSA, index = dict, columns = ["component_1","component_2"])
    
    # Compute document similarity using LSA components
    similarity = numpy.asarray(numpy.asmatrix(termMatrixLSA) * numpy.asmatrix(termMatrixLSA).T)
    normalisedLSA = similarity[0].item(1)
    return normalisedLSA            
            
    
def main():

    data = pd.read_csv("fullTraining.csv", delimiter=",")
    #data = pd.read_csv("fullTesting.csv", delimiter=",")
    
    data = data.dropna()
    
    string1List = data['#1 String']
    string2List = data['#2 String']
    similarity = data['Quality']
    
    thresholdValue = .5
    
    dict = {} 
    counter = 0
    
    totalSimilar = 0
    totalDissimilar = 0
    
    cosineTotalCorrectSimilar = 0
    cosineTotalCorrectNotSimilar = 0
    cosineTotalCorrect = 0
    cosinePredictedSimilar = 0
    cosinePredictedNotSimilar = 0
    cosineSimilar = 0
    
    DCTotalCorrectSimilar = 0
    DCTotalCorrectNotSimilar = 0
    DCTotalCorrect = 0
    DCPredictedSimilar = 0
    DCPredictedNotSimilar = 0
    DCSimilar = 0
    
    DLDTotalCorrectSimilar = 0
    DLDTotalCorrectNotSimilar = 0
    DLDTotalCorrect = 0
    DLDPredictedSimilar = 0
    DLDPredictedNotSimilar = 0
    DLDSimilar = 0
    
    LCSTotalCorrectSimilar = 0
    LCSTotalCorrectNotSimilar = 0
    LCSTotalCorrect = 0
    LCSPredictedSimilar = 0
    LCSPredictedNotSimilar = 0
    LCSSimilar = 0
    
    JDTotalCorrectSimilar = 0
    JDTotalCorrectNotSimilar = 0
    JDTotalCorrect = 0
    JDPredictedSimilar = 0
    JDPredictedNotSimilar = 0
    JDSimilar = 0
    
    LSATotalCorrectSimilar = 0
    LSATotalCorrectNotSimilar = 0
    LSATotalCorrect = 0
    LSAPredictedSimilar = 0
    LSAPredictedNotSimilar = 0
    LSASimilar = 0
    
    overallTotalCorrectSimilar = 0
    overallTotalCorrectNotSimilar = 0
    overallTotalCorrect = 0
    overallPredictedSimilar = 0
    overallPredictedNotSimilar = 0
    
    menu = 1
    
    #Determines the similarity between each sentence pair based on the similarity metric in the dataset
    for files in string1List:
        if similarity[counter] == 0:
            totalDissimilar = totalDissimilar + 1
        else:
            totalSimilar = totalSimilar + 1
        counter = counter + 1    
        
    counter = 0
    
    
    while menu == 1:
        print("\nMAIN MENU")
        
        menuChoice = input("View threshold value and data stats - Press 1\nRun similarity algorithms on data - Press 2\n"+
        "View algorithm accuracies - Press 3\nView overall system accuracy - Press 4\n"+
        "Change threshold value - Press 5\nQuit System - Press 6\n")
        if menuChoice <= str(0) or menuChoice >= str(7):
            print("Invalid menu choice\n")
            
        else:
            #Prints the dataset and threshold stats
            if menuChoice == str(1):
                print("\nTotal similar strings in data: " + str(totalSimilar))
                print("Total dissimilar strings in data: " + str(totalDissimilar))
                print("Theshold value: " + str(thresholdValue))
                print("\n")
            
            #Runs each similarity algorithm 
            elif menuChoice == str(2):
                counter = 0
                cosineTotalCorrectSimilar = 0
                cosineTotalCorrectNotSimilar = 0
                cosineTotalCorrect = 0
                cosinePredictedSimilar = 0
                cosinePredictedNotSimilar = 0
                cosineSimilar = 0
                
                DCTotalCorrectSimilar = 0
                DCTotalCorrectNotSimilar = 0
                DCTotalCorrect = 0
                DCPredictedSimilar = 0
                DCPredictedNotSimilar = 0
                DCSimilar = 0
                
                DLDTotalCorrectSimilar = 0
                DLDTotalCorrectNotSimilar = 0
                DLDTotalCorrect = 0
                DLDPredictedSimilar = 0
                DLDPredictedNotSimilar = 0
                DLDSimilar = 0
                
                LCSTotalCorrectSimilar = 0
                LCSTotalCorrectNotSimilar = 0
                LCSTotalCorrect = 0
                LCSPredictedSimilar = 0
                LCSPredictedNotSimilar = 0
                LCSSimilar = 0
                
                JDTotalCorrectSimilar = 0
                JDTotalCorrectNotSimilar = 0
                JDTotalCorrect = 0
                JDPredictedSimilar = 0
                JDPredictedNotSimilar = 0
                JDSimilar = 0
                
                LSATotalCorrectSimilar = 0
                LSATotalCorrectNotSimilar = 0
                LSATotalCorrect = 0
                LSAPredictedSimilar = 0
                LSAPredictedNotSimilar = 0
                LSASimilar = 0
                
                overallTotalCorrectSimilar = 0
                overallTotalCorrectNotSimilar = 0
                overallTotalCorrect = 0
                overallPredictedSimilar = 0
                overallPredictedNotSimilar = 0
                
                print("Running similarity algorithms\n...\nPlease wait\n")
                for files in string1List:                    
                    ##Cosine##
                    vector1 = makeVector(string1List[counter])
                    vector2 = makeVector(string2List[counter])
                    cosineValue = cosine(vector1, vector2)
                    
                    if cosineValue <= thresholdValue:
                        cosinePredictedNotSimilar = cosinePredictedNotSimilar + 1
                        cosineSimilar = 0
                        
                        if similarity[counter] == 0:
                            cosineTotalCorrect = cosineTotalCorrect + 1
                            cosineTotalCorrectNotSimilar = cosineTotalCorrectNotSimilar + 1 
                        
                    else:
                        cosinePredictedSimilar = cosinePredictedSimilar + 1
                        cosineSimilar = 1
                          
                        if similarity[counter] == 1:
                            cosineTotalCorrect = cosineTotalCorrect + 1
                            cosineTotalCorrectSimilar = cosineTotalCorrectSimilar + 1
                          
                    
                    ##Dice's Coefficient##
                    diceCoefficient = dicesCoefficient(string1List[counter], string2List[counter])

                    if diceCoefficient <=thresholdValue:
                        DCPredictedNotSimilar = DCPredictedNotSimilar + 1
                        DCSimilar = 0
                               
                        if similarity[counter] == 0:
                            DCTotalCorrect = DCTotalCorrect + 1
                            DCTotalCorrectNotSimilar = DCTotalCorrectNotSimilar + 1 
                        
                    else:
                        DCPredictedSimilar = DCPredictedSimilar + 1
                        DCSimilar = 1
                        
                        if similarity[counter] == 1:
                            DCTotalCorrect = DCTotalCorrect + 1
                            DCTotalCorrectSimilar = DCTotalCorrectSimilar + 1
                             
                    
                    ##Damerau Levenshtein Distance##
                    DLD = damerauLevenshteinDistance(string1List[counter], string2List[counter])
                    normalisedDLD = 1 - (DLD / max(len(string1List[counter]), len(string2List[counter])))

                    if normalisedDLD <=thresholdValue:
                        DLDPredictedNotSimilar = DLDPredictedNotSimilar + 1
                        DLDSimilar = 0
                        
                        if similarity[counter] == 0:
                            DLDTotalCorrect = DLDTotalCorrect + 1
                            DLDTotalCorrectNotSimilar = DLDTotalCorrectNotSimilar + 1 
                       
                    else:
                        DLDPredictedSimilar = DLDPredictedSimilar + 1
                        DLDSimilar = 1
                        
                        if similarity[counter] == 1:
                            DLDTotalCorrect = DLDTotalCorrect + 1
                            DLDTotalCorrectSimilar = DLDTotalCorrectSimilar + 1
                               
                    
                    ##Longest Common Substring##
                    l = ""
                    a = lcs(string1List[counter], string2List[counter])
                    for s in a:
                        l = l + s
                    
                    normalisedLCS = len(l) / len(string1List[counter]) 

                    if normalisedLCS <=thresholdValue:
                        LCSPredictedNotSimilar = LCSPredictedNotSimilar + 1
                        LCSSimilar = 0
                        
                        if similarity[counter] == 0:
                            LCSTotalCorrect = LCSTotalCorrect + 1
                            LCSTotalCorrectNotSimilar = LCSTotalCorrectNotSimilar + 1 
                        
                    else:
                        LCSPredictedSimilar = LCSPredictedSimilar + 1
                        LCSSimilar = 1
                        
                        if similarity[counter] == 1:
                            LCSTotalCorrect = LCSTotalCorrect + 1
                            LCSTotalCorrectSimilar = LCSTotalCorrectSimilar + 1
                              
                    
                    ##Jaro Distance##
                    jaroValue = jaro(string1List[counter], string2List[counter])

                    if jaroValue <=thresholdValue:
                        JDPredictedNotSimilar = JDPredictedNotSimilar + 1
                        JDSimilar = 0
                        
                        if similarity[counter] == 0:
                            JDTotalCorrect = JDTotalCorrect + 1
                            JDTotalCorrectNotSimilar = JDTotalCorrectNotSimilar + 1 
                        
                    else:
                        JDPredictedSimilar = JDPredictedSimilar + 1
                        JDSimilar = 1
                        
                        if similarity[counter] == 1:
                            JDTotalCorrect = JDTotalCorrect + 1
                            JDTotalCorrectSimilar = JDTotalCorrectSimilar + 1
                        
                                                
                    ##Latent Semantic Analysis##
                    dict = [string1List[counter], string2List[counter]]
                    lsaValue = LSA(dict)

                    if lsaValue <=thresholdValue:
                        LSAPredictedNotSimilar = LSAPredictedNotSimilar + 1
                        LSASimilar = 0
                        
                        if similarity[counter] == 0:
                            LSATotalCorrect = LSATotalCorrect + 1
                            LSATotalCorrectNotSimilar = LSATotalCorrectNotSimilar + 1 
                        
                    else:
                        LSAPredictedSimilar = LSAPredictedSimilar + 1
                        LSASimilar = 1
                        
                        if similarity[counter] == 1:
                            LSATotalCorrect = LSATotalCorrect + 1
                            LSATotalCorrectSimilar = LSATotalCorrectSimilar + 1
                        
                                        
                    #Overall system similarity verdict    
                    similarVerdict = (cosineSimilar + DCSimilar + DLDSimilar + LCSSimilar + JDSimilar + LSASimilar)/6
                    
                    if similarVerdict < thresholdValue:
                        overallPredictedNotSimilar = overallPredictedNotSimilar + 1
                        
                        if similarity[counter] == 0:
                            overallTotalCorrect = overallTotalCorrect + 1
                            overallTotalCorrectNotSimilar = overallTotalCorrectNotSimilar + 1 
                    
                    else:
                        overallPredictedSimilar = overallPredictedSimilar + 1
                        if similarity[counter] == 1:
                            overallTotalCorrect = overallTotalCorrect + 1
                            overallTotalCorrectSimilar = overallTotalCorrectSimilar + 1
                            
                    counter = counter + 1  
                    
                print("Similarity algorithms have completed\n")
            
                
            #Prints individual algorithm accuracies
            elif menuChoice == str(3):
                ##Cosine Accuracies##
                print("Cosine Similarity Accuracy")
                print("--------------------------")
                
                print("\nPredicted similar: " + str(cosinePredictedSimilar))
                print("Predicted not similar: " + str(cosinePredictedNotSimilar))
            
                print("\nPredicted similar corrctly: " + str(cosineTotalCorrectSimilar)) 
                print("Predicted not similar corrctly: " + str(cosineTotalCorrectNotSimilar))
                
                print("\nTotal correctly evaluated: " + str(cosineTotalCorrect))
                
                cosineSimilarAccuracy = round(((cosineTotalCorrectSimilar/totalSimilar) * 100), 2)
                cosineNotSimilarAccuracy = round(((cosineTotalCorrectNotSimilar/totalDissimilar) * 100), 2)
                print("\nSimilar accuracy: " + str(cosineSimilarAccuracy) + "%")
                print("Not similar accuracy: " + str(cosineNotSimilarAccuracy) + "%")
                
                cosineAccuracy = (cosineSimilarAccuracy + cosineNotSimilarAccuracy)/200
                roundedCosineAccuracy = round((cosineAccuracy * 100), 2)
                print("Overall algorithm accuracy: " + str(roundedCosineAccuracy) + "%")
                print("\n")
                
                
                
                ##Dice's Coefficient Accuracies##
                print("Dice's Coefficient Accuracy")
                print("---------------------------")
                
                print("\nPredicted similar: " + str(DCPredictedSimilar))
                print("Predicted not similar: " + str(DCPredictedNotSimilar))
            
                print("\nPredicted similar corrctly: " + str(DCTotalCorrectSimilar)) 
                print("Predicted not similar corrctly: " + str(DCTotalCorrectNotSimilar))
                
                print("\nTotal correctly evaluated: " + str(DCTotalCorrect))
                
                DCSimilarAccuracy = round(((DCTotalCorrectSimilar/totalSimilar) * 100), 2)
                DCNotSimilarAccuracy = round(((DCTotalCorrectNotSimilar/totalDissimilar) * 100), 2)
                print("\nSimilar accuracy: " + str(DCSimilarAccuracy) + "%")
                print("Not similar accuracy: " + str(DCNotSimilarAccuracy) + "%")
                
                DCAccuracy = (DCSimilarAccuracy + DCNotSimilarAccuracy)/200
                roundDCAccuracy = round((DCAccuracy * 100), 2)
                print("Overall algorithm accuracy: " + str(roundDCAccuracy) + "%")
                print("\n")
                
                
                           
                ##Damerau Levenshtein Distance Accuracies##
                print("Damerau Levenshtein Distance Accuracy")
                print("-------------------------------------")
                
                print("\nPredicted similar: " + str(DLDPredictedSimilar))
                print("Predicted not similar: " + str(DLDPredictedNotSimilar))
            
                print("\nPredicted similar corrctly: " + str(DLDTotalCorrectSimilar)) 
                print("Predicted not similar corrctly: " + str(DLDTotalCorrectNotSimilar))
                
                print("\nTotal correctly evaluated: " + str(DLDTotalCorrect))
                
                DLDSimilarAccuracy = round(((DLDTotalCorrectSimilar/totalSimilar) * 100), 2)
                DLDNotSimilarAccuracy = round(((DLDTotalCorrectNotSimilar/totalDissimilar) * 100), 2)
                print("\nSimilar accuracy: " + str(DLDSimilarAccuracy) + "%")
                print("Not similar accuracy: " + str(DLDNotSimilarAccuracy) + "%")
                
                DLDAccuracy = (DLDSimilarAccuracy + DLDNotSimilarAccuracy)/200
                roundDLDAccuracy = round((DLDAccuracy * 100), 2)
                print("Overall algorithm accuracy: " + str(roundDLDAccuracy) + "%")
                print("\n")
                
                
                      
                ##Longest Common Substring Accuracies##
                print("Longest Common Substring Accuracy")
                print("---------------------------------")
                
                print("\nPredicted similar: " + str(LCSPredictedSimilar))
                print("Predicted not similar: " + str(LCSPredictedNotSimilar))
            
                print("\nPredicted similar corrctly: " + str(LCSTotalCorrectSimilar)) 
                print("Predicted not similar corrctly: " + str(LCSTotalCorrectNotSimilar))
                
                print("\nTotal correctly evaluated: " + str(LCSTotalCorrect))
                
                LCSSimilarAccuracy = round(((LCSTotalCorrectSimilar/totalSimilar) * 100), 2)
                LCSNotSimilarAccuracy = round(((LCSTotalCorrectNotSimilar/totalDissimilar) * 100), 2)
                print("\nSimilar accuracy: " + str(LCSSimilarAccuracy) + "%")
                print("Not similar accuracy: " + str(LCSNotSimilarAccuracy) + "%")
                
                LCSAccuracy = (LCSSimilarAccuracy + LCSNotSimilarAccuracy)/200
                roundLCSAccuracy = round((LCSAccuracy * 100), 2)
                print("Overall algorithm accuracy: " + str(roundLCSAccuracy) + "%")
                print("\n")
                
                
                            
                ##Jaro Distance Accuracies##
                print("Jaro Distance Accuracy")
                print("----------------------")
                
                print("\nPredicted similar: " + str(JDPredictedSimilar))
                print("Predicted not similar: " + str(JDPredictedNotSimilar))
            
                print("\nPredicted similar corrctly: " + str(JDTotalCorrectSimilar)) 
                print("Predicted not similar corrctly: " + str(JDTotalCorrectNotSimilar))
                
                print("\nTotal correctly evaluated: " + str(JDTotalCorrect))
                
                JDSimilarAccuracy = round(((JDTotalCorrectSimilar/totalSimilar) * 100), 2)
                JDNotSimilarAccuracy = round(((JDTotalCorrectNotSimilar/totalDissimilar) * 100), 2)
                print("\nSimilar accuracy: " + str(JDSimilarAccuracy) + "%")
                print("Not similar accuracy: " + str(JDNotSimilarAccuracy) + "%")
                
                JDAccuracy = (JDSimilarAccuracy + JDNotSimilarAccuracy)/200
                roundJDAccuracy = round((JDAccuracy * 100), 2)
                print("Overall algorithm accuracy: " + str(roundJDAccuracy) + "%")
                print("\n")
                
                
                ##Latent Semantic Analysis Accuracies##
                print("Latent Semantic Analysis Accuracy")
                print("---------------------------------")
                
                print("\nPredicted similar: " + str(LSAPredictedSimilar))
                print("Predicted not similar: " + str(LSAPredictedNotSimilar))
            
                print("\nPredicted similar corrctly: " + str(LSATotalCorrectSimilar)) 
                print("Predicted not similar corrctly: " + str(LSATotalCorrectNotSimilar))
                
                print("\nTotal correctly evaluated: " + str(LSATotalCorrect))
                
                LSASimilarAccuracy = round(((LSATotalCorrectSimilar/totalSimilar) * 100), 2)
                LSANotSimilarAccuracy = round(((LSATotalCorrectNotSimilar/totalDissimilar) * 100), 2)
                print("\nSimilar accuracy: " + str(LSASimilarAccuracy) + "%")
                print("Not similar accuracy: " + str(LSANotSimilarAccuracy) + "%")
                
                LSAAccuracy = (LSASimilarAccuracy + LSANotSimilarAccuracy)/200
                roundLSAAccuracy = round((LSAAccuracy * 100), 2)
                print("Overall algorithm accuracy: " + str(roundLSAAccuracy) + "%")
                print("\n")
                counter = 0
                
                
            #Prints the overall system accuracy   
            elif menuChoice == str(4):
                ##Overall Accuracys##
                print("Overall system accuracy")
                print("-----------------------")
                
                print("\nPredicted similar: " + str(overallPredictedSimilar))
                print("Predicted not similar: " + str(overallPredictedNotSimilar))
            
                print("\nPredicted similar corrctly: " + str(overallTotalCorrectSimilar)) 
                print("Predicted not similar corrctly: " + str(overallTotalCorrectNotSimilar))
                
                print("\nTotal correctly evaluated: " + str(overallTotalCorrect))
                
                overallSimilarAccuracy = round(((overallTotalCorrectSimilar/totalSimilar) * 100), 2)
                overallNotSimilarAccuracy = round(((overallTotalCorrectNotSimilar/totalDissimilar) * 100), 2)
                print("\nSimilar accuracy: " + str(overallSimilarAccuracy) + "%")
                print("Not similar accuracy: " + str(overallNotSimilarAccuracy) + "%")
                
                overallAccuracy = (overallSimilarAccuracy + overallNotSimilarAccuracy)/200
                overallRoundedAccuracy = round((overallAccuracy * 100), 2)
                print("Overall system accuracy: " + str(overallRoundedAccuracy) + "%")
                print("\n")   
                counter = 0
                
            
            #Allows user to enter a new threshold value
            elif menuChoice == str(5):
                new = input("Enter new threshold value")
                if new <= str(-1) or new >= str(1):
                    print("Threshold value must be a value between 0 and 1, value has not been changed")
                else:
                    thresholdValue = float(new)
            
            
            #Exits system
            elif menuChoice == str(6):
                menu = 0
        
    
   
main()