import os

import fnmatch
import nltk
from nltk.tag import hmm


import csv


from nltk.tag import CRFTagger

def preProcessing(pathToTrainingData):
    outputFile = open("processedFile.txt","w")
    checkCue = "CUE-"
    startingTag = "B-CUE"
    continueTag = "I-CUE"
    outsideTag = "O-CUE"
    processedTokens = []
    for root, dirs, files in os.walk(pathToTrainingData, topdown=True):
        for fileName in fnmatch.filter(files, '*.txt'):
            file = open(os.path.join(pathToTrainingData, fileName), 'r')
            isCueFirst = True
            lineList = []
            for line in file:
                tokens=line.split()
                if len(tokens)<3:
                    continue
                if checkCue in line:
                    if isCueFirst==True:
                        tokens[2] = startingTag
                        isCueFirst = False
                    else:
                        tokens[2] = continueTag
                else:
                    isCueFirst = True
                    tokens[2] = outsideTag
                stringJoin1 = tokens[0]+"\t"+tokens[1]+"\t"+tokens[2]
                processedTokens.append(stringJoin1)
                outputFile.write(stringJoin1+'\n')
    outputFile.close()
    return processedTokens


def preProcessingOld(pathToTrainingData):
    checkCue = "CUE-"
    startingTag = "B-CUE"
    continueTag = "I-CUE"
    outsideTag = "O-CUE"
    for root, dirs, files in os.walk(pathToTrainingData, topdown=True):
        trainingList = []
        for fileName in fnmatch.filter(files, '*.txt'):
            file = open(os.path.join(pathToTrainingData, fileName), 'r')
            isCueFirst = True
            lineList = []
            for line in file:
                tokens=line.split()
                if len(tokens)<3:
                    continue
                if checkCue in line:
                    if isCueFirst==True:
                        tokens[2] = startingTag
                        isCueFirst = False
                    else:
                        tokens[2] = continueTag
                else:
                    isCueFirst = True
                    tokens[2] = outsideTag
                stringJoin1 = tokens[0]
                stringJoin2 = tokens[2]
                tupple = (stringJoin1.decode('utf-8'), stringJoin2.decode('utf-8'))
                lineList.append(tupple)
                if (tokens[0] is "."):
                    trainingList.append(lineList)
                    lineList = []

    return trainingList

def preprocessForTagging(testFolder):
    listSentences = []
    sentence =[]
    for root, dirs, files in os.walk(testFolder, topdown=True):
        for fileName in fnmatch.filter(files, '*.txt'):
            file = open(os.path.join(testFolder, fileName), 'r')
            if len(sentence) > 0:
                listSentences.append(sentence)
                sentence = []
            for line in file:
                lineTokens = line.split()
                if (len(lineTokens) >= 2):
                    if lineTokens[0]=='.':
                        sentence.append(lineTokens[0].decode('utf-8'))
                        listSentences.append(sentence)
                        sentence = []
                    else:
                        sentence.append(lineTokens[0].decode('utf-8'))
    return listSentences



def performTagging(ct, testFolder):

    testSentences = preprocessForTagging(testFolder)
    taggedSentences = ct.tag_sents(testSentences)
    tag = "null"
    tokenId = 0
    uncertainRanges = ""
    rangeStart = -1
    rangeEnd = -1
    for sentence in taggedSentences:
        for word in sentence:
            prevTag = tag
            tag = word[1]
            if("B-CUE" in tag):
                rangeStart = tokenId
                rangeEnd = tokenId

            elif("I-CUE" in tag):
                rangeEnd = tokenId
            elif("O-CUE" in tag):
                tag = "_"
                if ("CUE" in prevTag):
                    uncertainRanges = uncertainRanges + str(rangeStart) + "-" + str(rangeEnd) + " "
            tokenId += 1
    returnData = {}
    returnData["phraseRanges"] = uncertainRanges
    #returnData["sentenceRanges"] = uncertainSentenceRange
    return returnData

pathToTrainingData = "/Users/shraddha/Documents/Semester 2/NLP/Project2/nlp_project2_uncertainty/train"
processedTokens = preProcessingOld(pathToTrainingData)
ct = CRFTagger()
ct.train(processedTokens,'model.crf.tagger')


publicTestFolder = "/Users/shraddha/Documents/Semester 2/NLP/Project2/nlp_project2_uncertainty/test-public"
privateTestFolder = "/Users/shraddha/Documents/Semester 2/NLP/Project2/nlp_project2_uncertainty/test-private"


#privateTestFolder = "C:/Users/Reema Bajwa/PycharmProjects/Project2/nlp_project2_uncertainty/test-private"
publicResult = performTagging(ct, publicTestFolder)
privateResult = performTagging(ct, privateTestFolder)
#privateResult = performTagging(tagger, privateTestFolder, "C:/Users/Reema Bajwa/PycharmProjects/Project2/nlp_project2_uncertainty/baseline2ResultsPrivate")

with open('predictionPhrase.csv', 'w') as csvfile:
    fieldnames = ['Type', 'Spans']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'Type': 'CUE-public', 'Spans': publicResult["phraseRanges"]})
    writer.writerow({'Type': 'CUE-private', 'Spans': privateResult["phraseRanges"]})



#with open('predictionSentence.csv', 'w') as csvfile:
#    fieldnames = ['Type', 'Indices']
#    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#    writer.writeheader()
#    writer.writerow({'Type': 'SENTENCE-public', 'Indices': publicResult["sentenceRanges"]})
#    writer.writerow({'Type': 'SENTENCE-private', 'Indices': privateResult["sentenceRanges"]})


    

