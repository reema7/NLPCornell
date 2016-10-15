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
            for line in file:
                lineTokens = line.split()
                if (len(lineTokens) >= 2):
                    if lineTokens[0]=='.':
                        listSentences.append(sentence)
                    else:
                        sentence.append(lineTokens[0].decode('utf-8'))
    return listSentences




def performTagging(tagger, testFolder, outputFolder):
    tokenId = 0
    sentenceId = 0
    foundInCurrentSentence = "false"
    uncertainRanges = ""
    uncertainSentenceRange = ""
    rangeStart = -1
    rangeEnd = -1
    for root, dirs, files in os.walk(testFolder, topdown=True):
        for fileName in fnmatch.filter(files, '*.txt'):
            fileTokens = []
            counter = 1
            tag = "null"
            file = open(os.path.join(testFolder, fileName), 'r')
            for line in file:
                lineTokens = line.split()
                if(len(lineTokens) >= 2):
                    word = lineTokens[0] + " " + lineTokens[1]
                    fileTokens.append(word)

            tagging = tagger.tag_sents(fileTokens)
            newFile = open(outputFolder + "/" + fileName, "w")
            for tupple in tagging:
                input = tupple[0]
                if ". ." in input:
                    if foundInCurrentSentence is "true":
                        uncertainSentenceRange = uncertainSentenceRange + str(sentenceId) + " "
                        foundInCurrentSentence = "false"
                    sentenceId = sentenceId + 1
                prevTag = tag
                tag = tupple[1]
                if("B-CUE" in tag):
                    tag = "CUE-" + str(counter)
                    rangeStart = tokenId
                    rangeEnd = tokenId
                    foundInCurrentSentence = "true"
                elif("I-CUE" in tag):
                    tag = "CUE-" + str(counter)
                    rangeEnd = tokenId
                elif("O-CUE" in tag):
                    tag = "_"
                    if ("CUE" in prevTag):
                        counter += 1
                        uncertainRanges = uncertainRanges + str(rangeStart) + "-" + str(rangeEnd) + " "
                newFile.write(input + " " + tag + "\n")
                tokenId += 1
            newFile.close()
    returnData = {}
    returnData["phraseRanges"] = uncertainRanges
    returnData["sentenceRanges"] = uncertainSentenceRange
    return returnData

pathToTrainingData = "C:/Users/Reema Bajwa/PycharmProjects/Project2/nlp_project2_uncertainty/train"
processedTokens = preProcessingOld(pathToTrainingData)
ct = CRFTagger()
ct.train(processedTokens,'model.crf.tagger')

publicTestFolder = "C:/Users/Reema Bajwa/PycharmProjects/Project2/nlp_project2_uncertainty/test-public"
testSentences = preprocessForTagging(publicTestFolder)

taggedSentences  = ct.tag_sents(testSentences)

#privateTestFolder = "C:/Users/Reema Bajwa/PycharmProjects/Project2/nlp_project2_uncertainty/test-private"
#publicResult = performTagging(ct, publicTestFolder, "C:/Users/Reema Bajwa/PycharmProjects/Project2/nlp_project2_uncertainty/baseline3Results")
#privateResult = performTagging(tagger, privateTestFolder, "C:/Users/Reema Bajwa/PycharmProjects/Project2/nlp_project2_uncertainty/baseline2ResultsPrivate")




    

