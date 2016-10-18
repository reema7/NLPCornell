import os

import fnmatch


import csv


import newCRFFile
class BaselineDictBuilder:
    map = {}
    def buildBaselineDict(self, pathToTrainingData):
        baselineDictionary = open("baselineDictionary.txt", "w")
        for root, dirs, files in os.walk(pathToTrainingData, topdown=True):
            for fileName in fnmatch.filter(files, '*.txt'):
                file = open(os.path.join(pathToTrainingData,fileName), 'r')
                for line in file:
                    if "CUE" in line:
                        baselineDictionary.write(line.split()[0] + "\n")

    def buildDictMap(self):
        dictionaryFile= open("baselineDictionary.txt", "r")
        for line in dictionaryFile:
           word = line.split()[0]
           self.map[word] = self.map.get(word, 1)


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

                join = tokens[0].decode('utf-8')+" "+tokens[1].decode('utf-8')+" "+tokens[2].decode('utf-8')
                lineList.append(join)
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
                if (len(lineTokens) == 2):
                    if lineTokens[0]=='.':
                        sentence.append(lineTokens[0].decode('utf-8')+" "+lineTokens[1].decode('utf-8'))
                        listSentences.append(sentence)
                        sentence = []
                    else:
                        sentence.append(lineTokens[0].decode('utf-8')+" "+lineTokens[1].decode('utf-8'))
    return listSentences

def convertCRFFormatFeatures(sent):
    features = []
    for line in sent:
        featureList = []
        tokens = line.split()
        if(len(tokens)==2):
            word = tokens[0]
            featureList.append("WORD" + word)
            featureList.append(tokens[1])
            features.append(featureList)
    return features


def convertCRFFormat(sent):
    labels = []
    features = []
    result = {}
    for line in sent:
        featureList = []
        tokens = line.split()
        labels.append(tokens[2])
        word = tokens[0]
        featureList.append("WORD"+word)
        featureList.append(tokens[1])
        features.append(featureList)
    result['labels'] = labels
    result['features'] = features
    return  result

def newTagging(tagger,testFolder):
    result = []
    testSentences = preprocessForTagging(testFolder)
    for tokens in testSentences:
        features = convertCRFFormatFeatures(tokens)
        labels = tagger.tag(features)
        taggedSentence = list(zip(tokens,labels))
        result.append(taggedSentence)
    return result

def performTagging(ct, testFolder, dictionaryBuilder):

    testSentences = preprocessForTagging(testFolder)
    taggedSentences = ct.tag_sents(testSentences)
    tag = "null"
    tokenId = 0
    sentenceId = 0
    uncertainRanges = ""
    foundInCurrentSentence = "false"
    uncertainSentenceRange = ""
    rangeStart = -1
    rangeEnd = -1
    for sentence in taggedSentences:
        for word in sentence:
            actualWord = word[0].split()[0].encode("utf-8")
            if(actualWord in dictionaryBuilder.map):
                foundInCurrentSentence = "true"
            if ". ." in word[0]:
                if foundInCurrentSentence is "true":
                    uncertainSentenceRange = uncertainSentenceRange + str(sentenceId) + " "
                    foundInCurrentSentence = "false"
                sentenceId = sentenceId + 1
            prevTag = tag
            tag = word[1]
            if("B-CUE" in tag):
                rangeStart = tokenId
                rangeEnd = tokenId
                foundInCurrentSentence = "true"
            elif("I-CUE" in tag):
                rangeEnd = tokenId
            elif("O-CUE" in tag):
                tag = "_"
                if ("CUE" in prevTag):
                    uncertainRanges = uncertainRanges + str(rangeStart) + "-" + str(rangeEnd) + " "
            tokenId += 1
    returnData = {}
    returnData["phraseRanges"] = uncertainRanges
    returnData["sentenceRanges"] = uncertainSentenceRange
    return returnData

#building baseline dictionary
dictionaryBuilder = BaselineDictBuilder()
dictionaryBuilder.buildDictMap()


pathToTrainingData = "/Users/shraddha/Documents/Semester 2/NLP/Project2/nlp_project2_uncertainty/train"
processedTokens = preProcessingOld(pathToTrainingData)
ct = newCRFFile.CRFTagger()
ct.train(processedTokens,'model.crf.tagger')

#ct = CRFTagger()
#ct.train(processedTokens,'model.crf.tagger')



publicTestFolder = "/Users/shraddha/Documents/Semester 2/NLP/Project2/nlp_project2_uncertainty/test-public"
privateTestFolder = "/Users/shraddha/Documents/Semester 2/NLP/Project2/nlp_project2_uncertainty/test-private"


#privateTestFolder = "C:/Users/Reema Bajwa/PycharmProjects/Project2/nlp_project2_uncertainty/test-private"

publicResult = performTagging(ct, publicTestFolder, dictionaryBuilder)
privateResult = performTagging(ct, privateTestFolder, dictionaryBuilder)
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


with open('predictionSentence.csv', 'w') as csvfile:
    fieldnames = ['Type', 'Indices']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'Type': 'SENTENCE-public', 'Indices': publicResult["sentenceRanges"]})
    writer.writerow({'Type': 'SENTENCE-private', 'Indices': privateResult["sentenceRanges"]})




