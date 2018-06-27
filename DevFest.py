import argparse
from google.cloud import videointelligence

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

import numpy as np
import matplotlib.pyplot as plt

##-----------------------------------------------

#transcribes a vine split audio file (flac) into text using
#Google Cloud's speech API

def transcribe(x):
  client = speech.SpeechClient()
  audio = types.RecognitionAudio(uri="gs://devfest-gjm/Vine_Audio_flac/file"+str(x)+".flac")
  config = types.RecognitionConfig(encoding=enums.RecognitionConfig.AudioEncoding.FLAC,sample_rate_hertz=44100,language_code='en-US')
  response = client.recognize(config, audio)
  for result in response.results:
    return ('{}'.format(result.alternatives[0].transcript))

#returns a list of labels from vine videos using the
#Google Cloud videointelligence API

def analyze_labels(index):
  dict={}
  path="gs://devfest-gjm/Vines/"+str(index)+".mp4"
  video_client = videointelligence.VideoIntelligenceServiceClient()
  features = [videointelligence.enums.Feature.LABEL_DETECTION]
  operation = video_client.annotate_video(path, features=features)
  result = operation.result(timeout=90)
  segment_labels = result.annotation_results[0].segment_label_annotations
  for i, segment_label in enumerate(segment_labels):
    for i, segment in enumerate(segment_label.segments):
      dict[str(segment_label.entity.description)]=segment.confidence
  return dict

#returns the sentiment of text using the
#Google Cloud Natural Language API

def sentiment_analysis(text):
  if text == None:
    return 0
  client = language.LanguageServiceClient()
  document = types.Document(content=text,type=enums.Document.Type.PLAIN_TEXT)
  sentiment = client.analyze_sentiment(document).document_sentiment
  return sentiment.score


#creates list of transcripts of vines from google cloud storage
transcripts = []
for x in range(1,21):
  transcripts.append(transcribe(x))


######creates list of dictionaries corresponding to each video, which labels as keys
#######and confidence as the return value

#videos=[]
#for i in range(1,20):
#  videos.append(analyze_labels(i))

#Saved output
videos=[{'pet': 0.7467221021652222, 'dog': 0.6626756191253662, 'animal': 0.8771911263465881}, {'town': 0.727906346321106, 'car': 0.788735032081604, 'motor vehicle': 0.4480705261230469, 'street': 0.846523106098175, 'vehicle': 0.7546262145042419, 'public space': 0.6921080350875854, 'mode of transport': 0.40687695145606995, 'road': 0.6545488238334656}, {'selfie': 0.48593300580978394}, {'car': 0.46713146567344666, 'selfie': 0.5225968360900879, 'eyewear': 0.48952051997184753, 'glasses': 0.5547260046005249}, {'car': 0.533627450466156, 'motor vehicle': 0.4483453631401062, 'vehicle': 0.603063702583313}, {'car': 0.7672041654586792, 'land vehicle': 0.42129677534103394, 'motor vehicle': 0.49435415863990784, 'vehicle': 0.7462655901908875, 'convertible': 0.4442906677722931, 'sports car': 0.6067716479301453}, {'selfie': 0.6454963088035583, 'hat': 0.41284969449043274, 'headgear': 0.4021816551685333}, {'selfie': 0.46510806679725647}, {'plant': 0.48217031359672546}, {'aisle': 0.7675832509994507, 'shopping': 0.6225206851959229, 'supermarket': 0.9079504609107971, 'convenience store': 0.9152846336364746, 'grocer': 0.5464529991149902, 'inventory': 0.4804534316062927, 'convenience food': 0.6518909931182861, 'retail': 0.9329389929771423, 'grocery store': 0.9644016027450562}, {'selfie': 0.5571514368057251, 'video blog': 0.629494309425354}, {'emotion': 0.4327436089515686, 'smile': 0.6565071940422058, 'selfie': 0.7422916889190674, 'eyewear': 0.4419821500778198, 'facial expression': 0.46732452511787415}, {}, {'professional': 0.44678327441215515, 'businessperson': 0.7361025810241699, 'official': 0.4019990861415863, 'necktie': 0.4175505042076111, 'suit': 0.6633758544921875}, {'selfie': 0.5938089489936829, 'vine': 0.4925195276737213}, {}, {'long hair': 0.572756826877594}, {}, {'selfie': 0.5833327174186707}]


vid_sentiment=[]
for vid in videos:
  total_confidence=0
  for label in vid:
    total_confidence+=vid[label]
  results=0
  for label in vid:
    results+=sentiment_analysis(label)*vid[label]/total_confidence
  vid_sentiment.append(results)

#sentiment analysis of vine transcripts
transcript_sentiment=[]
for trn in transcripts:
  transcript_sentiment.append(sentiment_analysis(trn))

#plots video/transcript sentiment in scatter plot
transcript_sentiment = numpy.array(transcript_sentiment)
vid_sentiment = numpy.array(vid_sentiment)
plt.switch_backend('Agg')
plt.scatter(transcript_sentiment, vid_sentiment)
plt.savefig("devfest_scatter")
