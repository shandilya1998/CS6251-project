import pandas as pd
from tqdm import tqdm

df = pd.read_csv('raw/emotion_stimulus_dataset_parsed.csv')

data = []

for i in tqdm(range(len(df))):
    emotion = df['emotion'][i]
    if emotion == 'happy':
        data.append([df['sentences'][i], 4])
    elif emotion == 'sad':
        data.append([df['sentences'][i], 5])
    elif emotion == 'surprise':
        data.append([df['sentences'][i], 6])
    elif emotion == 'disgust':
        data.append([df['sentences'][i], 2])
    elif emotion == 'anger':
        data.append([df['sentences'][i], 1])
    elif emotion == 'fear':
        data.append([df['sentences'][i], 3])
    elif emotion == 'shame':
        data.append([df['sentences'][i], 7])
    else:
        data.append([df['sentences'][i], 0])

df = pd.read_csv('raw/daily_dialog_dataset_parsed.csv')

for i in tqdm(range(len(df))):
    data.append([df['sentences'][i], df['emotion'][i]])

df = pd.read_csv('raw/parsed_crowdflower_dataset.csv')

for i in tqdm(range(len(df))):
    emotion = df['sentiment'][i]
    sentence = df['parsed tweets'][i]
    if emotion == 'happiness':
        data.append([sentence, 4])
    elif emotion == 'sadness':
        data.append([sentence, 5])
    elif emotion == 'surprise':
        data.append([sentence, 6])
    elif emotion == 'anger':
        data.append([sentence, 1])
    elif emotion == 'empty' or emotion == 'neutral':
        data.append([sentence, 0]) 
    elif emotion == 'enthusiasm':
        data.append([sentence, 8]) 
    elif emotion == 'worry':
        data.append([sentence, 9])
    elif emotion == 'love':
        data.append([sentence, 10])
    elif emotion == 'fun':
        data.append([sentence, 11])
    elif emotion == 'hate':
        data.append([sentence, 12])
    elif emotion == 'boredom':
        data.append([sentence, 13])
    elif emotion == 'relief':
        data.append([sentence, 14])

df = pd.DataFrame(data, columns = ['sentence', 'emotion'])
df.to_csv('dataset.csv', index = False)
