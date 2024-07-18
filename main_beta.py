import re
import os
import random
import time
import clip
from statistics import mode

from PIL import Image
import numpy as np
import pandas
import torch
import torch.nn as nn
import torch.utils
import torchvision
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import re

def process_text(text):
    # This code is based on the code written by Qing Li for VizWiz Python API available at the following link: 
    # (https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/eval.py).
    text = text.lower()
    num_word_to_digit = {
        'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3',
        'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8',
        'nine': '9', 'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)
    text = re.sub(r"(?<!\d)\.(?!\d)", '', text)
    text = re.sub(r'\b(a|an|the)\b', '', text)
    contractions = {
        "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
        "couldn'tve": "couldn’t’ve", "couldnt’ve": "couldn’t’ve", "didnt": "didn’t", "doesnt": "doesn’t", "dont": "don’t", "hadnt": "hadn’t", \
        "hadnt’ve": "hadn’t’ve", "hadn'tve": "hadn’t’ve", "hasnt": "hasn’t", "havent": "haven’t", "hed": "he’d", "hed’ve": "he’d’ve", \
        "he’dve": "he’d’ve", "hes": "he’s", "howd": "how’d", "howll": "how’ll", "hows": "how’s", "Id’ve": "I’d’ve", "I’dve": "I’d’ve", \
        "Im": "I’m", "Ive": "I’ve", "isnt": "isn’t", "itd": "it’d", "itd’ve": "it’d’ve", "it’dve": "it’d’ve", "itll": "it’ll", "let’s": "let’s", \
        "maam": "ma’am", "mightnt": "mightn’t", "mightnt’ve": "mightn’t’ve", "mightn’tve": "mightn’t’ve", "mightve": "might’ve", \
        "mustnt": "mustn’t", "mustve": "must’ve", "neednt": "needn’t", "notve": "not’ve", "oclock": "o’clock", "oughtnt": "oughtn’t", \
        "ow’s’at": "’ow’s’at", "’ows’at": "’ow’s’at", "’ow’sat": "’ow’s’at", "shant": "shan’t", "shed’ve": "she’d’ve", "she’dve": "she’d’ve", \
        "she’s": "she’s", "shouldve": "should’ve", "shouldnt": "shouldn’t", "shouldnt’ve": "shouldn’t’ve", "shouldn’tve": "shouldn’t’ve", \
        "somebody’d": "somebodyd", "somebodyd’ve": "somebody’d’ve", "somebody’dve": "somebody’d’ve", "somebodyll": "somebody’ll", \
        "somebodys": "somebody’s", "someoned": "someone’d", "someoned’ve": "someone’d’ve", "someone’dve": "someone’d’ve", \
        "someonell": "someone’ll", "someones": "someone’s", "somethingd": "something’d", "somethingd’ve": "something’d’ve", \
        "something’dve": "something’d’ve", "somethingll": "something’ll", "thats": "that’s", "thered": "there’d", "thered’ve": "there’d’ve", \
        "there’dve": "there’d’ve", "therere": "there’re", "theres": "there’s", "theyd": "they’d", "theyd’ve": "they’d’ve", \
        "they’dve": "they’d’ve", "theyll": "they’ll", "theyre": "they’re", "theyve": "they’ve", "twas": "’twas", "wasnt": "wasn’t", \
        "wed’ve": "we’d’ve", "we’dve": "we’d’ve", "weve": "we've", "werent": "weren’t", "whatll": "what’ll", "whatre": "what’re", \
        "whats": "what’s", "whatve": "what’ve", "whens": "when’s", "whered": "where’d", "wheres": "where's", "whereve": "where’ve", \
        "whod": "who’d", "whod’ve": "who’d’ve", "who’dve": "who’d’ve", "wholl": "who’ll", "whos": "who’s", "whove": "who've", "whyll": "why’ll", \
        "whyre": "why’re", "whys": "why’s", "wont": "won’t", "wouldve": "would’ve", "wouldnt": "wouldn’t", "wouldnt’ve": "wouldn’t’ve", \
        "wouldn’tve": "wouldn’t’ve", "yall": "y’all", "yall’ll": "y’all’ll", "y’allll": "y’all’ll", "yall’d’ve": "y’all’d’ve", \
        "y’alld’ve": "y’all’d’ve", "y’all’dve": "y’all’d’ve", "youd": "you’d", "youd’ve": "you’d’ve", "you’dve": "you’d’ve", \
        "youll": "you’ll", "youre": "you’re", "youve": "you’ve"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    punctuations = r"[;\/\[\]\"{}()=+\\_><@`,?!]"
    text = re.sub(punctuations, ' ', text)
    text = re.sub(r'\s+,', ',', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, processor=None, answer=True):
        self.processor = processor  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pandas.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.addition_answer = pandas.read_csv('./data/unique_to_set.csv')
        # This is the class_mapping file related to VizWiz, which serves as the complement table.
        # (https://huggingface.co/spaces/CVPR/VizWiz-CLIP-VQA/raw/main/data/annotations/class_mapping.csv).
        self.answer = answer
        #self.df = self.df.head(100)
        
        self.question2idx = {}
        self.answer2idx = {}
        self.idx2question = {}
        self.idx2answer = {}
        
        
        for question in self.df["question"]:
            question = process_text(question)
            words = question.split(" ")
            for word in words:
                if word not in self.question2idx:
                    self.question2idx[word] = len(self.question2idx)
        self.idx2question = {v: k for k, v in self.question2idx.items()}  # 逆変換用の辞書(question)

        
        if self.answer:
            # 回答に含まれる単語を辞書に追加
            # answer2idx 中存储了答案和idx的映射，每一个答案对应一个index
            # answer2idx, key: answer(word or words); value: idx
            for answers in self.df["answers"]:
                for answer in answers:
                    word = answer["answer"]
                    word = process_text(word)
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)

            for answer in self.addition_answer['answer']:
                word = process_text(answer)
                if word not in self.answer2idx:
                    self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)
    
    def update_dict(self, dataset):
        self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer
    
    def __getitem__(self, idx):
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.processor(image)
        
        question = np.zeros(len(self.idx2question) + 1)
        question_words = self.df["question"][idx].split(" ")
        for word in question_words:
            try:
                question[self.question2idx[word]] = 1  # one-hot表現に変換
            except KeyError:
                question[-1] = 1  # 未知語
        #print(image.shape, question.shape)
        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            answer_confidence = [{'yes': '1', 'maybe': '0.6', 'no': '0.1'}.get(answer["answer_confidence"], '0') for answer in self.df["answers"][idx] ]
    
            mode_answer_idx = mode(answers)
            answer_weight = 0
            count_mode_answer = answers.count(mode_answer_idx)
            if count_mode_answer >= 3:
                answer_weight = sum((conf if ans == mode_answer_idx else -conf for ans, conf in zip(answers, answer_confidence))) / len(answers)

            return image, torch.Tensor(question), torch.Tensor(answers), int(mode_answer_idx), float(answer_weight)
        else:
            return image, torch.Tensor(question)

    def __len__(self):
        return len(self.df)

class VQAModel(nn.Module):
    def __init__(self, clip_model, vocab_size, num_classes):
        super(VQAModel, self).__init__()
        self.clip_model = clip_model
        #self.text_encoder = nn.Linear(vocab_size, 512)
        self.text_encoder = nn.LSTM(input_size=vocab_size, hidden_size=512, batch_first=True)
        
        self.dropout = nn.Dropout(0.6)
        #self.img_fc1 = nn.Linear(1024, 512)
        #self.fc2 = nn.Linear(512, num_classes)
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
        #self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, images, questions):
        images = images.long()

        image_features = self.clip_model.encode_image(images).float()
        #image_features = self.img_fc1(image_features)
        image_features = self.dropout(image_features)
        #text_features = self.clip_model.encode_text(questions).float()
        text_features, (hidden, cell) = self.text_encoder(questions)
        combined_features, _ = self.attention(image_features.unsqueeze(1), text_features, text_features)
        combined_features = combined_features.squeeze(1)
        combined = torch.cat([combined_features, text_features[:, -1, :]], dim=1)
        outputs = self.fc(combined)
        return outputs

def ensure_make_dir_os(path):
    if not os.path.exists(path):
        os.makedirs(path)

def collate_fn_train(batch):
    images, questions, answers, mode_answer_idxs = zip(*batch)
    images = torch.stack(images)
    questions = torch.nn.utils.rnn.pad_sequence(questions, batch_first=True, padding_value=0)
    answers = torch.stack(answers)
    mode_answer_idxs = torch.tensor(mode_answer_idxs)
    #print(f'train: {questions.shape}')
    return images, questions, answers, mode_answer_idxs

def collate_fn_test(batch):
    images, questions = zip(*batch)
    images = torch.stack(images)
    questions = torch.nn.utils.rnn.pad_sequence(questions, batch_first=True, padding_value=0)
    #print(f'test: {questions.shape}')
    return images, questions

def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)

def train(model, dataloader, optimizer,criterion, device):
    model.train()
    
    total_loss = 0
    total_acc = 0
    simple_acc = 0
    log_softmax = nn.LogSoftmax(dim=1).to(device)
    
    start = time.time()
    for image, question, answers, mode_answer, answer_weight in dataloader:
        image, question, answers, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device), answer_weight.to(device)
        #print(question.shape)
        #print(image.shape, question.shape, answer.shape)
        optimizer.zero_grad()
        pred = model(image, question)
        #print(pred)
        #print(pred.shape, mode_answer.shape)
        
        if pred.shape[0] == 1:
            try:
                loss = criterion(pred, mode_answer)
            except:
                print(f'pred: {pred.shape()}')
                continue
        else:
            loss = criterion(pred, mode_answer.squeeze())
        
        #a Soft Cross-Entropy loss : 
        # a weighted average of the negative log-probabilities of each unique ground-truth answer
        #nll = -log_softmax(pred)
        #gather = nll.gather(1, answers)
        #loss = gather.sum(dim=1).mean() / 10
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)
        #print(f'total_acc:{total_acc}')
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy
        
    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def eval(model, dataloader, optimizer, criterion, device):
    model.eval()
    
    total_loss = 0
    total_acc = 0
    simple_acc = 0
    
    log_softmax = nn.LogSoftmax(dim=1).to(device)
    
    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, answers, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)
        
        pred = model(image, question)
        if pred.shape[0] == 1:
            try:
                loss = criterion(pred, mode_answer)
            except:
                print(f'pred: {pred.shape()}')
                continue
        else:
            loss = criterion(pred, mode_answer.squeeze())
        #nll = -log_softmax(pred)
        #loss = (nll * answers / 10).sum(dim=1).mean()
        
        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()
    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

import hashlib

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # load CLIP model and processor
    #processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    #clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    clip_model, processor = clip.load('ViT-B/16')
    #print(processor, clip_model)
    
    train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", processor=processor)
    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", processor=processor, answer=False)
    test_dataset.update_dict(train_dataset)
    num_classes = len(train_dataset.answer2idx)
    model = VQAModel(clip_model, len(train_dataset.question2idx)+1, num_classes).float().to(device)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    num_epoch = 5
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    
    for epoch in range(num_epoch):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        
        print(f"====================Train===================="
              f"【{epoch + 1}/{num_epoch}】\n"
            f"train time: {train_time:.2f} [s]\n"
            f"train loss: {train_loss:.4f}\n"
            f"train acc: {train_acc:.4f}\n"
            f"train simple acc: {train_simple_acc:.4f}", flush=True)
        #ensure_make_dir_os(f"./rn50_linear/rn50_embed")
        torch.save(model.state_dict, f'./models/clip_vit16p_3/clip_epoch{epoch+1}.pth')
        
        # output test
        model.eval()
        submission = []

        for images, questions in test_loader:
            images, questions = images.to(device), questions.to(device)
            preds = model(images, questions)
            preds = torch.argmax(preds, dim=1)
            preds = preds.tolist()
            submission.extend(preds)
            
        submission = [test_dataset.idx2answer[id] for id in submission]
        submission = np.array(submission)
        #torch.save(model.state_dict(), "model_3.pth")
        #ensure_make_dir_os(f"./rn50_linear/rn50_embed")
        np.save(f"./submission/clip_vit16p_3/submission_epoch{epoch+1}.npy", submission)
        print('Finish submission', flush=True)
    
if __name__ == "__main__":
    main()
