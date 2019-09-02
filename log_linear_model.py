# -*- coding: utf-8 -*-

import datetime
from math import exp

class sentence:
    def __init__(self):
        self.word = []
        self.tag = []
        self.wordchars = []

class dataset:
    def __init__(self):
        self.sentences = []
        self.name = ""
        self.total_word_count = 0
    
    def open_file(self, inputfile):
        self.inputfile = open(inputfile, mode = 'r', encoding='utf-8')
        self.name = inputfile.split('.')[0]

    def close_file(self):
        self.inputfile.close()

    def read_data(self, sentenceLen):
        wordCount = 0
        sentenceCount = 0
        sen = sentence()
        for s in self.inputfile:
            if(s == '\r\n' or s == '\n'):
                sentenceCount += 1
                self.sentences.append(sen)
                sen = sentence()
                if(sentenceLen !=-1 and sentenceCount >= sentenceLen):
                    break
                continue
            list_s = s.split('\t')
            str_word = list_s[1]#.decode('utf-8')
            str_tag = list_s[3]
            list_wordchars = list(str_word)
            sen.word.append(str_word)
            sen.tag.append(str_tag)
            sen.wordchars.append(list_wordchars)
            wordCount += 1
        self.total_word_count = wordCount
        print(self.name + ".conll contains " + str(len(self.sentences)) + " sentences")
        print(self.name + ".conll contains " + str(self.total_word_count) + " words")
        
class log_linear_model:
    def __init__(self):
        self.feature_dict = {}
        self.feature_keys = []
        self.feature_values = []
        self.feature_length = 0
        self.tag_dict = {}
        self.tag_length = 0
        self.w = []
        self.g = []
        self.update_index = []
        self.g_update_id = dict()
        self.train = dataset()
        self.dev = dataset()

        self.train.open_file("train.conll")
        self.train.read_data(-1)
        self.train.close_file()

        self.dev.open_file("dev.conll")
        self.dev.read_data(-1)
        self.dev.close_file()
        
    def create_feature(self, sen, pos):
        f = []
        wi = sen.word[pos]
        f.append("02:" + wi)
        
        if(pos == 0):
            wim1 = "$$"
        else:
            wim1 = sen.word[pos - 1]
        f.append("03:" + wim1)
        
        len_sen = len(sen.word)
        if(pos == len_sen - 1):
            wip1 = "##"
        else:
            wip1 = sen.word[pos + 1]
        f.append("04:" + wip1)
        
        cim1m1 = wim1[-1]
        f.append("05:" + wi + cim1m1)
        
        cip10 = wip1[0]
        f.append("06:" + wi + cip10)
        
        ci0 = wi[0]
        f.append("07:" + ci0)
        
        cim1 = wi[-1]
        f.append("08:" + cim1)
        
        len_str = len(wi)
        for k in range(1, len_str - 1):
            cik = wi[k]
            f.append("09:" + cik)
            f.append("10:" + ci0 + cik)
            f.append("11:" + cim1 + cik)
            
        if(len_str == 1):
            f.append("12:" + wi + cim1m1 + cip10)
            
        for k in range(len_str - 1):
            cik = wi[k]
            cikp1 = wi[k + 1]
            if(cik == cikp1):
                f.append("13:" + cik + "consecutive")
        
        for k in range(1, len_str):
            if k > 4:
                break
            f.append("14:" + wi[:k])
            f.append("15:" + wi[-k:])
        #print(pos, f)
        return f
            
    def create_feature_space(self):
        for sen in self.train.sentences:
            for pos in range(len(sen.word)):
                f = self.create_feature(sen, pos)
                for feature in f:
                    if feature not in self.feature_dict:
                        self.feature_dict[feature] = len(self.feature_dict)
                
                tag = sen.tag[pos]
                if tag not in self.tag_dict:
                    self.tag_dict[tag] = len(self.tag_dict)
                    
        self.feature_length = len(self.feature_dict)
        self.tag_length = len(self.tag_dict)
        self.feature_keys = list(self.feature_dict.keys())
        self.feature_values = list(self.feature_dict.values())
        
        self.g = [0]*(self.feature_length * self.tag_length)
        self.w = [0]*(self.feature_length * self.tag_length)
        
        print("the total number of features is " + str(self.feature_length))
        print("the total number of tags is " + str(self.tag_length))
        
    def get_feature_id(self, fv):
        fv_id = []
        for f in fv:
            if f in self.feature_dict:
                fv_id.append(self.feature_dict[f])
        return fv_id
        
        
    def dot(self, fv_id, offset):
        score = 0
        for f_id in fv_id:
            score += self.w[f_id + offset]
        return score
                
    def max_tag(self, sen, pos):
        max_score = -1
        max_tag = ""
        fv = self.create_feature(sen, pos)
        fv_id = self.get_feature_id(fv)
        for t in self.tag_dict:
            offset = self.tag_dict[t]*self.feature_length
            score = self.dot(fv_id, offset)
            if(score > max_score):
                max_score = score
                max_tag = t
        #print(max_tag)
        return max_tag
    
    def update_w(self, eta):
        #for i in self.update_index: # 为什么用dict程序跑的很快，list就很慢
        for i in self.g_update_id:
            self.w[i] -= eta*self.w[i]
            self.w[i] += eta*self.g[i]
        
            
    def update_g(self, correct_tag, sen, pos):
        fv = self.create_feature(sen, pos)
        fv_id = self.get_feature_id(fv)
        correct_tag_id = self.tag_dict[correct_tag]
        offset = correct_tag_id*self.feature_length
        for f_id in fv_id:
            self.g[offset + f_id] += 1
        
        # 分母
        denominator = 0
        for t in self.tag_dict:
            offset = self.tag_dict[t]*self.feature_length
            denominator += exp(self.dot(fv_id, offset))
        
        for t in self.tag_dict:
            offset = self.tag_dict[t]*self.feature_length
            prop = exp(self.dot(fv_id, offset)) / denominator
            for f_id in fv_id:
                self.g[offset + f_id] -= prop
                """if (offset + f_id) in self.update_index:
                    pass
                else:
                    self.update_index.append(offset + f_id)"""
                self.g_update_id[offset + f_id] = 0
            
    def evaluate(self, dataset):
        count = 0
        total_count = 0
        for s in dataset.sentences:
            for pos in range(len(s.word)):
                max_tag = self.max_tag(s, pos)
                if(max_tag == s.tag[pos]):
                    count += 1
                total_count += 1
        print(dataset.name +".conll precision:" + str(count / total_count))
        return count, total_count, count / total_count
        
    def online_training(self, max_epochs):
        max_train_precision = 0.0
        max_dev_precision = 0.0
        max_iterator = 0
        b = 0
        eta = 0.5
        print("*******start iteration************")
        for epoch in range(max_epochs):
            print("epoch:" + str(epoch))
            count = 0
            for sen in self.train.sentences:
                for pos in range(len(sen.word)):
                    correct_tag = sen.tag[pos]
                    self.update_g(correct_tag, sen, pos)
                    b += 1
                    count += 1
                    if(b % 50 == 0):
                        self.update_w(eta)
                        b = 0
                        eta = max(eta*0.999, 0.00001)
                        self.g = [0]*(self.feature_length * self.tag_length)
                        #self.update_index = []
                        self.g_update_id = {}
            
            count_train, total_count_train, pre_train = self.evaluate(self.train)
            count_dev, total_count_dev, pre_dev = self.evaluate(self.dev)
            
            if(pre_train > max_train_precision):
                max_train_precision = pre_train
            if(pre_dev > max_dev_precision):
                max_dev_precision = pre_dev
                max_iterator = epoch
            
        print("**********stop iteration**************")
        print("train.conll max precision:" + str(max_train_precision))
        print("dev.conll max precision:" + str(max_dev_precision) + " in epoch " + str(max_iterator))
        
if __name__ == "__main__":
    starttime = datetime.datetime.now()
    llm = log_linear_model()
    llm.create_feature_space()
    max_epochs = 25
    llm.online_training(max_epochs)
    endtime = datetime.datetime.now()
    print("executing time is "+str((endtime-starttime).seconds)+" s")