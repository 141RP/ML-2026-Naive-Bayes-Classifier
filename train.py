import json
import os

def clean(t):
    t = t.lower().strip()
    for ch in t:
        if ch in """[]!.,"-!—@;':#$%^&*()+/?""":
            t = t.replace(ch, " ")
    return t



def build_words(words):
    cnt = {}
    for word in words:
        if word in cnt:
            cnt[word] += 1
        else:
            cnt[word] = 1
    return cnt



class ProbabilityTrainer:
    """Trains a very simple Naive-Bayes-style token probability model for spam vs ham.+"""

    def __init__(self, stoplist, infile):
        self.sfile = stoplist
        self.tfile = infile
        self.filtered = set()
        self.word_count = {}
        self.final_probs = {}
        self.ham_ct = 0
        self.spam_ct = 0
        self.k = 1


    def read_stoplist(self):
        """Load stop words into a set for fast checks"""
        with open(self.sfile, "r", encoding="unicode-escape") as f:
            self.filtered = set(f.read().splitlines())


    def compute_probs(self):
        """raw counts into conditional probabilities"""
        for token in self.word_count:
            ham_occ = self.word_count[token][0]
            spam_occ = self.word_count[token][1]
            self.word_count[token][0] = (ham_occ + self.k) / (2 * self.k + self.ham_ct)
            self.word_count[token][1] = (spam_occ + self.k) / (2 * self.k + self.spam_ct)
        self.final_probs = self.word_count


    def run_training(self):
        """Read training file, update counts for each token by class, then compute smoothed probabilities"""
        with open(self.tfile, "r", encoding="unicode-escape") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tag = line[0]
                subj = line[2:]
                terms = clean(subj).split()
                filtered_terms = [w for w in terms if w not in self.filtered]
    
                
                if tag == "1":
                    self.spam_ct += 1
                else:
                    self.ham_ct += 1
    
    
                for w in filtered_terms:
                    if w not in self.word_count:
                        self.word_count[w] = [0, 0]
                    if tag == "1":
                        self.word_count[w][1] += 1
                    else:
                        self.word_count[w][0] += 1
        self.compute_probs()



    def export_model(self):
        """Write probability table and class totals"""
        os.makedirs("Model", exist_ok=True)
        with open("Model/Vocab.json", "w") as f:
            json.dump(self.final_probs, f, indent=4)
        with open("Model/Counts.txt", "w") as f:
            f.write(f"{self.ham_ct} {self.spam_ct}")



stopfile = input("Stop words file (Data/StopWords.txt): ")
trainfile = input("Training data file (Data/SpamTrain.txt): ")

trainer = ProbabilityTrainer(stopfile, trainfile)

trainer.read_stoplist()
trainer.run_training()
trainer.export_model()
