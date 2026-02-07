import json
import matplotlib.pyplot as plt
import numpy as np
import math



def clean_msg(txt):
    txt = txt.lower().strip()
    for c in txt:
        if c in """[]!.,"-!—@;':#$%^&*()+/?""":
            txt = txt.replace(c, " ")
    return txt



class SpamPredictor:
    """Loads a trained word-probability dictionary and evaluates spam/ham predictions
    on a test set across different cutoff thresholds"""
    def __init__(self, vocab_path, count_path, stops_path, testdata):
        self.vocab_path = vocab_path
        self.count_path = count_path
        self.stops_path = stops_path
        self.testdata = testdata
        self.dictionary = {}
        self.total_ham = 0
        self.total_spam = 0
        self.stoplist = set()


    def load_all(self):
        """Load model vocabulary, class counts, and stop words"""
        with open(self.vocab_path, "r") as f:
            self.dictionary = json.load(f)
        with open(self.count_path, "r") as f:
            parts = f.read().split()
            self.total_ham = int(parts[0])
            self.total_spam = int(parts[1])
        with open(self.stops_path, "r", encoding="unicode-escape") as f:
            self.stoplist = set(f.read().splitlines())

    def eval_with_cut(self, threshold):
        """Evaluate model on the test set using a cutoff threshold"""
        TN = FP = FN = TP = 0
        total_messages = self.total_spam + self.total_ham
    
    
        with open(self.testdata, "r", encoding="unicode-escape") as f:
            for line in f:
                if not line.strip():
                    continue
                flag = line[0]
                line = line[2:]
                words = clean_msg(line).split()
                relevant = set([w for w in words if w not in self.stoplist])
    
                log_spam = math.log(self.total_spam / total_messages)
                log_ham = math.log(self.total_ham / total_messages)
    
    
    
                for word in self.dictionary:
                    p_ham = self.dictionary[word][0]
                    p_spam = self.dictionary[word][1]
    
                    if word in relevant:
                        if p_ham > 0:
                            log_ham += math.log(p_ham)
                        if p_spam > 0:
                            log_spam += math.log(p_spam)
                    else:
                        if (1 - p_ham) > 0:
                            log_ham += math.log(1 - p_ham)
                        if (1 - p_spam) > 0:
                            log_spam += math.log(1 - p_spam)
                            
                            
                threshold = max(threshold, 1e-10) #preventing logic errors with "0"
                prediction = "1" if log_spam > log_ham + math.log(threshold) else "0"
    
                if flag == prediction:
                    if prediction == "1":
                        TP += 1
                    else:
                        TN += 1
                else:
                    if prediction == "1":
                        FP += 1
                    else:
                        FN += 1
    
        acc = (TP + TN) / (TP + TN + FP + FN)
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        
        TPR = TP / self.total_spam if self.total_spam > 0 else 0
        FPR = FP / self.total_ham if self.total_ham > 0 else 0
        TNR = TN / self.total_ham if self.total_ham > 0 else 0
        FNR = FN / self.total_spam if self.total_spam > 0 else 0
    
        return {
            "cut": threshold,
            "acc": round(acc, 3),
            "prec": round(prec, 3),
            "rec": round(rec, 3),
            "f1": round(f1, 3),
            "tpr": round(TPR, 3),
            "fpr": round(FPR, 3),
            "tnr": round(TNR, 3),
            "fnr": round(FNR, 3),}

    
    
    def assess_all(self):
        c_values = np.arange(0, 0.95, 0.05)
        outcomes = [self.eval_with_cut(c) for c in c_values]
    
        # Save to CSV (ratios)
        with open("metrics_output.csv", "w") as f:
            f.write("Cut,Acc,Prec,Rec,F1,TPR,FPR,TNR,FNR\n")
            for i in outcomes:
                f.write(f"{i['cut']},{i['acc']},{i['prec']},{i['rec']},{i['f1']},{i['tpr']},{i['fpr']},{i['tnr']},{i['fnr']}\n")
    
    
        cut = [x["cut"] for x in outcomes]
        
        accs = [x["acc"] for x in outcomes]
        pres = [x["prec"] for x in outcomes]
        recs = [x["rec"] for x in outcomes]
        f1s = [x["f1"] for x in outcomes]
        
        tprs = [x["tpr"] for x in outcomes]
        fprs = [x["fpr"] for x in outcomes]
        tnrs = [x["tnr"] for x in outcomes]
        fnrs = [x["fnr"] for x in outcomes]
    
        plt.figure()
        plt.plot(cut, accs, label="Acc", marker='o')
        plt.plot(cut, pres, label="Prec", marker='o')
        plt.plot(cut, recs, label="Rec", marker='o')
        plt.plot(cut, f1s, label="F1", marker='o')
        plt.xlabel("Cutoff (for TP)")
        plt.ylabel("Metric Values")
        plt.title("Cutoff vs Various Metrics")
        plt.legend()
        plt.grid()
        plt.savefig("metrics_plot.png")
        plt.show()
    
        plt.figure()
        plt.plot(cut, tprs, label="TP", marker='o')
        plt.plot(cut, fprs, label="FP", marker='o')
        plt.plot(cut, tnrs, label="TN", marker='o')
        plt.plot(cut, fnrs, label="FN", marker='o')
        plt.xlabel("Cutoff (for TP)")
        plt.ylabel("Category Rate (ratio)")
        plt.title("Cutoff vs Categories in Confusion Matrix")
        plt.legend()
        plt.grid()
        plt.savefig("rates_plot.png")
        plt.show()





vocab_fp = input("Trained dictionary file path (Model/Vocab.json): ")
count_fp = input("Ham/Spam counts file path (Model/Counts.txt): ")
stops_fp = input("Stopwords file path (Data/StopWords.txt): ")
test_fp = input("Test data file path (Data/SpamTest.txt): ")


test_model = SpamPredictor(vocab_fp, count_fp, stops_fp, test_fp)
test_model.load_all()
test_model.assess_all()