import re 
import collections
import numpy as np 
import random

class Helper(object):
    """This class is responsible for reading documents and converintg it into a list of words:frequencies"""
    def __init__(self):        
        return   

    def PreProcess(self,Data_file):
        with open(Data_file,'r') as content_file:
                    content = content_file.read()
    
        docs = re.split("\n",content)
        vocab = list(set(re.split("\s+",re.sub("\n"," ",content))))

        #Check the number of documents is the correct one
        nbDocs = docs[0];
        if(int(nbDocs) != (len(docs)-1) ):
            print("The number of documents provided (%s) is different from the checksum (%s)" % (len(docs)-1, nbDocs) )

        docs = docs[1:len(docs)]
        
        docs_processed = []
        for d in docs:
            words = re.split("\s+",d)
            frequencies = collections.Counter(words)
            word_frequencies = {vocab.index(w):count for w,count in frequencies.items()}
            docs_processed.append(word_frequencies)

        #return results
        return {"documents":docs_processed,"vocab":vocab}
    
    
    # Write the content of a processed data into a file
      
    def writeProcessedData(self,data,vocab,filename):
           
        if(len(data) ==0):
            return 0

        f = open(filename,"w")
        for doc in data:
            content = ["%s:%s" % (w,count) for w,count in doc.items()]
            f.write("%s\n" % " ".join(content))        
        f.close()

        f1 = open("vocab.txt",'w')
        f1.write(" ".join(vocab))
        f1.close()

        return 0


    
    # Read data from an already processed file
    
    def readProcssed(self,filename,vocab_file):
        Data = []

        with open(filename,"r") as content_file:
            content = content_file.read()

        lines = re.split("\n",content)
        #Read formated data
        for doc in lines:
            words = re.split("\s+",doc)
            word_freq = {re.split(":",w)[0]:int(re.split(":",w)[1]) for w in words}
            Data.append(word_freq)
        
        #Read vocabulary
        with open(vocab_file,"r") as words_list:
            vocab = words_list.read()

        vocab = re.split("\s+",vocab)

        return {"documents":Data,"vocab":vocab}

    
    #  E step to update P(d|z)
    
    def update_p_dz(self,p_wz,p_dz,p_z,docs,nbTopic):

        new_p_dz = np.zeros((len(docs),nbTopic))
        for d in  range(len(docs)):   
            items = list(docs[d].items())  
            #We shuffle to not keep always the same sampling order
            random.shuffle(items)                               
            for itm in items:
                [w,ft] = list(itm)
                #Calculate p(z|d,w)
                p_z_wd = p_z * p_dz[d,:] * p_wz[w,:]
                p_z_wd = p_z_wd / sum(p_z_wd)
                # n(d,w) * p(z|d,w)
                new_p_dz[d,:] += ft * p_z_wd

        row_sum = new_p_dz.sum(axis=0) 
        new_p_dz = new_p_dz / row_sum[np.newaxis, :]
        return new_p_dz
      
    
    # E-Step to update p(w|z)
    
    def update_p_wz(self,p_wz,p_dz,p_z,docs,nbTopic,nbWord):
       
       new_p_dw = np.zeros((nbWord,nbTopic))

       for d in  range(len(docs)):                                    
            items = list(docs[d].items())              
            random.shuffle(items)                               
            for itm in items:
                [w,ft] = list(itm)
                #Calculate p(z|d,w)
                p_z_wd = p_z * p_dz[d,:] * p_wz[w,:]
                p_z_wd = p_z_wd / sum(p_z_wd)
                # sum_d=0^1 [ n(d,w) * p(z|d,w) ]
                new_p_dw[w,:] += ft * p_z_wd
       
       row_sum = new_p_dw.sum(axis=0) 
       new_p_dw = new_p_dw / row_sum[np.newaxis, :]

       return new_p_dw
   
    
    # E-Step to update p(z)
    def update_p_z(self,p_wz,p_dz,p_z,docs,nbTopic):
        
       p_z_new = np.zeros(nbTopic)
       total_count = 0
       for d in  range(len(docs)):                                    
            items = list(docs[d].items())  
            #We shuffle to not keep always the same sampling order
            random.shuffle(items)                               
            for itm in items:
              [w,ft] = list(itm)               
              #Calculate p(z|d,w)
              p_z_wd = p_z * p_dz[d,:] * p_wz[w,:]
              p_z_wd = p_z_wd / sum(p_z_wd)
              # sum_d sum_w n(d,w) * p(z|d,w)
              total_count += ft
              p_z_new +=  ft * p_z_wd   
                    
       p_z_new = p_z_new / total_count
       return p_z_new

    def log_likelihood(self,p_wz,p_dz,p_z,docs):
      L = 0;
      for d in  range(len(docs)):                                    
            for w,ft in docs[d].items():                
              L += ft * np.log10( sum(p_z * p_dz[d,:] * p_wz[w,:]))
      return L

    def GetTopword(self,p_wz,vocab, N = 10):
      topwords = []
      probabilities = []
      for z in range(len(p_wz[0,:])):
          words = np.argsort(p_wz[:,z])
          probs = np.sort(p_wz[:,z])[-10:]          
          words = [ vocab[words[k]] for k in range(-N,0) ]         
          #words = vocab[words[-10:-1]]
          topwords.append(words)
          probabilities.append(probs)

      return {"Top":topwords,"Probs":probabilities }

    def writeResult(p_wz,p_dz,p_z,vocab):
      return 0
