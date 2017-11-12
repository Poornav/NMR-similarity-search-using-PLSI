import collections
import numpy as np
from Helper import *
import matplotlib.pyplot as plt 
import sys , getopt




# This function plots the final results

def plot_convergence(likelihood_vals, Z, topwords):
    fig = plt.figure()
    plt.subplot(211)
    plt.plot(range(1,len(likelihood_vals)+1), likelihood_vals, 'bo', linewidth=2)
    plt.xlabel("iteration")
    plt.ylabel(r'$L(\theta)$')

    plt.subplot(212)
    RowLabels = ["Topic %s" % k for k in range(1,Z+1)]

    plt.axis('off')
    topics_table = plt.table(cellText=topwords,
                              rowLabels = RowLabels,                          
                              loc='center')
   
    topics_table.auto_set_font_size(False)
    topics_table.set_fontsize(10)
    plt.title("Top words per topic")
    plt.show()
    return 0


def printUsage():
    print("Usage : python PLSI.py [-h] [-n max_iteration] [-d raw_data | -p processed_data -v vocab] [-z #Topics]\n")
    print("Options:\n")
    print("-h   : The actual help\n")
    print("-d raw_data   : file containing row data, each line contains the document content, at the end a procced and vocabulary files will be generated\n")
    print("-p processed_dat : each line represent a document formated as follow <word>:<count>. The open -v can be specified to get the words\n")
    print("-v vocab : This option specifies the file that contains the vocabulary\n")
    print("-z #Topics : The number of topics (default 5)")
    print("-n max_iteration : used to set the number of iterations until convergence (default 100)\n")

#this function will be called if processed_data does not exist
def process_raw(f):
    helper = Helper()
    print("Pre-processing documents\n")
    docs = helper.PreProcess(f)

    print(len(docs["documents"]) , 'document processed \n' , len(docs["vocab"]) ,  'different words')

    if(len(docs["documents"])>0):
        helper.writeProcessedData(docs["documents"],docs["vocab"], "Processed_data.txt")
    
    return(docs)
#if processed_data.txt already exists, this function is called.
def get_processedData(f, vocab_file):
    helper = Helper()
    print("Reading data \n")
    docs = helper.readProcssed(f, vocab_file)
    print("%s documents processed \n %s different words \n" % (len(docs["documents"]),len(docs["vocab"])) )
    return(docs)

def main(argv):

    #number of iterations for the EM update rule
    MAX_ITR = 100
    EPSILON = 0.001
    print("hello")
    #file is the text file that contains all the documents.
    #@Poornav check this part; 
    #Each line in a file is a document; 
    f = "data.txt"   
    docs = {"documents":[], "vocab":[]}
    proccessed = 0
    #vocabulary file ( -v when called from CLI will override this info )
    vocab_file = "vocab.txt"
    #This is the number of classes
    Z = 5

    try:
      opts, args = getopt.getopt(argv,'n:p:d:v:z:h')

    except getopt.GetoptError :
      printUsage()
      sys.exit(0)

    for opt, arg in opts :        
        if(opt == "-d"):            
            if(len(arg)==0):
                print("please specify the data file")
                sys.exit(0)
            f = arg
        else:
            if(opt == "-p"):
                proccessed = 1
                if(len(arg)==0):
                  print("please specify the data file")
                  sys.exit(0)
                f = arg                
            else:
                if(opt == "-v"):
                    if(len(arg)==0):
                        print("please specify the vocabulary file")
                        sys.exit(0)
                    vocab_file = arg                 

        if(opt == "-h"):
            printUsage()
            sys.exit(0)

        if(opt == "-n"):
            if(len(arg)==0):
                sys.exit(0)
            MAX_ITR = int(arg)

        if(opt == "-z"):
            if(len(arg)==0):
                sys.exit(0)
            Z = int(arg)


   
    if(proccessed == 0):
        docs = process_raw(f)
    else:
        if(len(vocab_file)==0):
            print("please specify the vocabulary file")
            sys.exit(2)        
        docs = get_processedData(file, vocab_file)
        

        
    helper = Helper()                
       
    #Randomly initilize probabilities
    # d(w|z)  |W| * |Z|
    p_wz  = np.random.rand(len(docs["vocab"]),Z)  
    row_sum = p_wz.sum(axis=1) 
    p_wz = p_wz / row_sum[:, np.newaxis]

    # p(d|z) |D| * |Z|
    p_dz  = np.random.rand(len(docs["documents"]), Z) 
    row_sum = p_dz.sum(axis=1)
    p_dz = p_dz / row_sum[:, np.newaxis]

    # p(z)
    p_z   = np.random.rand(Z) 
    p_z = p_z / sum(p_z)

    converge= 0
    it = 0
    likelihood = 0

    likelihood_vals = []
    print("Estimating probabilities\n")
    while( (converge == 0) and it <MAX_ITR) :    
        p_wz = helper.update_p_wz(p_wz,p_dz,p_z,docs["documents"],Z,len(docs["vocab"]))
        p_dz = helper.update_p_dz(p_wz,p_dz,p_z,docs["documents"],Z)
        p_z  = helper.update_p_z(p_wz,p_dz,p_z,docs["documents"],Z)

        likelihood_new = helper.log_likelihood(p_wz,p_dz,p_z,docs["documents"])    

        if(abs(likelihood-likelihood_new) <= EPSILON):
            converge = 1

        it +=1
        likelihood_vals.append(likelihood_new)
        print("Iteration : %s \t likelihood : %s" % (it,likelihood_new))
        likelihood = likelihood_new


    print("Getting the list of topwords")
    restuls = helper.GetTopword(p_wz,docs["vocab"],10)

    print("Writting results to file results.txt\n")
    res_file = open("results.txt","w")
    for i in range(len(restuls["Top"])):
        res_file.write( "%s\n" % " ".join(restuls["Top"][i]))
        res_file.write( "%s\n" % " ".join([ str(p) for p in restuls["Probs"][i] ] ))

    res_file.close()
    plot_convergence(likelihood_vals, Z, restuls["Top"])




if __name__ == "__main__":
    main(sys.argv[1:])

