
# coding: utf-8

# In[62]:

import xml.etree.ElementTree as etree
import Peak
from os import walk
from os import listdir
from os.path import isfile, join


# In[63]:

peaks = list()
data_dir = '../../data/hmdb_data/two_d_nmr_spectra'
#get all the filenames in data_dir



# In[65]:

def get_data():
    two_d_nmr_files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    for f in two_d_nmr_files:
        #Read xml file
        tree = etree.parse(data_dir+'/'+f)
        root = tree.getroot()
        nmr_two_d_peaks = tree.findall('nmr-two-d-peaks')

        #find the peak tags in the xml file
        entries = nmr_two_d_peaks[0].findall("nmr-two-d-peak")

        #read the peak values and populate a Peak object
        for entry in entries:
            p = Peak.Peak()
            p.h = float(entry.find('chemical-shift-x').text)
            p.c = float(entry.find('chemical-shift-y').text)
            p.compound = root.find('id').text
            peaks.append(p)
    return peaks


# In[ ]:



