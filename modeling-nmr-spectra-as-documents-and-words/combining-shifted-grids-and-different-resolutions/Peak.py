class Peak:
    def __init__(self):
        # compound to which the NMR belongs
        self.compound = None
        # family to which the compound belongs
        self.family = None
        # x and y coordinates (h and c values) of a peak
        self.h = None 
        self.c = None
        # mappings of peaks to words
        self.word_map = list(set())

    def map_peak_to_word(self,word):
        '''This method takes a set as an argument and appends it to word_map'''
        self.word_map.append(word)
    
    def map_peak_to_word(self,word):
        '''Overloaded function that takes a list(set) as an argument and concatenates it with word_map'''
        self.word_map = self.word_map + word
