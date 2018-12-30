class TextFormatter: 
  
    COLORCODE = { 
        'k': 0,  # black 
        'r': 1,  # red 
        'g': 2,  # green 
        'y': 3,  # yellow 
        'b': 4,  # blue 
        'm': 5,  # magenta 
        'c': 6,  # cyan 
        'w': 7   # white 
    } 
  
    STYLECODE = { 
        'b': 1,  # bold 
        'f': 2,  # faint 
        'i': 3,  # italic 
        'u': 4,  # underline 
        'x': 5,  # blinking 
        'y': 6,  # fast blinking 
        'r': 7,  # reverse 
        'h': 8,  # hide 
        's': 9,  # strikethrough 
    } 
  
    # constructor 
    def __init__(self, fg, bg=None, st=None):
        self.prop = {}
        self.prop["fg"] = 30 + self.COLORCODE[fg]
        self.prop["bg"] = 40 + self.COLORCODE[bg] if bg is not None else None
        self.prop["st"] = self.STYLECODE[st] if st is not None else None
 
    # formatting function 
    def format(self, string): 
        w = [self.prop['st'],self.prop['fg'], self.prop['bg']] 
        w = [ str(x) for x in w if x is not None ] 
        # return formatted string 
        return '\x1b[%sm%s\x1b[0m' % (';'.join(w), string) if w else string 
  
    # output formatted string 
    def out(self, string): 
        print(self.format(string))

error = TextFormatter("r", st="b")
eva = TextFormatter("m")
info = TextFormatter("m")
log = TextFormatter("b", st="b")

