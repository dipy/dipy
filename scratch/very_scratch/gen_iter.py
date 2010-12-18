class Reverse:
    "Iterator for looping over a sequence backwards"
    def __init__(self, data):
        self.data = data
        self.index = len(data)
    def __iter__(self):
        return self
    def next(self):
        if self.index == 0:
            raise StopIteration
        self.index = self.index - 1
        return self.data[self.index]

class ReverseGen:
    'Iterator class using generator'    
    def __init__(self, data):
        self.data = data                
    def __iter__(self):        
        for index in range(len(self.data)-1, -1, -1):
            yield self.data[index]
            
rev = Reverse('golf')
iter(rev)

print('class')
for char in rev:
    print char    
    
def reverse(data):
    for index in range(len(data)-1, -1, -1):
        yield data[index]

print('generator')
for char in reverse('golf'):
    print char
    
print('class generator')
revgen = ReverseGen('golf')
iter(rev)
for char in revgen:
    print char
   

    
    


    
