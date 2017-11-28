"""
    Parse URL related to some group of vehicles containing "page=x" part.
    Produces file containing URLs for various page numbers.
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--url", type=str)
parser.add_argument("--outfile", type=str)
parser.add_argument("--num_pages", type=int)

args = parser.parse_args()
url = args.url
num_of_pages=args.num_pages
file_name = args.outfile

word= "page="

start_idx = 0
for i in range(len(url)-len(word)):
    print url[i]
    if url[i] == 'p':
        if url[i:(i+len(word))] == word:
            start_idx = i
            print i
            break

f = open(file_name, 'w')
for i in range(num_of_pages):
    line_url = url[:start_idx] + word + str(i+1) + url[(start_idx+len(word)+1):] + '\n'
    f.write(line_url)    
    print line_url    

f.close()   











    
