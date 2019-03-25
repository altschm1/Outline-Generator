def extract_relevant_info(text):
    titles = []
    text_bytes = []
    content = []
    title_tag = False
    i = 0
    while i < len(text):
        if text[i:i+7] == "<title>":
            title_tag = True
            titles.append("")
            i = i + 7
        
        if title_tag:
            titles[-1] = titles[-1] + text[i]
        
        if text[i:i+8] == "</title>":
            title_tag = False
            i = i + 8
        
        i = i + 1
        
    return titles, content

if __name__ == "__main__":
    # read the file
    f = open("artists_xml.xml")
    text = f.read()
    f.close()
    
    # get a list of titles, and wiki content from the whole xml page
    titles, text_stream = extract_relevant_info(text)
    print(titles)
	
    # create a set of txt files with name title.txt in folder music
    