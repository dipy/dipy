def clean_string(teststr):
    if teststr[:2] == "\n\n":
        teststr = teststr[2:]
    if teststr[-2:] == "\n\n":
        teststr = teststr[:-2]
    return teststr


def make_notebook(fname):
    file_path = os.path.join(directory, fname)
    f = open(file_path, "r")
    fdata = f.read()
    f.close()
    allcells = fdata.split("\"\"\"")

    textcells = [clean_string(allcells[i]) for i in range(1, len(allcells), 2)]
    codecells = [clean_string(allcells[i]) for i in range(2, len(allcells), 2)]
    codecells = [new_code_cell(source=codecells[i], execution_count=i,)
                 for i in range(len(codecells))]
    textcells = [new_markdown_cell(source=textcells[i])
                 for i in range(len(textcells))]

    cells = []
    for i in range(0, len(allcells)):
        try:
            cells.append(textcells[i])
            cells.append(codecells[i])
        except:
            pass

    nb0 = new_notebook(cells=cells,
                       metadata={
                           'language': 'python',
                       }
                       )

    f = codecs.open("../examples_notebook/" + fname +
                    '.ipynb', encoding='utf-8', mode='w')
    nbf.write(nb0, f, 4)
    f.close()

file_list = validated_examples
directory = "../examples"

for fname in file_list:
    try:
        make_notebook(fname)
    except:
        print("Some error")
        continue
