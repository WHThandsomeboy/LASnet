import math

PathSpectogramFolder = '/e/wht_project/eeg_data/5s_spectograms'

patients = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "14", "15", "16", "17", "18", "19", "20",
            "21", "22", "23"]


# patients = ["05"]


# nSeizure = 0
#
# interictalSpectograms = []
# preictalSpectograms = []


def loadSpectogramData(indexPat):
    global PathSpectogramFolder
    nFileForSeizure = 0
    nSeizure = 0
    interictalSpectograms = []
    preictalSpectograms = []
    f = open(PathSpectogramFolder + '/patient' + patients[indexPat] + '/datamenu.txt', 'r')
    line = f.readline()
    while not "SEIZURE" in line:
        line = f.readline()
    nSeizure = int(line.split(":")[1].strip())
    line = f.readline()
    line = f.readline()
    nSpectograms = int(line.strip())
    nFileForSeizure = math.ceil(math.ceil(nSpectograms / 100) / nSeizure)
    line = f.readline()

    # Interictal files
    cont = -1
    indFilePathRead = 0
    while "npy" in line and indFilePathRead < nSeizure * nFileForSeizure:
        if indFilePathRead % nFileForSeizure == 0:
            interictalSpectograms.append([])
            cont = cont + 1
            interictalSpectograms[cont].append(line.split(' ')[2].rstrip())  # .rstrip() remove \n
            indFilePathRead = indFilePathRead + 1
        else:
            if len(line.split(' ')) >= 3:
                interictalSpectograms[cont].append(line.split(' ')[2].rstrip())
            indFilePathRead = indFilePathRead + 1

        line = f.readline()
    line = f.readline()  # PREICTAL
    line = f.readline()  # spectogram
    line = f.readline()  # seizure(SEIZURE X)

    # Preictal files
    cont = -1
    indFilePathRead = 0
    while line.strip() != "":
        if "SEIZURE" in line:
            # 这里换行了，下面就不用重复再换行了
            line = f.readline()
            if len(line.split(' ')) >= 3:
                preictalSpectograms.append([])
                cont = cont + 1
                preictalSpectograms[cont].append(line.split(' ')[2].rstrip())
                indFilePathRead = indFilePathRead + 1
        else:
            if len(line.split(' ')) >= 3:
                preictalSpectograms[cont].append(line.split(' ')[2].rstrip())

            indFilePathRead = indFilePathRead + 1
        if "SEIZURE" not in line:
            line = f.readline()
    f.close()
    return interictalSpectograms, preictalSpectograms, nSeizure
