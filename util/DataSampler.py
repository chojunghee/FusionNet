# code from https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
import random

def getSampleIndices(ds, frac, num_classes, COCO=False, shuffle=False):
    class_counts = {}
    k = int(len(ds)*frac/num_classes)
    SampleIndices = []
    data_number = [i for i in range(len(ds))]

    if shuffle:
        random.shuffle(data_number)

    if COCO == True:
        SampleIndices = data_number[:k]
    
    else:
        for indice in data_number:
            # get the corresponding label of data indice
            c = int(ds[indice][1])
            # count the label in dictionary
            class_counts[c] = class_counts.get(c, 0) + 1
            if class_counts[c] <= k:
                SampleIndices.append(indice)
    
    print('We sample {:d} ({:.2f} %) training data'.format(len(SampleIndices),frac*100))

    return SampleIndices
