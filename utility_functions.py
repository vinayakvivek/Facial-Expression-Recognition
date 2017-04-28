import os, shutil, sys, time, re, glob, csv

# Get images, labels tuple for CK+ datset
def importCKPlusDataset(dir = 'CKPlus', categories = None, includeNeutral = False, contemptAs = None):

    # Note: "Neutral" is not labeled in the CK+ dataset
    categoriesCK = ['Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']

    if categories is None:
        categoriesEitW = [ 'Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise']
    else:
        categoriesEitW = categories

    # Root directories for images and labels. Should have no other .txt or .png files present
    dirImages = dir + '/CKPlus_Images'
    dirLabels = dir + '/CKPlus_Labels'

    if contemptAs is not None:
        # Verify a valid string was supplied
        try:
            ind = categoriesEitW.index(contemptAs)
        except ValueError:
            raise ValueError("\nError in importCKPlusDataset(): contemptAs = '" + contemptAs + "' is not a valid category. Exiting.\n")

    # Get all possible label and image filenames
    imageFiles = glob.glob(dirImages + '/*/*/*.png')
    labelFiles = glob.glob(dirLabels + '/*/*/*.txt')

    # Get list of all labeled images:
    # Convert label filenames to image filenames
    # Label looks like: CK_Plus/CKPlus_Labels/S005/001/S005_001_00000011_emotion.txt
    # Image looks like: CK_Plus/CKPlus_Images/S005/001/S005_001_00000011.png
    allLabeledImages = []

    for label in labelFiles:
        img = label.replace(dirLabels,dirImages)
        img = img.replace('_emotion.txt','.png')
        allLabeledImages.append(img)

    # Construct final set of labeled image file names and corresponding labels
    # Be sure not to include images labeled as "contempt", since those are not part of our vocabulary
    labeledImages = []
    labels = []
    labelNames = []
    contemptImages = []
    for ind in range(len(labelFiles)):
        curLabel = labelFiles[ind]
        curImage = allLabeledImages[ind]

        # Open the image as binary read-only
        with open(curLabel, 'rb') as csvfile:

            # Convert filestream to csv-reading filestream
            rd = csv.reader(csvfile)
            str = rd.next()

            # Get integer label in CK+ format
            numCK = int(float(str[0]))

            # Get text label from CK+ number
            labelText = categoriesCK[numCK-1]

            if labelText != 'Contempt':
                numEitW = categoriesEitW.index(labelText)
                labeledImages.append(curImage)
                labels.append(numEitW)
                labelNames.append(labelText)
            elif contemptAs is not None:
                # Lump "Contempt" in with another category
                numEitW = categoriesEitW.index(contemptAs)
                labeledImages.append(curImage)
                labels.append(numEitW)
                labelNames.append(labelText)
            else:
                # Discard "Contempt" image
                contemptImages.append(curImage)

    images = labeledImages

    # For testing only:
    #images = images[0:10]
    #labels = labels[0:10]

    return images, labels #, labelNames

# Get entire dataset
# Inputs: Dataset root directory; optional dataset name
# Returns: List of all image file paths; list of correct labels for each image
def importDataset(dir, dataset, categories):
    imgList = glob.glob(dir+'/*')
    labels = None

    # Datset-specific import rules:
    if dataset.lower() == 'jaffe' or dataset.lower() == 'training':
        # Get Jaffe labels
        jaffe_categories_map = {
            'HA': categories.index('Happy'),
            'SA': categories.index('Sad'),
            'AN': categories.index('Angry'),
            'FE': categories.index('Fear'),
            'DI': categories.index('Disgust'),
            'SU': categories.index('Surprise')
            }

        labels = []

        for img in imgList:
            if os.path.isdir(img):
                continue
            key = img.split('.')[3][0:2]
            if (key in jaffe_categories_map):
                labels.append(jaffe_categories_map[key])

    elif dataset.lower() == 'ckplus':
        # Pathnames and labels for all images
        imgList, labels = importCKPlusDataset(dir, categories=categories, contemptAs=None)

    elif dataset.lower() == 'misc':
        labels = [0,1,2,3,4,5,6]

    else:
        print 'Error - Unsupported dataset: ' + dataset
        return None

    # Make sure some dataset was imported
    if len(imgList) <= 0:
        print 'Error - No images found in ' + str(dir)
        return None

    # Return list of filenames
    return imgList, labels