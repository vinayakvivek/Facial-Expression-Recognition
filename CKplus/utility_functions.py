import os, shutil, sys, time, re, glob, csv

# Get images, labels tuple for CK+ datset
def importCKPlusDataset(dir = 'CKPlus', categories = None, includeNeutral = False, contemptAs = None):
    ############################################################################
    # Function: importCKPlusDataset
    # Depending on preferences, this ranges from 309 - 920 images and labels
    #    - 309 labeled images
    #    - 18 more "Contempt" images (not in our vocabulary)
    #    - 593 neutral images
    # 
    # For this to work, make sure your CKPlus dataset is formatted like this:
    # CKPlus = root (or whatever is in your 'dir' variable)
    # CKPlus/CKPlus_Images = Root for all image files (no other file types here)
    #    Example image path:
    #    CKPlus/CKPlus_Images/S005/001/S005_001_00000011.png
    # 
    # CKPlus/CKPlus_Labels = Root for all image labels (no other file types)
    #    Example label path:
    #    CKPlus/CKPlus_Labels/S005/001/S005_001_00000011_emotion.png
    #
    # CKPlus/* - anything else in this directory is ignored, as long as it
    # is not in the _Images or _Labels subdirectories
    # 
    # Optional inputs:
    # dir - Custom root directory for CKPlus dataset (if not 'CKPlus')
    #
    # includeNeutral - Boolean to include neutral pictures or not
    #    Note: Every sequence begins with neutral photos, so neutral photos
    #    greatly outnumber all other combined (approximately 593 to 327)
    #
    # contemptAs - Since it's not in our vocabulary, by default all pictures
    # labeled "Contempt" are discarded. But if you put a string here, e.g.
    # "Disgust", pictures labeled "Contempt" will be lumped in with "Disgust"
    # instead of being discarded.
    #
    #
	# RETURN VALUES:
	# images, labels = List of image file paths, list of numeric labels
	# according to EitW numbers
	#
	# Author: Dan Duncan
	#
	############################################################################
	print(includeNeutral)

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

    # if includeNeutral:
    #     # Add all neutral images to our list too:
    #     # The first image in every series is neutral
    #     neutralPattern = '_00000001.png'
    #     neutralInd = categoriesEitW.index('Neutral')
    #     neutralImages = []
    #     neutralLabels = []
    #     neutralLabelNames = []

    #     for imgStr in imageFiles:
    #         if neutralPattern in imgStr:
    #             neutralImages.append(imgStr)
    #             neutralLabels.append(neutralInd)
    #             neutralLabelNames.append('Neutral')

    #     # Combine lists of labeled and neutral images
    #     images = labeledImages + neutralImages
    #     labels = labels + neutralLabels
    #     labelNames = labelNames + neutralLabelNames

    # else:
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
            'NE': categories.index('Neutral'),
            'AN': categories.index('Angry'),
            'FE': categories.index('Fear'),
            'DI': categories.index('Disgust'),
            'SU': categories.index('Surprise')
            }

        labels = []

        for img in imgList:
            if os.path.isdir(img):
                continue
            key = img.split('.')[1][0:2]
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