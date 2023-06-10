import cv2 as cv

def get_hog_features(image): # returns a histogram of gradients for a given image
    winSize = (64,64) # HOG parameters
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    winStride = (8,8)
    padding = (8,8)
    locations = ((10,20),)
    hist = hog.compute(image,winStride,padding,locations)
    return hist

def extract_hog_features(images):
    features = []
    labels = []
    x=0
    for lab in images:

        for img in lab:
    
            hog_features = get_hog_features(img)
    #        features.sort(key=lambda x: x.distance)
            features.append(hog_features)
            labels.append(x)
            print("HOG FEATURES", hog_features)
        x+=1    

    return (features, labels)