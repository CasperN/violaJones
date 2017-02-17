import numpy as np
from PIL import Image
import os
import re
import itertools
import sys
from multiprocessing import Pool, cpu_count
from functools import partial
import dill
import pickle
import time

"""
To run this program in bash:

python facedetector0.py [test_type] [num images] [num_haar boosters (only used in type a)] [output file name (only used in type p)]

I did not give the program the ability to choose which classifier types or cascade design in bash

'a' tests adabooster
'c' tests cascade
'p' tests cascade and runs the detector over the given image

This code is paralleized so a few thousand functions in a 3 step may run in a half hour
"""


# Read images
def get_data(facedir ='faces', backgrounddir = 'background', limit = 2000):

    data = []
    for directory, y in [(facedir, 1),(backgrounddir, -1)]:
        # Take only jpgs (.DSstore is a file in every dir)
        files = [f for f in os.listdir(directory) if re.match(r'.*\.jpg', f)]
        for filename in np.random.choice(files,limit):
            img = Image.open('{}/{}'.format(directory,filename),'r')
            # Convert images to GrayScale
            x = np.array(img.convert('L'))
            data += [(x,y)]
    # shuffle so backgrounds are mixed in with faces
    np.random.shuffle(data)
    images, labels = (np.array(i) for i in zip(*data))
    return images, labels


################################################################################
# Haar functions, decision stumps
################################################################################

def h1x3(ii,i0,j0,i1,j1):
    # 1 row 3 column Haar function (too long to be lambda )
    di = (i1-i0)/3
    column1 = HAARS['1x1'](ii,i0       , j0, i0 + di  , j1 )
    column2 = HAARS['1x1'](ii,i0 + di  , j0, i0 + 2*di, j1 )
    column3 = HAARS['1x1'](ii,i0 + 2*di, j0,  i1      , j1 )
    return column3 + column1 - column2

def h3x1(ii,i0,j0,i1,j1):
    # 3 row 1 column Haar function
    dj = (j1-j0)/3
    row1 = HAARS['1x1'](ii,i0, j0       , i1, j0 + dj  )
    row2 = HAARS['1x1'](ii,i0, j0 + dj  , i1, j0 + 2*dj)
    row3 = HAARS['1x1'](ii,i0, j0 + 2*dj, i1,  j1      )
    return row3 + row1 - row2

def h2x2(ii,i0,j0,i1,j1):
    # 2 row 2 columns
    b1 = HAARS['1x1'](ii, i0       , j0       , (i0+i1)/2, (j0+j1)/2 )
    b2 = HAARS['1x1'](ii, (i0+i1)/2, j0       , i1       , (j0+j1)/2 )
    b3 = HAARS['1x1'](ii, i0       , (j0+j1)/2, (i0+i1)/2, j1        )
    b4 = HAARS['1x1'](ii, (i0+i1)/2, (j0+j1)/2, i1       , j1        )
    return b1 - b2 - b3 + b4

def haar_fcns_dict(shape = (64,64)):
    """
    Dictionary holding the various haar features
    """
    n,m = shape[0]-1, shape[1]-1
    h = {}
    # Subset window features
    h['1x1'] = lambda ii,i0,j0,i1,j1: ii[i1,j1] - ii[i1,j0] - ii[i0,j1] + ii[i0,j0]
    h['2x1'] = lambda ii,i0,j0,i1,j1: h['1x1'](ii,i0,j0,i1,j1) \
                                         - h['1x1'](ii,i0,j0,(i1+i0)/2,j1)
    h['1x2'] = lambda ii,i0,j0,i1,j1: h['1x1'](ii,i0,j0,i1,j1) \
                                         - h['1x1'](ii,i0,j0,i1,(j0+j1)/2)
    h['1x3'] = h1x3
    h['3x1'] = h3x1
    h['2x2'] = h2x2
    return h

HAARS = haar_fcns_dict()

def get_features(shape = (64,64)):
    '''
    Combinatorical explosion yet somehow faster than 1 round of boosting
    '''
    min_window = 10
    max_window = 64
    print('min subwindow size: {} | max subwindow size: {}'.format(
            min_window,max_window))
    features = []
    feature_types = ('1x3','3x1','2x2')
    print 'Feature types: ' + ', '.join(feature_types)
    for i0,i1 in itertools.combinations(range(0,64 - min_window), 2):
        i1 += min_window - 1 # i1 >= i0 + 1 initially by itertools
        if i1 - i0 > max_window: continue

        for j0,j1 in itertools.combinations(range(0,64 - min_window), 2):
            j1 += min_window - 1 # j1 >= j0 + 1 initially by itertools
            if j1 - j0 > max_window: continue

            for t in feature_types:
                features.append( (t,i0,j0,i1,j1) )

    print('%d features available'%len(features))
    np.random.shuffle(features)
    return features



def calculate_feature(f,ii):
    """
    calculate feature f on integra image ii
    """
    h_type = f[0]
    args = [ii] + list(f[1:])
    res = HAARS[ h_type ]( * args )
    return res


def find_haar_thresholds(f,D,iimages,labels):
    """
    given a haar function, it finds the best polarity and threshold
    to minimize error over distribution
    """
    n = len(iimages)
    h_scores = np.array([calculate_feature(f,ii) for ii in iimages])
    idx = np.argsort(h_scores)
    h_scores = h_scores[idx]
    _, u_idx =  np.unique(h_scores,return_index = True)
    slabels = labels[idx] # lables sorted by haar score
    # Find cumilative sum of 2 types of error of weights sorted by haar score
    face_error = np.cumsum( D[idx] * (slabels ==  1))
    back_error = np.cumsum( D[idx] * (slabels ==  -1))
    # find threshold and polarity to weighted error
    if back_error[-1] < face_error[-1]: # Check threshold above max(h_scores)
        min_error = back_error[-1]
        min_error_i = n-1
        polarity = -1
    else:
        min_error = face_error[-1]
        min_error_i = n-1
        polarity = 1
    # Check threshold between unique values of h_scrores
    # descending because numpy.unique gives FIRST instances of unique indicies
    for i in u_idx[::-1]:
        i = i-1 # to check threshold between i-1,i values of h_scores
        error_i_p = face_error[i] + back_error[-1] - back_error[i]
        error_i_m = back_error[i] + face_error[-1] - face_error[i]
        if error_i_p < min_error:
            min_error = error_i_p
            min_error_i = i
            polarity = 1
        if error_i_m <= min_error:
            min_error = error_i_m
            min_error_i = i
            polarity = -1
    # Get threshold from index
    if min_error_i == -1:    # threshold is min(h_scores) - 1
        threshold = h_scores[0] - 1
        polarity *= -1
    elif min_error_i == n-1: # threshold is max(h_scores) + 1
        threshold = h_scores[-1] + 1
    else:                    # threshold bewteen indicies
        threshold = (float(h_scores[min_error_i ])
                        + h_scores[min_error_i + 1 ])/2

    return min_error, f, threshold, polarity


def make_stump(f,polarity,threshold):
    '''
    Makes decision stump from haar score
    '''
    if polarity == 1:
        stump = lambda ii: ((calculate_feature(f,ii) - threshold) >= 0)*2-1
    else:
        stump = lambda ii: ((calculate_feature(f,ii) - threshold) <= 0)*2-1
    return stump


################################################################################
### Adaboost
################################################################################

def p_find_haar_threholds(args):
    # Makes the multiprocessing work, weird things happen with lambdas
    return find_haar_thresholds( * args )


def best_learner_pool(D,learners,iimages,labels):
    """
    Pooled version
    """
    args = ( (f, D, iimages, labels) for f in learners )
    error_and_stump = POOL.map(p_find_haar_threholds, args)
    res = min(error_and_stump)
    return res


def adaboosted_classifier(learners,iimages,labels,fpr_threshold = .3):
    """
    Trains a meta learner from basic decision stumps by adaboost algorithm
    """
    assert len(learners) > 0, 'no learners!'
    t = 0
    n = len(iimages)
    num_backgrounds = sum(labels == -1)
    functions = []
    D = np.array([1.0/n for i in range(n)]) # weight distribution over data
    FPR = 1
    while FPR >= fpr_threshold:
        t += 1
        error, f, threshold, polarity = best_learner_pool(D, learners, iimages, labels)
        stump = make_stump(f,polarity,threshold)
        #check error
        try:
            error2 = sum(D * (np.array(map(stump,iimages)) != labels) )
            assert abs(error - error2) < 10**-9, \
            'Error from find_haar_thresholds incorrect, true error: %f'%error2
        except:
            error = error2
        # Reweight Distribution
        alpha = .5 * np.log( (1-error) / error )
        normalizer = 2 * ( error * (1-error) )**.5
        D = D * np.exp( - alpha * labels * \
                        np.array([stump(ii) for ii in iimages])) / normalizer
        # check distribution
        try:
            assert abs(sum(D) - 1) < 10** -12, 'normalization'
        except:
            print('---------------- Normalization failed: Renormalizing')
            D /= sum(D)
        # Update Adaptively selected functions
        functions.append((stump,alpha,f,threshold,polarity))
        b = (lambda functions:
                        lambda x: sum(a * h(x) for h,a,_,__,___ in functions)
                    )(functions)
        # Find threshold so no false negatives
        boosted_scores = np.array([b(ii) for ii in iimages ])
        # Choose th such that there are no false negatives
        th = min(np.min(boosted_scores[labels == 1]),0)
        num_false_positives = sum(boosted_scores[labels == -1] >= th,.0)
        FPR = num_false_positives / num_backgrounds
        # PRINT stuff
        uFpr = sum(boosted_scores[labels == -1] >=0, .0)/num_backgrounds
        uFnr = sum(boosted_scores[labels ==  1] < 0, .0)/(n - num_backgrounds)
        Boosted_error = sum( (boosted_scores >= 0)*2-1 != labels, 0.0)/len(boosted_scores)
        print(' t=%d\taFPR %d pct\tBest W Err %f | unadj: err %f fpr %f fnr %f '%(
            t,FPR*100,error,Boosted_error,uFpr,uFnr))


    print ('features selected:')
    for lmda, a, f, t, p in functions:
        print '\t' + str(a) + '\t' + str(f) + '\t polarity:{}\t threshold:{}'.format(p,t)

    predicted_faces = (boosted_scores >= th)
    final_boosted_classifier = (lambda functions,th:
        lambda ii: (sum(a* h(ii) for h,a,_,__,___ in functions) > 0)*2-1
    )(functions,th)

    return final_boosted_classifier, predicted_faces


def classifier_cascade_fcn(iimage,boosted_functions, ):
    '''
    if it only yields true then its a face
    '''
    for b in boosted_functions:
        yield b(iimage) == 1 # face


def cascade(iimages,labels,boosting_specs=[(127,.9),(500,.7),(1000,.5),(2000,.3),(4000,.3),(7878,.3)]):
    '''
    Classifier cascade: adaboosts few haar functions to classify easy backgrounds
    then adds more haar functions to filter out harder backgrounds, etc
    '''
    features = get_features()
    boosted = []
    # Use fewer features initially to classify fast/easy true negatives
    for n_fcns, fpr_threshold in boosting_specs:
        learners = features[:n_fcns]
        print ('\n'+'-'*7 + ' Boosting with %d learners %d faces, '
                             '%d backgrounds, and %f FPR th '%(
            len(learners), sum(labels == 1), sum(labels == -1),fpr_threshold ) \
            + '-'*7 + '\n'
            )
        b, predicted_faces= adaboosted_classifier(learners, iimages,labels,
                                                    fpr_threshold)
        boosted.append(b)
        # remove easy backgrounds
        labels  = labels[predicted_faces]
        iimages = iimages[predicted_faces]

    print ('-'*30 + ' Cascade Complete')
    final = (lambda boosted:
                lambda ii: all(classifier_cascade_fcn(ii,boosted)) * 2 -1
            )(boosted)
    return final


################################################################################
##### Testing
################################################################################


def find_faces(classifier, step = 4,imagefilename ='class.jpg' , output_save_name = 'output.jpg'):
    """
    takes an images and runs the classifier on every 64*64 square
    draws a rectangle around that square, returns the image.
    """
    print ('finding faces in ' + imagefilename)
    img = Image.open(imagefilename,'r')
    res = np.array(img)
    iimage = res.astype(dtype='int64').cumsum(0).cumsum(1)
    # go through all squares
    shape = res.shape
    i = 0
    num_detections = 0
    face_places = []
    while i < shape[0] - 64:
        j = 0
        while j < shape[1] - 64:
            if any(in_face(i,j) for in_face in face_places):
                # skip this here face region
                j += 64
                continue
            #ii = iimage[i:i+64,j:j+64] - iimage[i,j] + res[i,j] # normalize the window
            ii = res[i:i+64,j:j+64].astype(dtype='int64').cumsum(0).cumsum(1)
            if classifier(ii) > 0:
                num_detections += 1
                # Outlaw this face region
                in_face = (lambda i,j: \
                    lambda i_, j_: ((i-64<=i_<=i+64) and (j-64<=j_<=j+64))
                )(i,j)
                face_places.append(in_face)
                # Draw box on result
                res[i:i+64,j]    = 255 # Left
                res[i:i+64,j+64] = 255 # Right
                res[i,j:j+64]    = 255 # Top
                res[i+64,j:j+64] = 255 # Bottom
            j+= step
        i+= step

    print('Found %d Faces! The human thinks there are like 58? He can code but not count'%num_detections)

    im = Image.fromarray(res)
    im.save(output_save_name)


def test_adabooster(limit, num_haar):
    '''
    Test the FNR, FPR of adabooster
    '''
    images, labels = get_data(limit = limit)
    iimages = images.cumsum(1).cumsum(2).astype(dtype='int64')
    features = get_features()
    print('number of features %d'%len(features))

    print('using %d \n'%num_haar)
    features = features[:num_haar]

    b,_ = adaboosted_classifier(features,iimages,labels,fpr_threshold=.3)

    test_images, test_labels = get_data(limit = limit)
    test_iimages = test_images.cumsum(1).cumsum(2).astype(dtype='int64')
    predictions = np.array(map(b,test_iimages))

    correct = sum(predictions == test_labels ,0.0)
    false_positives = sum(predictions[test_labels == 1]  != 1 , 0.0)
    false_negatives = sum(predictions[test_labels == -1] != -1, 0.0)
    print('')
    print('percent correct {} \t FPR {} \t FNR {}'.format(
        correct/(2*limit) , false_positives/limit, false_negatives/limit
    ))


def test_cascade(limit):
    '''
    Test the FNR, FPR of adabooster
    '''
    images, labels = get_data(limit = limit)
    iimages = images.cumsum(1).cumsum(2).astype(dtype='int64')

    thresholds = [.3,.3,.3]
    numhaars   = [3000,6000,9000]

    b = cascade(iimages,labels,boosting_specs = zip(numhaars,thresholds))

    test_images, test_labels = get_data(limit = limit)
    test_iimages = test_images.cumsum(1).cumsum(2).astype(dtype='int64')
    predictions = np.array(map(b,test_iimages))
    correct = sum(predictions == test_labels ,0.0)
    false_positives = sum(predictions[test_labels == 1]  != 1 , 0.0)
    false_negatives = sum(predictions[test_labels == -1] != -1, 0.0)
    print('percent correct {} \t FPR {} \t FNR {}'.format(
        correct/(2*limit) , false_positives/limit, false_negatives/limit
    ))
    return b


if __name__ == '__main__':

    s = time.time()
    test = sys.argv[1]
    limit = int(sys.argv[2])
    num_haar = int(sys.argv[3])
    output_save_name = sys.argv[4]

    POOL = Pool(cpu_count())

    if test == 'a':
        test_adabooster(limit,num_haar)
    elif test == 'c':
        test_cascade(limit)
    elif test == 'p':
        b = test_cascade(limit)
        find_faces(b, output_save_name = output_save_name)

    print '\nDone! time elapsed {}'.format(time.time()-s)


'''
Testing code for FIND HAAR threshold

    try:
        true_error = sum( D* (np.array(map(full_classifier,iimages)) != labels) )
        assert abs(true_error - min_error) < 10 ** -9, true_error
    except:

        print('expected error: {} \t true error: {}'.format(min_error,true_error))
        print('para: {} \t min_error_i: {}\t threshold: {}\t error: {}'.format(
                      para,min_error_i,threshold,min_error))
        print ('')
        L_ = slabels[:min_error_i + 1]
        R_ = slabels[min_error_i + 1:]
        print 'L: {}'.format(L_)
        print 'L Correctly predicts: {}/{}'.format(sum(L_ != para), len(L_))
        print 'R: {}'.format(R_)
        print 'R Correctly predicts: {}/{}'.format(sum(R_ == para), len(R_))

        print('\n the following are true values, from full_classifier\n')
        L = slabels[h_scores[idx] < threshold]
        R = slabels[h_scores[idx] >= threshold]
        print 'L: {}'.format(L)
        print 'L correctly predicts {}/{}'.format(sum(L!=para),len(L))
        print 'R: {}'.format(R)
        print 'R correctly predicts {}/{}'.format(sum(R==para),len(R))
        print true_error

        assert False


# FROM ADABOOST CLASSFIER


'''
