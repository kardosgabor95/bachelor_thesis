#ez a program a szakdolgozatomhoz keszult, hogy tesztelni tudjam a tanitott konvolucios neurelis halot
#futtatasahoz szuksegesek az alabb importalt konyvtarak, bele ertve a Caffe telepiteset is
#futtatas: python application.py <halot leiro .prototxt file> <tanitott ertekeket tartalmazo .caffemodel file> <kepek forrasat tartalmazo konyvtar> <feldolgozott kepek celja>


import cv2
import time
import caffe
import numpy as np
import glob
import ntpath
from Box import Box
import sys


winW = 195
winH = 195
treshold = 0.9992




def set_caffe(net_arch, model):
    caffe.set_device(0)
    caffe.set_mode_gpu()

    net = caffe.Net(net_arch, model, caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load('/media/gabor/save_1TB/szakdoga/caffenet/train_14/data/szakdolg_mean.npy').mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)
    
    net.blobs['data'].reshape(1,3,227,227)
    
    return (transformer, net)
    


def sliding_window(image, stepSize, winSize):
    for y in xrange(0, image.shape[0] - winSize[0], stepSize):
        for x in xrange(0, image.shape[1] - winSize[1], stepSize):
            yield(x, y, image[y:y + winSize[1], x:x + winSize[0]])
    
    
def merge(boxArray):
    boxes = []
    length = len(boxArray)
    while (len(boxArray) > 0):
        actualBox = boxArray[0]
        
        print "main box num: " + str(len(boxes))
        
        j = 1
        while (j < length):
            print "j: " + str(j)
            if(j == len(boxArray)):
                break
            xOK = 0
            yOK = 0
            # cast ellenorzese
            if (actualBox.cast == boxArray[j].cast):
                if (actualBox.x <= boxArray[j].x and (actualBox.x + actualBox.w) >= boxArray[j].x):
                    xOK = 1
                if (actualBox.x >= boxArray[j].x and actualBox.x <= (boxArray[j].x + boxArray[j].w)):
                    xOK = 1
                if (actualBox.y <= boxArray[j].y and (actualBox.y + actualBox.h) >= boxArray[j].y):
                    yOK = 1
                if (actualBox.y >= boxArray[j].y and actualBox.y <= (boxArray[j].y + boxArray[j].h)):
                    yOK = 1
                if (xOK and yOK):
                    minX = min(actualBox.x, boxArray[j].x)
                    maxX = max(actualBox.x + actualBox.w, boxArray[j].x + boxArray[j].w)
                    minY = min(actualBox.y, boxArray[j].y)
                    maxY = max(actualBox.y + actualBox.h, boxArray[j].y + boxArray[j].h)
                    actualBox = Box(minX, minY, maxX-minX, maxY-minY, actualBox.cast, actualBox.prob)
                    del boxArray[j]
                    length -= 1
                    print "MERGE"
            if not(xOK and yOK):
                j += 1
            else:
                j = 1
                
        boxes.append(actualBox)
        del boxArray[0]
    return boxes
            
                
def run(net_arch, model, image_dir, output_dir):
    (transformer, net) = set_caffe(net_arch, model)

    for filename in glob.glob(image_dir + '*.png'):
        boxes = []
        image = cv2.imread(filename)
        caffeImage = caffe.io.load_image(filename)
                
        for (x, y, window) in sliding_window(caffeImage, stepSize=97, winSize=(winW, winH)):
            im = window
            net.blobs['data'].data[...] = transformer.preprocess('data', im)
            
            out = net.forward()
            
            probability = net.blobs['prob'].data.max()
            cast = out['prob'].argmax()
            
            if probability > treshold and cast != 0:
                boxes.append(Box(x, y, winW, winH, cast, probability))
            
                labels = np.loadtxt("/home/gabor/deep-learning/szakdoga/image_data/pairs.txt", str, delimiter=' ')
                top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
                print labels[top_k]

        person = 0
        bike = 0
        car = 0    
        
        print "length boxes" + str(len(boxes))
        boxes2 = merge(boxes)
        
        
        
        output = image.copy()
        for box in boxes2:
            color = (0, 0, 0)
            cast = 'none'
            if box.cast == 1:
                color = (255, 0, 0)
                cast = 'person'
                person += 1
            elif box.cast == 2:
                color = (0, 255, 0)
                cast = 'bike'
                bike += 1
            elif box.cast == 3:
                color = (0, 0, 255)
                cast = 'car'
                car += 1
                
            cv2.rectangle(output, (box.x, box.y), (box.x + box.w, box.y + box.h), color, 2)
            cv2.putText(output, str(box.prob), (box.x, box.y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(output, cast, (box.x+200, box.y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        basename = str(ntpath.basename(filename))
        cv2.imwrite(output_dir + basename, output)
    
    
if __name__ == "__main__":
    
    #halo leiro .prototxt
    net = sys.argv[1]
    #tanitott modell file
    model = sys.argv[2]
    #feldolgozando kepetet tartalmazo konyvtar
    image_dir = sys.argv[3]
    #feldolgozott kepek celja
    output_dir = sys.argv[4]

    run(net, model, image_dir, output_dir)
    
    
    
    
