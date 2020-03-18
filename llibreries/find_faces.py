import cv2
import cv2.cv as cv
import dlib
import numpy as np

import sys
import os
import glob
import argparse
#from PIL import Image as pilimage

#from keras.models import load_model
#from keras.preprocessing import image

import random
import threading
import thread
import multiprocessing as mp

from scipy.misc import imresize

import sys, getopt
from video import create_capture
from common import clock, draw_str

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
def detect(gray, cascade, args, size_min=30, size_max=500):
    rects = cascade.detectMultiScale(gray, scaleFactor=args.factor_scale, minNeighbors=args.min_neighbors, minSize=(size_min, size_min), maxSize=(size_max,size_max), flags = cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects
    
    face_cascades = [cv2.CascadeClassifier(os.path.expanduser(path)) for path in args.haarcascades]
    
    faces = []
    for cascade in face_cascades:
            f = cascade.detectMultiScale(gray, scaleFactor = args.factor_scale, 
                                        minNeighbors = args.min_neighbors, flags = cv2.CASCADE_SCALE_IMAGE, 
                                        minSize = (size_min,size_min), maxSize = (size_max,size_max))
            f = list(f)
            if f: faces += f
    
    
    #
    if len(faces) == 0:
        return []
    #faces[:,2:] += faces[:,:2]
    return faces


def video1Smoth(args):
    
    print help_message

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade=', 'smile-cascade='])
    try: video_src = video_src[0]
    except: video_src = 0
    args = dict(args)
    #cascade_fn = args.get('--cascade', "../../data/haarcascades/haarcascade_frontalface_alt.xml")
    cascade_fn = args.get('--cascade', "../../data/haarcascades/haarcascade_frontalface_alt.xml")
    nested_fn  = args.get('--nested-cascade', "../../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml")
    #Personal
    smile_fn = args.get('--smile-cascade', "../../data/haarcascades/haarcascade_eye.xml")

    cascade = cv2.CascadeClassifier(cascade_fn)
    nested = cv2.CascadeClassifier(nested_fn)
    smile = cv2.CascadeClassifier(smile_fn)

    cam = create_capture(video_src, fallback='synth:bg=../cpp/lena.jpg:noise=0.05')

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        t = clock()

        rects = detect(img, args, size_min, size_max)
        #rects = detect(gray, cascade)
        vis = img.copy()
        draw_rects(vis, rects, (0, 255, 0))
        for x1, y1, x2, y2 in rects:
            roi = gray[y1:y2, x1:x2]
            vis_roi = vis[y1:y2, x1:x2]
            subrects = detect(roi.copy(), nested)
            draw_rects(vis_roi, subrects, (255, 0, 0))
            # Propio
            subrects2 = detect(roi.copy(), smile)
            draw_rects(vis_roi, subrects2, (0, 0, 255))
        dt = clock() - t

        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
        cv2.imshow('facedetect', vis)

        if 0xFF & cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows()

  
  
# Funcion para distorsionar las caras blur_lvl comprente entre 0.0 a 1.0
def imageSmooth(image,outputDir, mutex=None, blur_lvl=0.3, scale_factor=1.1, min_neighbors=1,
                face_cascade_paths=['haarcascades/haarcascade_frontalface_alt2.xml', 'haarcascades/haarcascade_profileface.xml'],
                exts=['*.jpg','*.jpeg','*.png'],
                min_size=-1, max_size=-1,
                flags=cv2.CASCADE_SCALE_IMAGE):
        
    # Comprovar si estan los directorios salida y crearlos en su defecto
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)
        
    # Dades sobre el nom de l'arxiu imatge
    img_basename = image.split('/')[-1].split('.')[0]
    img_ext = image.split('/')[-1].split('.')[1]

    #poner distintos filtros para el classificador
    face_cascades = [cv2.CascadeClassifier(os.path.expanduser(path)) for path in face_cascade_paths]
        
    # Carrega de la imatge
    img =cv2.imread(image)
    if img is None:
        print "\rImagen " + image + " no cargada"
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        w,h = gray.shape
        if max_size == -1:
            size_max = int(max([w*0.35,h*0.35]))
        else:
            size_max = min([w,h])
            
        if min_size == -1:
            size_min = int(min([w*0.05,h*0.05]))
        else:
            size_min = min_size
        
        #Equalizar Histograma per millorar qualitat per lluminositat
        cv2.equalizeHist(gray, gray);
        
        faces = []
        for cascade in face_cascades:
            f = cascade.detectMultiScale(gray, scaleFactor = scale_factor, 
                                        minNeighbors = min_neighbors, flags = flags, minSize = (size_min,size_min), maxSize = (size_max,size_max))
            f = list(f)
            if f: faces += f
        
        centros = []
        nfaces = 0
        for i, ( x, y, w, h ) in enumerate(faces):
            unico =True # Controla que no haya recuedros repetidos

            #Para Gesionar los recuadros repetidos pulir
            for cx,cy in centros:
                if cx > x and cy > y and cx < (x+w) and cy < (y+h):
                        unico = False # Recuadro repetido
            
            if unico:
                nfaces += 1
                # Guardar recorte expandido sin recuadro
                img_crop_ext = img[y:y+h,x:x+w]
                
                # Distorsionar cara se multiplica longuitud cuadrado cara por nivel
                img_blur = cv2.blur(img,(int(h*blur_lvl),int(h*blur_lvl)))
                
                # Possar la zona amb la cara distorsionada
                img[y:y+h,x:x+w] = img_blur[y:y+h,x:x+w]
        
        #outfname = outputDir+"/"+image.split('/')[-1]
        outfname = outputDir+'/'+img_basename+'-'+str(nfaces)+'.'+img_ext
        # Bloquejar el thread per poder guardar la foto
        if mutex is not None:
            mutex.acquire()
        cv2.imwrite(outfname, img)
        
        # Desbloquejar tots els threads
        if mutex is not None:
            mutex.release()


# Funcion para distorsionar las caras blur_lvl comprente entre 0.0 a 1.0
def smoothFaces(inputDir,outputDir, blur_lvl = 0.3, scale_factor=1.1, min_neighbors=1,
                face_cascade_paths=['haarcascades/haarcascade_frontalface_alt2.xml', 'haarcascades/haarcascade_profileface.xml'],
                exts=['*.jpg','*.jpeg','*.png'],
                min_size=-1, max_size=-1,
                flags=cv2.CASCADE_SCALE_IMAGE):

    #Comprovar directorio donde estan las fotografias
    if os.path.exists(inputDir):
        
        # Comprovar si estan los directorios salida y crearlos en su defecto
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)
            
        # objecte locker per controlar els threads
        mutex = threading.Lock() #Per a threading
        ncpu = mp.cpu_count()
            
        # Iniciar variables
        threads, images, run_th = [],[],[]
        m, per = 0, -1 

        # read list of images
        for ext in exts:    
            images.extend(glob.glob(inputDir+'/'+ext))
        
        #numero de fotos
        n=len(images)
        
        print ""
        print " Distorsionant cares de " + str(n) + " Imatge(s) "
        print " Parametres: scale_factor " + str(scale_factor) + " min_neighbors " + str(min_neighbors) 
        if min_size == -1 :
            print '          minSize ( Automatic al 5% , Automatic 5% ) '
        else:
            print '          minSize ('+ min_size +' , '+ min_size +') '
            
        if max_size == -1:
            print '          maxSize (Automatic 35% , Automatic 35%)'
        else:
            print '          maxSize (' + max_size +','+ max_size +')'
        
        print " Filtres xml utilitzats:"
        for fil in face_cascade_paths:
            print " |--> " + fil
        print "------------------------------------------------------------------------------"
        
        for image in images:
            #imageSmooth(image,outputDir, None ,blur_lvl = blur_lvl ,scale_factor = scale_factor , min_neighbors = min_neighbors,face_cascade_paths=face_cascade_paths, exts=exts,flags=flags)
            try:
                # Es crea un fil perque un thread cerqui les cares per cada imatge, aixi es paral-lelitza amb millor eficiencia
                threads.append(threading.Thread(target=imageSmooth, 
                        args=(image,outputDir, mutex ,blur_lvl , scale_factor , min_neighbors,face_cascade_paths, exts, min_size, max_size, flags,)))
            except:
                print "Error desconocido del thread"
            
        #Preperar els treballs en grups del numero de cpus -1
        while len(threads) > 0:
            while len(run_th) < (ncpu+1) and len(threads) > 0:
                # Treure el thread de la llista per incorporar-lo en una de treball
                run_th.append(threads.pop())
            
            # Iniciar els grups de threads
            for th in run_th:
                th.start()
            # Esperar que acabin els threads    
            for th in run_th:
                if per < int((m/float(n))*100):
                    per = int((m/float(n))*100)
                    progressBar(per)
                th.join()
                m += 1
            # Vuidar la llista per incorporar nous threads    
            run_th =[]
                
        # Visualizar el final del proceso        
        progressBar(100)
        sys.stdout.write("\x1B[37m") # Canvia de color a normal
        print('\n')


# Barra Porcentaje
def progressBar(per):
    # colors (37 -> Blanc , 31 -> vermell, 32 -> verd, 33 -> tronja, 34 -> blau fosc, 35 -> lila, 36 -> Blau Clar, 30 -> negre)
    #colors = ("\x1B[37m","\x1B[31m","\x1B[32m","\x1B[33m","\x1B[34m","\x1B[35m","\x1B[36m") # Per la consola
    colors = ("\x1B[37m","\x1B[31m","\x1B[34m","\x1B[32m","\x1B[33m","\x1B[35m","\x1B[36m")
    sys.stdout.write('\r')
    sys.stdout.write(colors[random.randint(3,6)]) # Canvia de color
    sys.stdout.write("#"*int(per/2) + " [ "+ str(per) + " %] ")
    sys.stdout.flush()
    

