from PIL import Image
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import data
from skimage import io as skio
from skimage.filters import threshold_otsu
import largestinteriorrectangle as lir

image = Image.open("dataset/ISIC_0000146.jpg")
new_image = image.resize((256, 256))
new_image.save("ISIC_0000146_test.jpg")

img = cv.imread("ISIC_0000146_test.jpg")
size = img.shape
img = img[10:size[0]-10, 10:size[1]-10]
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, mask = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)
#cv.imwrite(
contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
contour = np.array([contours[0][:, 0, :]])
print(contour)
inner_bb = lir.lir(contour)
cropped_img = img[inner_bb[1]:inner_bb[1] + inner_bb[3],inner_bb[0]:inner_bb[0] + inner_bb[2]]
cropped_img = cv.resize(cropped_img, (256, 256))
cv.imwrite("foreground.jpg", cropped_img)

path = 'foreground.jpg'
image= cv.imread(path, cv.IMREAD_COLOR)
# DULL RAZOR (REMOVE HAIR)
# Gray scale
grayScale = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
# Black hat filter
kernel = cv.getStructuringElement(1, (9, 9))
blackhat = cv.morphologyEx(grayScale, cv.MORPH_BLACKHAT, kernel)
# Gaussian filter
bhg = cv.GaussianBlur(blackhat, (3, 3), cv.BORDER_DEFAULT)
# Binary thresholding (MASK)
ret, mask = cv.threshold(bhg, 10, 255, cv.THRESH_BINARY)
# Replace pixels of the mask
dst = cv.inpaint(image, mask, 6, cv.INPAINT_TELEA)
cv.imwrite('ISIC_0000146_clean.jpg', dst)


def histogram(im):
    
    nl,nc=im.shape
    
    hist=np.zeros(256)
    
    for i in range(nl):
        for j in range(nc):
            hist[im[i][j]]=hist[im[i][j]]+1
            
    for i in range(256):
        hist[i]=hist[i]/(nc*nl)
        
    return(hist)
    
def otsu_thresh(im):
    
    h=histogram(im)
    
    m=0
    for i in range(256):
        m=m+i*h[i]
    
    maxt=0
    maxk=0
    
    
    for t in range(256):
        w0=0
        w1=0
        m0=0
        m1=0
        for i in range(t):
            w0=w0+h[i]
            m0=m0+i*h[i]
        if w0 > 0:
            m0=m0/w0
        
        for i in range(t,256):
            w1=w1+h[i]
            m1=m1+i*h[i]
        if w1 > 0:   
            m1=m1/w1
        
        k=w0*w1*(m0-m1)*(m0-m1)    
        
        if k > maxk:
            maxk=k
            maxt=t
            
            
    thresh=maxt
        
    return(thresh)

image = skio.imread('ISIC_0000146_clean.jpg')
image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
thresh = threshold_otsu(image_gray)
thresh_multiplier = 1.05 
binary = image_gray > thresh_multiplier * thresh

thresh=otsu_thresh(image_gray)
binary = image_gray > thresh_multiplier * thresh

inverted_image = np.invert(binary)
#Convert to uint8 and save the result
inverted_image_uint8 = (255 * inverted_image).astype(np.uint8)
cv.imwrite('ISIC_0000146_seg.jpg', inverted_image_uint8)


# Charger les deux images à comparer
image_base = cv.imread('dataset/ISIC_0000146.jpg', cv.IMREAD_COLOR)
base_resize = cv.resize(image_base, (256, 256))  # Redimensionner pour avoir la même hauteur que les images binaires
image_prof = cv.imread('dataset/ISIC_0000146_Segmentation.jpg', cv.IMREAD_COLOR)
image_etudiant = cv.imread('ISIC_0000146_seg.jpg', cv.IMREAD_COLOR)
prof_resize = cv.resize(image_prof, (256, 256))
# Convertir les images en binaire
_, image_prof_bin = cv.threshold(prof_resize, 128, 255, cv.THRESH_BINARY)
_, image_etudiant_bin = cv.threshold(image_etudiant, 128, 255, cv.THRESH_BINARY)
# Calculer l'indice de Sørensen-Dice
intersection = np.sum(np.logical_and(image_prof_bin, image_etudiant_bin))
union = np.sum(np.logical_or(image_prof_bin, image_etudiant_bin))
sorensen_dice = 2.0 * intersection / (intersection + union)
# Créer une image composée des trois images côte à côte
composite_image = np.concatenate((base_resize, image_etudiant_bin, image_prof_bin), axis=1)
cv.imwrite('comparaison_image.jpg', composite_image)
# Afficher la valeur de l'indice de Dice sur l'image composite
plt.imshow(cv.cvtColor(composite_image, cv.COLOR_BGR2RGB))
plt.title(f"Indice de Dice : {sorensen_dice:.2f}")
plt.axis('off')
plt.show()
