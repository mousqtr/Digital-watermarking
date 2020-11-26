import numpy as np
import cv2
import matplotlib.pyplot as plt

I = plt.imread("cameraman.tif")

plt.figure(1)
plt.imshow(I, cmap="gray") #colormap binary
plt.title("Image originale")
plt.show()

TFD2_I = np.fft.fft2(I)
TFD2_I_centree = np.fft.fftshift(TFD2_I)



plt.figure(2)
np.seterr(divide = 'ignore')
plt.subplot(221)
plt.title("Partie reelle de sa TFD2")
plt.imshow(np.log(abs(np.real(TFD2_I))))

plt.subplot(222)
plt.title("Partie imaginaire de sa TFD2")
plt.imshow(np.log(abs(np.imag(TFD2_I))))

plt.subplot(223)
plt.title("Partie reelle de sa TFD2 centree")
plt.imshow(np.log(abs(np.real(TFD2_I_centree))))

plt.subplot(224)
plt.title("Partie imaginaire de sa TFD2 centree")
plt.imshow(np.log(abs(np.imag(TFD2_I_centree))))

plt.show()


plt.figure(3)

plt.subplot(121)
plt.title("Module de sa TFD2")
plt.imshow(np.log(np.abs(TFD2_I)))

plt.subplot(122)
plt.title("Module de sa TFD2 centree")
plt.imshow(np.log(np.abs(TFD2_I_centree)))
plt.show()

# verification des proprietes de la TF

arg_tfd2_centree = np.angle(TFD2_I_centree)

# verifier que l'arg est congrue a pi modulo pi

# question 3
K = 1024
T = np.random.rand(K,1)
# print(T)


def insertion(TFD,tatouage,alpha,nb):

    h = len(TFD)
    w = len(TFD[0])

    x0 = h // 2
    y0 = w // 2


    k=0
    k_fin=len(tatouage)
    for y in range(y0+1,y0+32): # rectangle en bas a droite (129,129)
        for x in range(x0+1,x0+32):

            TFD[y][x]=TFD[y][x]*(1+alpha*tatouage[k])

            k=k+1
    k=0
    for y2 in range(y0-1,y0-32,-1): #rectangle en haut a gauche (127,127)
        for x2 in range(x0-1,x0-32,-1):
            TFD[y2][x2]=TFD[y2][x2]*(1+alpha*tatouage[k])
            k=k+1


    return TFD


a = 0.05 # Valeur de alpha
TFD_tatouee = insertion(TFD2_I_centree,T,a,K)

# affichage du module de l'image tatoue
plt.figure(4)

plt.subplot(121)
plt.title("Module de sa TFD2")
plt.imshow(np.log(np.abs(TFD2_I)))

plt.subplot(122)
plt.title("Module de sa TFD2 tatoue")
plt.imshow(np.log(np.abs(TFD_tatouee)))

plt.show()


TFD_tatouee_inverse = np.fft.fftshift(TFD_tatouee)
image_tatouee_inverse = np.fft.ifft2(TFD_tatouee_inverse)

# Affichage de l'image tatoue et non tatoue
plt.figure(5)


plt.subplot(121)
plt.title("Image originale")
plt.imshow(I,cmap = 'gray')

plt.subplot(122)
plt.title("Image tatoue")
plt.imshow(np.real(image_tatouee_inverse),cmap = 'gray')
plt.show()

MAXf = np.amax(I)
print(MAXf)


# Calcul du PSNR

list_pn = [ -x for x in range(1, 10)]

list_alpha = []

# on calcul les puissance negative de 10 pour avoir des valeurs tres petites de alpha
for k in range(len(list_pn)):
    x = pow(10, list_pn[k])
    list_alpha.append(x)

# on remet la liste alpha dans l'ordre croissant
list_alpha.sort()

list_PSNR = []


def PSNR(im, TFD2Ic, alpha_i):
    DIM = TFD2_I_centree.shape
    N = DIM[0]*DIM[1]
    MSE = 0

    # calcul des tatouages et retour a l'image tatoue
    TFD_tatouee = insertion(TFD2Ic, T, alpha_i, K)
    TFD_tatouee_inverse = np.fft.fftshift(TFD_tatouee)
    image_tatouee_inverse = np.fft.ifft2(TFD_tatouee_inverse)
    MSE = ((im-np.real(image_tatouee_inverse)) ** 2).mean()

    return (10*(np.log10((MAXf**2)/MSE)))


# Chargement des valeurs du psnr et trace de la courbe

for k in range (len(list_alpha)) :
    x = PSNR(I,TFD2_I_centree,list_alpha[k])
    list_PSNR.append(x)

plt.figure(6)
plt.title("PSNR en fonction de alpha")
plt.plot(list_alpha,list_PSNR)
plt.show()

# Partie detection et et calcul de la correlation du tatouage

def detection(seuil, IT, image):

    TFD2_I = np.fft.fft2(image)
    TFD2_I_centree = np.fft.fftshift(TFD2_I)
    # coordonnee du centre et extraction des deux carres du tatouage

    h = len(TFD2_I)
    w = len(TFD2_I)

    x0 = h // 2
    y0 = w // 2

    new_T = np.zeros((h, w))  # creation d'une matrice nulle

    k=0

    # rectangle en bas a droite (129,129)
    for y in range(y0+1,y0+32):
        for x in range(x0+1,x0+32):
            new_T[y][x] =np.real(TFD2_I_centree[y][x])
            k=k+1
    k=0

    # rectangle en haut a gauche (127,127)
    for y2 in range(y0-1,y0-32,-1):
        for x2 in range(x0-1,x0-32,-1):
            new_T[y2][x2]=np.real(TFD2_I_centree[y2][x2])
            k=k+1

    gamma1 = 0
    gamma2 = 0
    k = len(new_T[0])

    # calcul de la correlation des images
    for i in range(0, k-1):
        for j in range(0, k-1):
            gamma1 += -((new_T[i, j] - new_T.mean()) * (IT[i, j] - IT.mean()))
            gamma2 += ((new_T[i, j] - new_T.mean()) ** 2) * ((IT[i, j] - IT.mean()) ** 2)
    gamma = gamma1 / np.sqrt(gamma2)  # calcul du coeff de correlation entre le tatouage et l'image

    gamma = 100-round(np.real(gamma)*pow(10,8))

    print(gamma)

    if gamma < seuil:
        print('false')
        return False
    else:
        print('true')
        return True


# Detection du tatouage

detection(65,TFD_tatouee,image_tatouee_inverse)

#  Nouveau tatouage et test de la correlation

T1 = np.cos(np.random.rand(K, 1)) # Nouveau tatouage

TFD2_I2 = np.fft.fft2(I)
TFD2_I_centree2 = np.fft.fftshift(TFD2_I2)

TFD2_I2_tatoue = insertion(TFD2_I_centree2,T1,a,K)

detection(65,TFD2_I2_tatoue , image_tatouee_inverse)

