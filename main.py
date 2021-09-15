import numpy as np
import cv2
import copy
from skimage import img_as_float
import os
import moviepy.video.io.ImageSequenceClip
import sys

init = []

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', img)
        init.append([x, y])


def external_energy(img_grd, point_i):
    return -img_grd[point_i[1], point_i[0]]


def internal_energy(point_i_mines, point_i, point_i_plus,
                    alpha, beta,
                    d0, use_d0 = True, coefficient_d = 0.05):
    if (use_d0):
        d = coefficient_d * d0
    else:
        d = 0
    x1 = (np.sqrt((point_i_plus[0] - point_i[0]) ** 2 +
                  (point_i_plus[1] - point_i[1]) ** 2) - d) ** 2
    x2 = (point_i_plus[0] - 2 * point_i[0] + point_i_mines[0]) ** 2 + \
         (point_i_plus[1] - 2 * point_i[1] + point_i_mines[1]) ** 2
    return alpha*np.sqrt(x1) + beta*np.sqrt(x2)


def find_contour(img1, img_grd0, points0, gama=100,
                 iterations=10, alpha=0.7, beta=0.3,
                 full_iterations=True, min_dist=0.01):
    points = copy.deepcopy(points0)
    img_grd = copy.deepcopy(img_grd0)
    a, b = alpha, beta
    min_E_prev = np.inf
    img = img1
    for iter in range(iterations):
        contour_points = np.concatenate((points, [points[0, :]]), axis=0)
        d0 = 0
        for i in range(len(contour_points[:, 0]) - 1):
            d0 += ((contour_points[i, 0] - contour_points[i + 1, 0]) ** 2 +
                   (contour_points[i, 1] - contour_points[i + 1, 1]) ** 2)
        d0 = np.sqrt(d0)
        d0 = d0 / (len(contour_points[:, 0]) - 1)

        contour_points = np.concatenate(([points[-2, :]], contour_points), axis=0)
        E = np.zeros([25])
        best_prev = np.int_(np.zeros([25, len(contour_points[:, 0])]))
        for n in range(2, len(contour_points[:, 0])):
            E0 = copy.deepcopy(E)
            for m in range(25):
                point_i_plus = np.array([0, 0])
                point_i_plus[0], point_i_plus[1] = contour_points[n, 0] + ((m // 5) - 2), \
                                                   contour_points[n, 1] + ((m % 5) - 2)
                point_i_mines = contour_points[n - 2, :]
                min_E_m = np.inf
                for m1 in range(25):
                    point_i = contour_points[n - 1, :]
                    point_i[0], point_i[1] = contour_points[n - 1, 0] + ((m1 // 5) - 2), \
                                             contour_points[n - 1, 1] + ((m1 % 5) - 2)
                    external_E = external_energy(img_grd, point_i)
                    internal_E = internal_energy(point_i_mines, point_i, point_i_plus, a, b,  d0)

                    E_m = gama * external_E + internal_E
                    if (E0[m] + E_m < min_E_m):
                        min_E_m = E0[m] + E_m
                        E[m] = E0[m] + E_m
                        best_prev[m, n] = m1
        min_E_m = np.min(E)
        ind = np.array(np.where(E==min_E_m))
        index = ind[0, 0]
        for i in range(len(contour_points[:, 0])-1 , 1, -1):
            index = best_prev[index, i]
            contour_points[i-1, 0], contour_points[i-1, 1] = contour_points[i-1, 0] + (index // 5 - 2), \
                                                         contour_points[i-1, 1] + (index % 5 - 2)
        points[:, :] = contour_points[1:len(contour_points[:, 0]) - 1, :]

        points_i = np.concatenate((np.copy(points), [points[0, :]]), axis=0)
        img = np.copy(img1)
        file = sys.argv[0]
        dirname = os.path.dirname(file)
        dirname = dirname + '/' + 'pic'
        path = dirname
        for i in range(len(points_i[:, 0]) - 1):
            start_p = tuple(points_i[i, :])
            end_p = tuple(points_i[i + 1, :])
            img = cv2.line(img, start_p, end_p, [0, 255, 0], 1)
            cv2.imwrite(os.path.join(path, '{i}.png'.format(i = iter)), img)
        if ((np.abs(min_E_prev - min_E_m) < min_dist) & (full_iterations==False)):
            return img
        min_E_prev = copy.deepcopy(min_E_m)
    return img

file = sys.argv[0]
dirname = os.path.dirname(file)
dirname = dirname + '/' + 'pic'

if os.path.exists(dirname)==0:
    os.mkdir(dirname)

img = cv2.imread('tasbih.jpg')
# img = data.astronaut()
img2 = np.copy(img)
length = int((img.shape[0] / 4))
width = int((img.shape[1] / 4))
img = cv2.resize(img, (width, length), interpolation=cv2.INTER_AREA)
img0 = np.copy(img)
img1 = np.copy(img)

img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
img_grd = cv2.Canny(img0, 100, 200)
# cv2.imwrite('sobellll.jpg', img_grd.astype(np.uint8))
img_grd = img_as_float(img_grd)-500

cv2.imshow('image', img)
cv2.setMouseCallback('image', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
init = np.array(init)
# initial_points = init
center = init[0, :]
point_on_circle = init[1, :]
R = np.sqrt((center[0] - point_on_circle[0]) ** 2 + (center[1] - point_on_circle[1]) ** 2)
s = np.linspace(0, 2 * np.pi * (1 - 1 / 30), 29)
r = center[0] + R * np.sin(s)
c = center[1] + R * np.cos(s)
initial_points = np.int_(np.array([r, c]).T)
points = initial_points

img_o = find_contour(img1, img_grd, points, iterations=50, gama=1000, alpha=10, beta=1)

image_folder= dirname
fps=3
image_files = []
num_of_img = len([img for img in os.listdir(image_folder) if img.endswith(".png")])
for i in range(num_of_img):
    image_files.append(image_folder + '/' + str(i) + ".png")
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile('contour.mp4')

img_o = cv2.resize(img_o, (img2.shape[1], img2.shape[0]), interpolation=cv2.INTER_AREA)
x, y = np.where((img_o[:, :, 0] == 0) & (img_o[:, :, 1] == 255) & (img_o[:, :, 2] == 0))
img2[x, y, 0], img2[x, y, 1], img2[x, y, 2] = 0, 255, 0
cv2.imwrite('res09.jpg', img2)
