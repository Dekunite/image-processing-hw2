import math
import cv2
import matplotlib.pyplot as plt
from convolution import convolution
import numpy as np
from timeit import default_timer as timer

def start():
    # read image
    image = cv2.imread("Valve_original_wiki.png", 0)
    if image is None:
        print("No image available")
        return "No image available"

    #cv2 sobel filters
    start = timer()
    cv2_hough_transform(image)
    end = timer()
    print("cv2 Sobel Total Time: ", end - start)

    image = cv2.imread("Valve_original_wiki.png", 0)
    #manual sobel filters
    start = timer()
    vertical, horizontal = manual_hough_transform(image)
    end = timer()
    print("Manual Sobel Total Time: ", end - start)
    show_image(vertical, "Manual Sobel Vertical")
    show_image(horizontal, "Manual Sobel Horizontal")

def cv2_hough_transform(image):
    cv2_hough_lines(image)
    cv2_hough_circles(image)

def cv2_hough_lines(image):
    # Apply edge detection method on the image
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    show_image(edges, "edege")

    # This returns an array of r and theta values
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    # The below for loop runs till r and theta values
    # are in the range of the 2d array
    for r, theta in lines[0]:
        # Stores the value of cos(theta) in a
        a = np.cos(theta)

        # Stores the value of sin(theta) in b
        b = np.sin(theta)

        # x0 stores the value rcos(theta)
        x0 = a * r

        # y0 stores the value rsin(theta)
        y0 = b * r

        # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        x1 = int(x0 + 1000 * (-b))

        # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        y1 = int(y0 + 1000 * (a))

        # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        x2 = int(x0 - 1000 * (-b))

        # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        y2 = int(y0 - 1000 * (a))

        # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
        # (0,0,255) denotes the colour of the line to be
        # drawn. In this case, it is red.
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # All the changes made in the input image are finally
    # written on a new image houghlines.jpg
    cv2.imshow('linesDetected.jpg', image)

def cv2_hough_circles(image):
    image = cv2.medianBlur(image, 5)
    cimg = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    houghImageCircle = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 100, param1=70, param2=60,
                                        minRadius=10, maxRadius=100)

    circles = np.uint16(np.around(houghImageCircle))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    # show_image(cimg, "lines")
    cv2.imshow('detected circles', cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def manual_hough_transform(image):
    manual_hough_lines(image)
    manual_hough_circles(image)

def manual_hough_lines(image):
    # Step 2: Hough Space

    img_shape = image.shape

    x_max = img_shape[0]
    y_max = img_shape[1]

    theta_max = 1.0 * math.pi
    theta_min = 0.0

    r_min = 0.0
    r_max = math.hypot(x_max, y_max)

    r_dim = 200
    theta_dim = 300

    hough_space = np.zeros((r_dim,theta_dim))

    for x in range(x_max):
        for y in range(y_max):
            if image[x,y,0] == 255: continue
            for itheta in range(theta_dim):
                theta = 1.0 * itheta * theta_max / theta_dim
                r = x * math.cos(theta) + y * math.sin(theta)
                ir = r_dim * ( 1.0 * r ) / r_max
                hough_space[ir,itheta] = hough_space[ir,itheta] + 1

    plt.imshow(hough_space, origin='lower')
    plt.xlim(0,theta_dim)
    plt.ylim(0,r_dim)

    tick_locs = [i for i in range(0,theta_dim,40)]
    tick_lbls = [round( (1.0 * i * theta_max) / theta_dim,1) for i in range(0,theta_dim,40)]
    plt.xticks(tick_locs, tick_lbls)

    tick_locs = [i for i in range(0,r_dim,20)]
    tick_lbls = [round( (1.0 * i * r_max ) / r_dim,1) for i in range(0,r_dim,20)]
    plt.yticks(tick_locs, tick_lbls)

    plt.xlabel(r'Theta')
    plt.ylabel(r'r')
    plt.title('Hough Space')

    plt.savefig("hough_space_r_theta.png",bbox_inches='tight')

    plt.close()

    #----------------------------------------------------------------------------------------#
    # Find maximas 1
    Sorted_Index_HoughTransform =  np.argsort(hough_space, axis=None)

    #print Sorted_Index_HoughTransform.shape, r_dim * theta_dim

    shape = Sorted_Index_HoughTransform.shape

    k = shape[0] - 1
    list_r = []
    list_theta = []
    for d in range(5):
        i = int( Sorted_Index_HoughTransform[k] / theta_dim )
        #print i, round( (1.0 * i * r_max ) / r_dim,1)
        list_r.append(round( (1.0 * i * r_max ) / r_dim,1))
        j = Sorted_Index_HoughTransform[k] - theta_dim * i
        list_theta.append(round( (1.0 * j * theta_max) / theta_dim,1))
        k = k - 1


    #theta = list_theta[7]
    #r = list_r[7]

    #print " r,theta",r,theta, math.degrees(theta)
    #----------------------------------------------------------------------------------------#
    # Step 3: Find maximas 2

    import scipy.ndimage.filters as filters
    import scipy.ndimage as ndimage

    neighborhood_size = 20
    threshold = 140

    data_max = filters.maximum_filter(hough_space, neighborhood_size)
    maxima = (hough_space == data_max)


    data_min = filters.minimum_filter(hough_space, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)

    x, y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1)/2
        y.append(y_center)

    plt.imshow(hough_space, origin='lower')
    plt.savefig('hough_space_i_j.png', bbox_inches = 'tight')

    plt.autoscale(False)
    plt.plot(x,y, 'ro')
    plt.savefig('hough_space_maximas.png', bbox_inches = 'tight')

    plt.close()

    #----------------------------------------------------------------------------------------#
    # Step 4: Plot lines

    line_index = 1

    for i,j in zip(y, x):

        r = round( (1.0 * i * r_max ) / r_dim,1)
        theta = round( (1.0 * j * theta_max) / theta_dim,1)

        fig, ax = plt.subplots()

        ax.imshow(image)

        ax.autoscale(False)

        px = []
        py = []
        for i in range(-y_max-40,y_max+40,1):
            px.append( math.cos(-theta) * i - math.sin(-theta) * r )
            py.append( math.sin(-theta) * i + math.cos(-theta) * r )

        ax.plot(px,py, linewidth=10)

        plt.savefig("image_line_"+ "%02d" % line_index +".png",bbox_inches='tight')

        #plt.show()

        plt.close()

        line_index = line_index + 1

    #----------------------------------------------------------------------------------------#
    # Plot lines
    i = 11
    j = 264

    i = y[1]
    j = x[1]

    print(i,j)

    r = round( (1.0 * i * r_max ) / r_dim,1)
    theta = round( (1.0 * j * theta_max) / theta_dim,1)

    print('r', r)
    print('theta', theta)


    fig, ax = plt.subplots()

    ax.imshow(image)

    ax.autoscale(False)

    px = []
    py = []
    for i in range(-y_max-40,y_max+40,1):
        px.append( math.cos(-theta) * i - math.sin(-theta) * r )
        py.append( math.sin(-theta) * i + math.cos(-theta) * r )

    print(px)
    print(py)

    ax.plot(px,py, linewidth=10)

    plt.savefig("PlottedLine_07.png",bbox_inches='tight')

    plt.show()
def manual_hough_circles(image):
    print("circ")

def show_image(image, title):
    plt.imshow(image, cmap="gray", aspect='auto')
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    start()
