import cv2
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
from CannyEdge import canny_edge_detect

def start():
    # read image
    image = cv2.imread("Valve_original_wiki.png", 0)
    if image is None:
        print("No image available")
        return "No image available"

    #cv2 hough transform
    start = timer()
    cv2_hough_transform(image)
    end = timer()
    print("cv2 Hough Transform Total Time: ", end - start)

    image = cv2.imread("Valve_original_wiki.png", 0)
    #manual hough transform
    start = timer()
    lines, circles = manual_hough_transform(image)
    end = timer()
    print("Manual hough Transform Total Time: ", end - start)
    show_image(lines, "Manual Hough Lines")
    show_image(circles, "Manual Hough Circles")

def cv2_hough_transform(image):
    cv2_hough_lines(image)
    cv2_hough_circles(image)

def cv2_hough_lines(image):
    canny_edge = cv2.Canny(image, 50, 150, apertureSize=3)
    hough_lines = cv2.HoughLines(canny_edge, 1, np.pi / 180, 200)

    for r, theta in hough_lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * r
        y0 = b * r
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        #draw a line from x1,y1 to x2,y2
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('linesDetected.jpg', image)

def cv2_hough_circles(image):
    image = cv2.medianBlur(image, 5)
    hough_circle = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 100, param1=70, param2=60,
                                        minRadius=10, maxRadius=100)
    hough_circle = np.uint16(np.around(hough_circle))
    for i in hough_circle[0, :]:
        #outer circle
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        #middle of circle
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
    cv2.imshow('Circles', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def manual_hough_transform(image):
    manual_hough_lines(image)
    manual_hough_circles(image)

def manual_hough_lines(image):
    #use the same manual canny edge detection technique
    canny_edges = canny_edge_detect(image,10,100,1)

    # run hough_lines_accumulator on the shapes canny_edges image
    H, rhos, thetas = get_hough_accumulator(canny_edges)
    indicies = hough_simple_peaks(H, 3) # find peaks
    plot_hough_acc(H) # plot hough space, brighter spots have higher votes
    hough_lines_draw(image, indicies, rhos, thetas)

    # Show image with manual Hough Transform Lines
    cv2.imshow('Major Lines: Manual Hough Transform', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#calculate hough accumulator
def get_hough_accumulator(img, rho_resolution=1, theta_resolution=1):
    rows, cols = img.shape
    diagonal = np.ceil(np.sqrt(rows**2 + cols**2)) # a**2 + b**2 = c**2
    rhos = np.arange(-diagonal, diagonal + 1, rho_resolution)
    thetas = np.deg2rad(np.arange(-90, 90, theta_resolution))

    #empty hough accumulator array
    H = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img) # find all edge (nonzero) pixel indexes

    for i in range(len(x_idxs)): # cycle through edge points
        x = x_idxs[i]
        y = y_idxs[i]

        for j in range(len(thetas)): # cycle through thetas and calc rho
            rho = int((x * np.cos(thetas[j]) +
                       y * np.sin(thetas[j])) + diagonal)
            H[rho, j] += 1

    return H, rhos, thetas


#find peaks of accumulator
def hough_simple_peaks(H, num_peaks):
    indices =  np.argpartition(H.flatten(), -2)[-num_peaks:]
    return np.vstack(np.unravel_index(indices, H.shape)).T


def plot_hough_acc(H, plot_title='Hough Accumulator Plot'):
    fig = plt.figure(figsize=(10, 10))
    fig.canvas.set_window_title(plot_title)

    plt.imshow(H, cmap='jet')

    plt.xlabel('Theta Direction'), plt.ylabel('Rho Direction')
    plt.tight_layout()
    plt.show()


def hough_lines_draw(img, indexes, rhos, thetas):
    for i in range(len(indexes)):
        rho = rhos[indexes[i][0]]
        theta = thetas[indexes[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


def manual_hough_circles(image):
    file_path = './res/HoughCircles.jpg'
    img = imarray("Valve_original_wiki.png")
    res = smoothen(img,display=False)                                               #set display to True to display the edge image
    res = edge(res,128,display=False)                                               #set display to True to display the edge image
    #detectCircles takes a total of 4 parameters. 3 are required.
    #The first one is the edge image. Second is the thresholding value and the third is the region size to detect peaks.
    #The fourth is the radius range.
    res = detectCircles(res,8.1,15,radius=[100,10])
    displayCircles(res)

def smoothen(img,display):
    #Using a 3x3 gaussian filter to smoothen the image
    gaussian = np.array([[1/16.,1/8.,1/16.],[1/8.,1/4.,1/8.],[1/16.,1/8.,1/16.]])
    img.load(img.convolve(gaussian))
    if display:
        img.disp
    return img

def edge(img,threshold,display=False):
    #Using a 3x3 Laplacian of Gaussian filter along with sobel to detect the edges
    laplacian = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    #Sobel operator (Orientation = vertical)
    sobel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

    #Generating sobel horizontal edge gradients
    G_x = img.convolve(sobel)

    #Generating sobel vertical edge gradients
    G_y = img.convolve(np.fliplr(sobel).transpose())

    #Computing the gradient magnitude
    G = pow((G_x*G_x + G_y*G_y),0.5)

    G[G<threshold] = 0
    L = img.convolve(laplacian)
    if L is None:                                                               #Checking if the laplacian mask was convolved
        return
    (M,N) = L.shape

    temp = np.zeros((M+2,N+2))                                                  #Initializing a temporary image along with padding
    temp[1:-1,1:-1] = L                                                         #result hold the laplacian convolved image
    result = np.zeros((M,N))                                                    #Initializing a resultant image along with padding
    for i in range(1,M+1):
        for j in range(1,N+1):
            if temp[i,j]<0:                                                     #Looking for a negative pixel and checking its 8 neighbors
                for x,y in (-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1):
                    if temp[i+x,j+y]>0:
                        result[i-1,j-1] = 1                                 #If there is a change in the sign, it is a zero crossing
    img.load(np.array(np.logical_and(result,G),dtype=np.uint8))
    if display:
        img.disp
    return img

def detectCircles(img,threshold,region,radius = None):
    (M,N) = img.shape
    if radius == None:
        R_max = np.max((M,N))
        R_min = 3
    else:
        [R_max,R_min] = radius

    R = R_max - R_min
    #Initializing accumulator array.
    #Accumulator array is a 3 dimensional array with the dimensions representing
    #the radius, X coordinate and Y coordinate resectively.
    #Also appending a padding of 2 times R_max to overcome the problems of overflow
    A = np.zeros((R_max,M+2*R_max,N+2*R_max))
    B = np.zeros((R_max,M+2*R_max,N+2*R_max))

    #Precomputing all angles to increase the speed of the algorithm
    theta = np.arange(0,360)*np.pi/180
    edges = np.argwhere(img[:,:])                                               #Extracting all edge coordinates
    for val in range(R):
        r = R_min+val
        #Creating a Circle Blueprint
        bprint = np.zeros((2*(r+1),2*(r+1)))
        (m,n) = (r+1,r+1)                                                       #Finding out the center of the blueprint
        for angle in theta:
            x = int(np.round(r*np.cos(angle)))
            y = int(np.round(r*np.sin(angle)))
            bprint[m+x,n+y] = 1
        constant = np.argwhere(bprint).shape[0]
        for x,y in edges:                                                       #For each edge coordinates
            #Centering the blueprint circle over the edges
            #and updating the accumulator array
            X = [x-m+R_max,x+m+R_max]                                           #Computing the extreme X values
            Y= [y-n+R_max,y+n+R_max]                                            #Computing the extreme Y values
            A[r,X[0]:X[1],Y[0]:Y[1]] += bprint
        A[r][A[r]<threshold*constant/r] = 0

    for r,x,y in np.argwhere(A):
        temp = A[r-region:r+region,x-region:x+region,y-region:y+region]
        try:
            p,a,b = np.unravel_index(np.argmax(temp),temp.shape)
        except:
            continue
        B[r+(p-region),x+(a-region),y+(b-region)] = 1

    return B[:,R_max:-R_max,R_max:-R_max]

def displayCircles(A):
    img = imread(file_path)
    fig = plt.figure()
    plt.imshow(img)
    circleCoordinates = np.argwhere(A)                                          #Extracting the circle information
    circle = []
    for r,x,y in circleCoordinates:
        circle.append(plt.Circle((y,x),r,color=(1,0,0),fill=False))
        fig.add_subplot(111).add_artist(circle[-1])
    plt.show()

def show_image(image, title):
    plt.imshow(image, cmap="gray", aspect='auto')
    plt.title(title)
    plt.show()

def convolve(image, kernel):
    k_rows, k_cols = kernel.shape
    rows, cols = image.shape
    result = np.zeros(image.shape)

    vert = (k_rows - 1)
    hort= (k_cols - 1)
    padding_vert = int(vert / 2)
    padding_hort = int(hort / 2)

    padded_shape_vert = rows + (2 * padding_vert)
    padded_shape_hort = cols + (2 * padding_hort)
    padded = np.zeros((padded_shape_vert, padded_shape_hort))

    padded[padding_vert:padded.shape[0] - padding_vert, padding_hort:padded.shape[1] - padding_hort] = image

    for row in range(rows):
        for col in range(cols):
            result[row, col] = np.sum(kernel * padded[row:row + k_rows, col:col + k_cols])

    return result


if __name__ == '__main__':
    image = cv2.imread("Valve_original_wiki.png", 0)
    manual_hough_lines(image)
