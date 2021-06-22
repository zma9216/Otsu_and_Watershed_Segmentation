import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from skimage import io, img_as_ubyte, color
from skimage.filters import gaussian, laplace, threshold_otsu
from skimage.measure import find_contours

os.chdir(os.path.dirname(__file__))


def log_filter(img, s=3):
    """

    User-defined function for applying Laplacian of Gaussian filter
    on an image array with user-defined sigma value.

    :param ndarray img: Image array to be filtered
    :param int s: User chosen sigma value. Default is 3.
    :return: Outputs the filtered image array
    :rtype: ndarray
    """
    # Kernel size
    k = 2 * round(3 * s) + 1
    # Get proper truncate parameter for skimage function
    # SDJ and Haotao Wang - https://stackoverflow.com/a/56677500
    t = ((k - 1) / 2) / s

    # Apply Gaussian filter
    blur = gaussian(img, sigma=s, truncate=t)
    # Apply Laplacian filter to get LoG filtered image array
    log = laplace(blur, ksize=k)

    return log


def get_mask(new_img):
    """

    Function to detect regional minima within the Laplacian of Gaussian volume

    :param ndarray new_img: Laplacian of Gaussian Volume. 3D image array.
    :return: Returns binary image of the same size as the LoG volume
    :rtype: ndarray
    """
    # Get minimum filter of LoG volume
    # Size picked based on 3D image array pixel connectivity from
    # https://www.mathworks.com/help/images/pixel-connectivity.html
    min_filter = scipy.ndimage.minimum_filter(new_img, size=6)
    # Get mask with locations of the local minima
    mask = (new_img == min_filter)

    # Collapse 3D binary image into a single channel image
    mask = np.sum(mask, axis=-1)
    return mask


def refine_blobs(blur, mask):
    """

    Function to apply Otsu thresholding on the blurred image to obtain
    optimal threshold and remove all minima from image with rough blobs

    :param ndarray blur: Gaussian filtered image array
    :param ndarray mask: Mask with rough blob locations
    """
    # Convert to uint8
    img_blurred = img_as_ubyte(blur)
    # Get row and col of image
    row, col = img_blurred.shape
    # Get optimal threshold value
    thresh = threshold_otsu(img_blurred)
    for i in range(row):
        for j in range(col):
            # Get pixel of blurred image
            pixel = img_blurred[i, j]
            if pixel < thresh:
                # Remove minima where pixel values are less than the obtained threshold
                mask[i, j] = 0


def part1():
    # Read image file as grayscale
    file = 'img_A4_P1.bmp'
    img = io.imread(file, as_gray=True)

    # Get LoG filtered image arrays
    log_a = log_filter(img, s=3)
    log_b = log_filter(img, s=4)
    log_c = log_filter(img, s=5)

    # Create figure with 3 x 1 grid of subplots
    # Display LoG filtered image arrays
    fig1, ax1 = plt.subplots(nrows=3, ncols=1)
    ax1[0].imshow(log_a, cmap='jet')
    ax1[0].set_title('Level 1')

    ax1[1].imshow(log_b, cmap='jet')
    ax1[1].set_title('Level 2')

    ax1[2].imshow(log_c, cmap='jet')
    ax1[2].set_title('Level 3')

    plt.tight_layout()
    plt.show()

    # Create LoG volume with 3 levels
    new_img = np.dstack((log_a, log_b, log_c))
    mask = get_mask(new_img)

    # Show the locations of all non-zero entries in collapsed array overlaid on the input image as red points
    blobs = np.nonzero(mask)
    plt.scatter(blobs[1], blobs[0], c='r', marker='.')
    plt.imshow(img, cmap='jet'), plt.title('Rough Blobs Detected in Image')
    plt.gca().invert_yaxis()
    plt.show()

    # Apply Gaussian blur
    blur = gaussian(img, sigma=2, truncate=3)

    # Create figure with 2 x 1 grid of subplots
    # Display input image and blurred image
    fig2, ax2 = plt.subplots(nrows=2, ncols=1)
    ax2[0].imshow(img, cmap='jet')
    ax2[0].set_title('Input Image')

    ax2[1].imshow(blur, cmap='jet')
    ax2[1].set_title('Blurred image')

    plt.tight_layout()
    plt.show()

    # Refine blobs using Otsu thresholding and display all remaining
    # minima locations overlaid on the input image as red points
    refine_blobs(blur, mask)
    refined = np.nonzero(mask)
    plt.scatter(refined[1], refined[0], c='r', marker='.')
    plt.imshow(img, cmap='jet'), plt.title('Refined Blobs Detected in Image')
    plt.gca().invert_yaxis()
    plt.show()


def getSmallestNeighborIndex(img, row, col):
    min_row_id = -1
    min_col_id = -1
    min_val = np.inf
    h, w = img.shape
    for row_id in range(row - 1, row + 2):
        if row_id < 0 or row_id >= h:
            continue
        for col_id in range(col - 1, col + 2):
            if col_id < 0 or col_id >= w:
                continue
            if row_id == row and col_id == col:
                continue
            if img[row_id, col_id] < min_val:
                min_row_id = row_id
                min_col_id = col_id
                min_val = img[row_id, col_id]
    return min_row_id, min_col_id

# Function computes the local minima in the given image by comparing each pixel in the image
# to its 8-connected neighbours and marking it as a local minimum if its value is smaller than all of them
# Implementation based on advice and instruction given in assignment video.
def getRegionalMinima(img):
    regional_minima = np.zeros(img.shape, dtype=np.int32)
    h, w = img.shape
    marker = 1
    for i in range(h):
        for j in range(w):
            min_row, min_col = getSmallestNeighborIndex(img, i, j)
            if img[i, j] <= img[min_row, min_col]:
                regional_minima[i, j] = marker
                marker += 1

    return regional_minima


# Function uses the minimum following algorithm to label
# the unlabeled pixels in the marker image generated by the previous function.
# Implementation based on advice and instruction given in assignment video.
def iterativeMinFollowing(img, markers):
    markers_copy = np.copy(markers)
    h, w = img.shape
    path = True
    while path:
        n_unmarked_pix = 0
        for i in range(h):
            for j in range(w):
                if markers_copy[i, j] == 0:
                    min_row, min_col = getSmallestNeighborIndex(img, i, j)

                    if markers_copy[min_row, min_col] > 0:
                        markers_copy[i, j] = markers_copy[min_row, min_col]

                    if markers_copy[i, j] == 0:
                        n_unmarked_pix += 1

        print('n_unmarked_pix: ', n_unmarked_pix)
        if n_unmarked_pix == 0:
            path = False

    return markers_copy


def imreconstruct(marker, mask):
    curr_marker = (np.copy(marker)).astype(mask.dtype)
    kernel = np.ones([3, 3])
    while True:
        next_marker = cv2.dilate(curr_marker, kernel, iterations=1)
        intersection = next_marker > mask
        next_marker[intersection] = mask[intersection]
        # perform morphological reconstruction of the image marker under the image mask,
        # and returns the reconstruction in imresonctruct
        if np.array_equal(next_marker, curr_marker):
            return curr_marker
        curr_marker = next_marker.copy()

        return curr_marker


def imimposemin(marker, mask):
    # adapted from its namesake in MATLAB
    fm = np.copy(mask)
    fm[marker] = -np.inf
    fm[np.invert(marker)] = np.inf
    if mask.dtype == np.float32 or mask.dtype == np.float64:
        mask_range = float(np.max(mask) - np.min(mask))
        if range == 0:
            h = 0.1
        else:
            h = mask_range * 0.001
    else:
        # Add 1 to integer images.
        h = 1
    fp1 = mask + h
    g = np.minimum(fp1, fm)
    # If marker:-inf. Else:(||grad||+h)
    return np.invert(imreconstruct(np.invert(fm.astype(np.uint8)), np.invert(g.astype(np.uint8))).astype(np.uint8))


def part2():
    test_image = np.loadtxt('A4_test_image.txt')
    markers = getRegionalMinima(test_image)
    print(markers)
    labels = iterativeMinFollowing(test_image, markers)
    print(labels)

    sigma = 2.5
    img_name = 'img_A4_P1.bmp'
    img_rgb = io.imread(img_name).astype(np.float32)
    img_gs = color.rgb2gray(img_rgb)

    img_blurred = cv2.GaussianBlur(img_gs, (int(2 * round(3 * sigma) + 1), int(2 * round(3 * sigma) + 1)), sigma)
    # borderType=cv2.BORDER_REPLICATE

    [img_grad_y, img_grad_x] = np.gradient(img_blurred)
    img_grad = np.square(img_grad_x) + np.square(img_grad_y)

    # refined blob locations generated generated in part 3 of lab 6
    blob_markers = np.loadtxt('A4_blob_markers.txt', dtype=np.bool, delimiter='\t')

    img_grad_min_imposed = imimposemin(blob_markers, img_grad)

    markers = getRegionalMinima(img_grad_min_imposed)
    plt.figure(0)
    plt.imshow(markers, cmap='jet')
    plt.title('markers')

    labels = iterativeMinFollowing(img_grad_min_imposed, np.copy(markers))
    plt.figure(1)
    plt.imshow(labels, cmap='jet')
    plt.title('labels')

    # contour of img_grad_min_imposed
    contours = find_contours(img_grad_min_imposed, 0.8)
    contour_id = 0
    pruned_contours = []
    n_pruned_contours = 0

    fig, ax = plt.subplots()
    ax.imshow(img_grad_min_imposed, interpolation='nearest', cmap='gray')
    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


if __name__ == '__main__':
    part1()
    part2()
