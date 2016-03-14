import numpy as np
import scipy.signal


def make_gaussian_filter(sigma, N):
    """Creates a NxN Gaussian filter with standard deviation sigma."""
    filt = np.empty((N, N))
    center = (np.array(filt.shape) - 1) / 2.0
    cov = np.eye(2) * sigma
    logconst1 = -np.log(2 * np.pi)
    logconst2 = -np.linalg.slogdet(cov)[0]
    for i in range(filt.shape[0]):
        for j in range(filt.shape[1]):
            diff = np.array([i, j]) - center
            logexp = -0.5 * np.dot(diff, np.dot(np.linalg.inv(cov), diff))
            filt[i, j] = np.exp(logconst1 + logconst2 + logexp)
    return filt / np.sum(filt)


def apply_filter(img, filt, mode='same', trim=True):
    """Applies the given filter to the given image using convolution. If the
    image has RGB channels, then this applies the filter to each channel
    separately. The `mode` keyword is passed to the convolution function. The
    `trim` keyword specifies whether values should be trimmed to [0, 255] and
    converted to the uint8 datatype.

    """
    if img.ndim == 3:
        new_img = []
        for i in range(img.shape[2]):
            new_img.append(scipy.signal.convolve2d(
                img[:, :, i].astype('float'), filt, mode=mode))
        new_img = np.array(new_img).transpose((1, 2, 0))
    else:
        new_img = scipy.signal.convolve2d(
            img.astype('float'), filt, mode=mode)

    if trim:
        new_img[new_img < 0] = 0
        new_img[new_img > 255] = 255
        new_img = new_img.astype('uint8')

    return new_img


def make_oriented_filters(sigma=3, N=7):
    gauss = make_gaussian_filter(sigma, N)
    filters = []

    for theta in np.linspace(0, 2 * np.pi, 17)[:-1]:
        if 0 <= theta < (np.pi / 4):
            x = np.tan(theta)
            filt = np.array([[0, 0, x], [-1 + x, 0, 1 - x], [-x, 0, 0]])
        elif (np.pi / 4) <= theta < (np.pi / 2):
            x = np.tan(theta - (np.pi / 4))
            filt = np.array([[0, x, 1 - x], [0, 0, 0], [-1 + x, -x, 0]])
        elif (np.pi / 2) <= theta < (3 * np.pi / 4):
            x = np.tan(theta - (np.pi / 2))
            filt = np.array([[x, 1 - x, 0], [0, 0, 0], [0, -1 + x, -x]])
        elif (3 * np.pi / 4) <= theta < np.pi:
            x = np.tan(theta - (3 * np.pi / 4))
            filt = np.array([[1 - x, 0, 0], [x, 0, -x], [0, 0, -1 + x]])
        elif np.pi <= theta < (5 * np.pi / 4):
            x = np.tan(theta - np.pi)
            filt = np.array([[0, 0, -x], [1 - x, 0, -1 + x], [x, 0, 0]])
        elif (5 * np.pi / 4) <= theta < (3 * np.pi / 2):
            x = np.tan(theta - (5 * np.pi / 4))
            filt = np.array([[0, -x, -1 + x], [0, 0, 0], [1 - x, x, 0]])
        elif (3 * np.pi / 2) <= theta < (7 * np.pi / 4):
            x = np.tan(theta - (3 * np.pi / 2))
            filt = np.array([[-x, -1 + x, 0], [0, 0, 0], [0, 1 - x, x]])
        elif (7 * np.pi / 4) <= theta < (2 * np.pi):
            x = np.tan(theta - (7 * np.pi / 4))
            filt = np.array([[-1 + x, 0, 0], [-x, 0, x], [0, 0, 1 - x]])
        
        GD = apply_filter(gauss, filt, mode='same', trim=False)
        filters.append(GD)

    return np.array(filters)


def grating(N=7, w=1, orientation='horizontal'):
    g = np.zeros((N, N))
    if orientation in ('both', 'horizontal'):
        for i in range(w):
            g[i::(2*w)] = 1
            g[(i + w)::(2*w)] = -1
    if orientation in ('both', 'vertical'):
        for i in range(w):
            g[:, (i + w)::(2*w)] = -1    
        g[g == 0] = 1

    return g


def checkerboard(N=7, w=1):
    g = np.zeros((N, N))
    for ii, i in enumerate(range(0, N, w)):
        for jj, j in enumerate(range(0, N, w)):
            if (ii % 2) and (jj % 2):
                g[i:(i+w), j:(j+w)] = 1
            elif (ii % 2) and not (jj % 2):
                g[i:(i+w), j:(j+w)] = -1
            elif not (ii % 2) and (jj % 2):
                g[i:(i+w), j:(j+w)] = -1
            elif not (ii % 2) and not (jj % 2):
                g[i:(i+w), j:(j+w)] = 1
                
    return g


def colorize(filt, rgb):
    rgb = (np.array(rgb) * 2) - 1
    rgb_filt = np.empty((3,) + filt.shape)
    rgb_filt[0] = filt * rgb[0]
    rgb_filt[1] = filt * rgb[1]
    rgb_filt[2] = filt * rgb[2]
    return rgb_filt


def make_weights(N=7):
    weights = np.zeros((96, 3, N, N))

    # oriented filters
    filters = list(make_oriented_filters(N=N))
    mag = filters[0].max()
    # horizontal gratings
    filters.append(grating(N=N, w=1, orientation='horizontal') * mag)
    filters.append(grating(N=N, w=2, orientation='horizontal') * mag)
    filters.append(grating(N=N, w=1, orientation='horizontal') * -mag)
    filters.append(grating(N=N, w=2, orientation='horizontal') * -mag)
    # vertical gratings
    filters.append(grating(N=N, w=1, orientation='vertical') * mag)
    filters.append(grating(N=N, w=2, orientation='vertical') * mag)
    filters.append(grating(N=N, w=1, orientation='vertical') * -mag)
    filters.append(grating(N=N, w=2, orientation='vertical') * -mag)
    # crosshatch
    filters.append(grating(N=N, w=1, orientation='both') * mag)
    filters.append(grating(N=N, w=1, orientation='both') * -mag)
    # checkerboard
    filters.append(checkerboard(N=N, w=1) * mag)
    filters.append(checkerboard(N=N, w=2) * mag)
    filters.append(checkerboard(N=N, w=3) * mag)
    filters.append(checkerboard(N=N, w=1) * -mag)
    filters.append(checkerboard(N=N, w=2) * -mag)
    filters.append(checkerboard(N=N, w=3) * -mag)

    for i, rgb in enumerate([[1, 1, 1], [1, 0, 1], [1, 1, 0]]):
        for j, filt in enumerate(filters):
            weights[i * len(filters) + j] = colorize(filt, rgb)
            
    return weights


def make_lm_filters():
    F = scipy.io.loadmat("F.mat")['F'].transpose((2, 0, 1))[:, 12:-12, 12:-12]
    filts = np.empty((144, 3, 25, 25))
    for i, filt in enumerate(F):
        filts[i] = colorize(filt, [1, 1, 1])
        filts[i+48] = colorize(filt, [1, 0, 1])
        filts[i+96] = colorize(filt, [1, 1, 0])
    return filts
