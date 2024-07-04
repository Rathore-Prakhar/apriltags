import cv2
import numpy as np

img = cv2.imread('pic.jpg')
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_orange = np.array([6, 100, 100])
upper_orange = np.array([20, 255, 255])
img_threshold = cv2.inRange(img_hsv, lower_orange, upper_orange)

contours, _ = cv2.findContours(img_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def fit_ellipse(x, y):
    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = np.dot(D1.T, D1)
    S2 = np.dot(D1.T, D2)
    S3 = np.dot(D2.T, D2)
    T = -np.dot(np.linalg.inv(S3), S2.T)
    M = S1 + np.dot(S2, T)
    M = np.array([M[2, :] / 2, -M[1, :], M[0, :] / 2])
    eigval, eigvec = np.linalg.eig(M)
    cond = 4 * eigvec[0] * eigvec[2] - eigvec[1]**2
    a1 = eigvec[:, cond > 0]
    A = np.vstack([a1, np.dot(T, a1)]).flatten()
    return A

def cart_to_pol(coeffs):
    if len(coeffs) == 0:
        return

    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    det = b**2 - a*c

    x0 = (c*d - b*f) / det
    y0 = (a*f - b*d) / det

    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)

    ap = np.sqrt(num / det / (fac - a - c))
    bp = np.sqrt(num / det / (-fac - a - c))

    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    r = (bp/ap)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)

    if b == 0:
        phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        phi += np.pi/2

    return x0, y0, ap, bp, e, phi

def get_ellipse_pts(params, npts=50, tmin=0, tmax=2*np.pi):
    if params is None:
        return

    x0, y0, ap, bp, e, phi = params
    t = np.linspace(tmin, tmax, npts)
    x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return x, y

for contour in contours:
    if len(contour) >= 5:
        x = contour[:, 0, 0]
        y = contour[:, 0, 1]

        coeffs = fit_ellipse(x, y)
        params = cart_to_pol(coeffs)

        if params is not None:
            x_pts, y_pts = get_ellipse_pts(params)
            for (x_pt, y_pt) in zip(x_pts, y_pts):
                cv2.circle(img, (int(x_pt), int(y_pt)), 2, (0, 255, 0), -1)

cv2.imshow('Detected Ellipses', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
