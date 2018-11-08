from ipy_notebooks.ipyimports import *



def rayleightest(data, axis=None, weights=None):
    """ Performs the Rayleigh test of uniformity.

    This test is  used to identify a non-uniform distribution, i.e. it is
    designed for detecting an unimodal deviation from uniformity. More
    precisely, it assumes the following hypotheses:
    - H0 (null hypothesis): The population is distributed uniformly around the
    circle.
    - H1 (alternative hypothesis): The population is not distributed uniformly
    around the circle.
    Small p-values suggest to reject the null hypothesis.

    Parameters
    ----------
    data : numpy.ndarray or Quantity
        Array of circular (directional) data, which is assumed to be in
        radians whenever ``data`` is ``numpy.ndarray``.
    axis : int, optional
        Axis along which the Rayleigh test will be performed.
    weights : numpy.ndarray, optional
        In case of grouped data, the i-th element of ``weights`` represents a
        weighting factor for each group such that ``np.sum(weights, axis)``
        equals the number of observations.
        See [1]_, remark 1.4, page 22, for detailed explanation.

    Returns
    -------
    p-value : float or dimensionless Quantity
        p-value.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.stats import rayleightest
    >>> from astropy import units as u
    >>> data = np.array([130, 90, 0, 145])*u.deg
    >>> rayleightest(data) # doctest: +FLOAT_CMP
    <Quantity 0.2563487733797317>

    References
    ----------
    .. [1] S. R. Jammalamadaka, A. SenGupta. "Topics in Circular Statistics".
       Series on Multivariate Analysis, Vol. 5, 2001.
    .. [2] C. Agostinelli, U. Lund. "Circular Statistics from 'Topics in
       Circular Statistics (2001)'". 2015.
       <https://cran.r-project.org/web/packages/CircStats/CircStats.pdf>
    .. [3] M. Chirstman., C. Miller. "Testing a Sample of Directions for
       Uniformity." Lecture Notes, STA 6934/5805. University of Florida, 2007.
    .. [4] D. Wilkie. "Rayleigh Test for Randomness of Circular Data". Applied
       Statistics. 1983.
       <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.211.4762>
    """

    def _length(data, p=1, phi=0.0, axis=None, weights=None):
        # Utility function for computing the generalized sample length
        C, S = _components(data, p, phi, axis, weights)
        return np.hypot(S, C)

    def _components(data, p=1, phi=0.0, axis=None, weights=None):
        # Utility function for computing the generalized rectangular components
        # of the circular data.
        if weights is None:
            weights = np.ones((1,))
        try:
            weights = np.broadcast_to(weights, data.shape)
        except ValueError:
            raise ValueError('Weights and data have inconsistent shape.')

        C = np.sum(weights * np.cos(p * (data - phi)), axis) / np.sum(weights, axis)
        S = np.sum(weights * np.sin(p * (data - phi)), axis) / np.sum(weights, axis)

        return C, S

    n = np.size(data, axis=axis)
    Rbar = _length(data, 1, 0.0, axis, weights)
    z = n * Rbar * Rbar

    # see [3] and [4] for the formulae below
    tmp = 1.0
    if (n < 50):
        tmp = 1.0 + (2.0 * z - z * z) / (4.0 * n) - (24.0 * z - 132.0 * z ** 2.0 +
                                                     76.0 * z ** 3.0 - 9.0 * z ** 4.0) / (288.0 *
                                                                                          n * n)

    p_value = np.exp(-z) * tmp
    return p_value


def getCircularMean(angles):
    n = len(angles)
    sineMean = np.divide(np.sum(np.sin(np.radians(angles))), n)
    cosineMean = np.divide(np.sum(np.cos(np.radians(angles))), n)
    vectorMean = np.arctan2(sineMean, cosineMean)
    r = np.sqrt(np.square(sineMean) + np.square(cosineMean))
    return np.degrees(vectorMean), r


def polarHist(df, ax=None, bins=40, bottom=0, kind='bar', fname='', start=100, stepSize=40, alpha=0.6, s=50,
              toShow=True):
    r, t = np.histogram((df.trajectory__pOri_x % 360), bins=bins, density=False)
    #     r=r/float(max(r))
    width = (2 * np.pi) / len(r)

    t = np.deg2rad(t[:-1])

    if ax is None:
        ax = plt.subplot(111, polar=True)

    if kind == 'outline':
        bars = ax.plot(t, r)
    elif kind == 'scatterStack':
        # initialize empty arrays
        i = 0
        tl = np.array([])
        rl = np.array([])

        for the in t:
            currRl = range(start, r[i] + start, stepSize)
            currTl = np.repeat(the, len(currRl))
            rl = np.append(rl, currRl)
            tl = np.append(tl, currTl)
            i += 1

        ax.scatter((tl), rl, alpha=alpha, s=s)
        #         ax.set_theta_zero_location("N")
        ax.set_theta_offset(0)

    else:  # 'mostly standard bar'
        bars = ax.bar(t, r, width=width, bottom=bottom)

    tor = df.trajectory__wbad.abs().mean()
    pval = rayleightest(np.deg2rad(df.trajectory__pOri_x[::100]))
    meanAng, angDisp = getCircularMean(df.trajectory__pOri_x)

    ax.arrow(0, 0, np.deg2rad(meanAng % 360), angDisp * max(r), lw=3, )  # shape='full',
    # length_includes_head=True, head_width=100)


    tit = 'Mean Angle : ' + str(round(meanAng, 1)) + '\nAngular Dispersion : ' + \
          str(round(angDisp, 4)) + '\nP-value : ' + str(pval)
    ax.set_title(tit)
    print pval
    #     ax.set_title('tortuosity is'+str(round(tor,3)))

    #     ax.autoscale()
    #     figFname=fname.split('pickle')[0]+'polarHist.png'
    #     plt.savefig(figFname,transparency=True)

    # if toShow:
    #     plt.show()
    # else:
    #     plt.close()

    return r, t, ax


def polarPitoo(df, ax=None, maxAngle=15, initRadius=0, radIncrement=1. / 165, c='r', s=10, alpha=0.05, ):
    initH = df.trajectory__pOri_x[0]
    r = initRadius
    radius = []
    theta = []

    for i in range(len(df)):
        currH = df.trajectory__pOri_x[i]

        if abs(currH - initH) > maxAngle:
            initH = df.trajectory__pOri_x[i]
            r = initRadius
        else:
            r += radIncrement

        t = np.deg2rad(currH)
        radius.append(r)
        theta.append(t)

    if ax is None:
        ax = plt.subplot(111, projection='polar')

    # c = ax.scatter(theta, r, c='r', s=10, cmap='hsv', alpha=0.75)
    c = ax.scatter(theta, radius, c=c, s=s, alpha=alpha)

    plt.show()

    return radius, theta, c


def quadPolarHist(df=None, bins=40, bottom=0, kind='bar', fname='',
                  toShow=True, start=100, stepSize=40, alpha=0.6, s=50):
    if df is None:
        dfDump = pickle2df()
        df = dfDump[0]
        fname = dfDump[-1]

    # make a empty polar fig
    f, axarr = plt.subplots(2, 2, subplot_kw=dict(projection='polar'), figsize=(10, 12))

    # polar plots as quad
    d = polarHist(df[(df.trajectory__case == 0) & (df.trajectory__headingControl)],
                  ax=axarr[0, 1], bins=bins, bottom=bottom, kind=kind, start=start,
                  stepSize=stepSize, alpha=alpha, s=s, toShow=toShow)
    d = polarHist(df[(df.trajectory__case == 1) & (df.trajectory__headingControl)],
                  ax=axarr[0, 0], bins=bins, bottom=bottom, kind=kind, start=start,
                  stepSize=stepSize, alpha=alpha, s=s, toShow=toShow)
    d = polarHist(df[(df.trajectory__case == 2) & (df.trajectory__headingControl)],
                  ax=axarr[1, 0], bins=bins, bottom=bottom, kind=kind, start=start,
                  stepSize=stepSize, alpha=alpha, s=s, toShow=toShow)
    d = polarHist(df[(df.trajectory__case == 3) & (df.trajectory__headingControl)],
                  ax=axarr[1, 1], bins=bins, bottom=bottom, kind=kind, start=start,
                  stepSize=stepSize, alpha=alpha, s=s, toShow=toShow)

    # set lims on min/max
    ylims = []
    for row, col in axarr:
        ylims.append(row.get_ylim()[1])
        ylims.append(col.get_ylim()[1])
    yMax = max(ylims)
    yMin = 0
    for row, col in axarr:
        row.set_ylim(yMin, yMax)
        col.set_ylim(yMin, yMax)

    figFname = fname.split('pickle')[0] + kind + 'polarHist.png'
    f.suptitle(fname.split('/')[-1], fontsize=12)
    plt.savefig(figFname, transparency=True, dpi=300)
    # if toShow:
    #     plt.show()
    # else:
    #     plt.close()

    return axarr


def pickleDir2Polar(foldPath=None, bins=40, bottom=0, kind='bar', toShow=True,
                    start=100, stepSize=40, alpha=0.6, s=50):
    filenames = fileList( foldPath=foldPath,defDir='/media/rhagoletis/Traj/')


    for i,filename in enumerate(filenames):
        try:
            print ('\n\nfile {0} of {1}'.format(i,len(filenames))),
            print "analysing file ",filename.split('/')[-1]
            dfDump = pickle2df(path=filename)


            df = dfDump[0]
            fname = dfDump[-1]

            a = quadPolarHist(df, bins, bottom, kind, fname, toShow)
            plt.close()
        except Exception as e:
            print e
            pass
#
# tic()
# a=pickleDir2Polar(kind='scatterStack',start=500,stepSize=130,alpha=0.6,s=50,toShow=False)
# toc()

from joblib import Parallel, delayed
import multiprocessing

filenames=fileList()
# inputs = range(10)
tic()

num_cores = multiprocessing.cpu_count()

def processInput(filename,bins=40, bottom=0, kind='scatterStack', toShow=True,
                    start=500, stepSize=130, alpha=0.6, s=50):
    dfDump=pickle2df(path=filename)
    df=dfDump[0]
    fname=dfDump[-1]
    a=quadPolarHist(df,bins,bottom,kind,fname,toShow)
    plt.close()
results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in filenames)

toc()