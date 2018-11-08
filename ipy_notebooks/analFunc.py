# %matplotlib inline
# import mpld3
# mpld3.enable_notebook()
import rosbag_pandas
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import easygui
import rosbag
import json
import numpy as np
import cPickle as pkl
import time
import seaborn as sns
import datetime as dt
import fnmatch
import os
import bokeh.palettes as bp
from bokeh.layouts import column
from bokeh.models import Arrow, OpenHead, NormalHead, VeeHead, Div
from bokeh.io import export_png
from bokeh.plotting import figure, output_file, show, gridplot, save
from datetime import datetime


def TicTocGenerator():
    # Generator that returns time differences
    ti = 0  # initial time
    tf = time.time()  # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf - ti  # returns the time difference


TicToc = TicTocGenerator()  # create an instance of the TicTocGen generator


# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print("Elapsed time: %f seconds.\n" % tempTimeInterval)


def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)


def pickler(obj, path):
    """
    Pickle a Python object
    """
    with open(path, "wb") as pfile:
        pkl.dump(obj, pfile)


def depickler(path):
    """
    Extracts a pickled Python object and returns it
    """
    with open(path, "rb") as pfile:
        data = pkl.load(pfile)
    return data



def circles(x, y, s, c='b', ax=None, vmin=None, vmax=None, **kwargs):
    """
    Make a scatter of circles plot of x vs y, where x and y are sequence
    like objects of the same lengths. The size of circles are in data scale.

    Parameters
    ----------
    x,y : scalar or array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, )
        Radius of circle in data unit.
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)
        `c` can be a 2-D array in which the rows are RGB or RGBA, however.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
        norm, cmap, transform, etc.

    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`

    Examples
    --------
    a = np.arange(11)
    circles(a, a, a*0.2, c=a, alpha=0.5, edgecolor='none')
    plt.colorbar()

    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection

    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None
    if 'fc' in kwargs: kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs: kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs: kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs: kwargs.setdefault('linewidth', kwargs.pop('lw'))

    patches = [Circle((x_, y_), s_) for x_, y_, s_ in np.broadcast(x, y, s)]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        collection.set_array(np.asarray(c))
        collection.set_clim(vmin, vmax)
    if ax is not None:
        ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    if c is not None:
        plt.sci(collection)
    return collection



def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.


def fileList(foldPath=None, included=['*.dfpickle'], excluded=[], defDir='/home/rhagoletis/catkin/src/World/bags/'):
    '''
    foldPath : path of folder
    included : default files to included
    excluded : to be removed

    '''
    if foldPath is None:
        import easygui as gui
        foldPath = gui.diropenbox(default=defDir)

    matches = []
    for root, dirnames, filenames in os.walk(foldPath):

        for include in included:
            for filename in fnmatch.filter(filenames, include):
                toKeep = True

                for exclude in excluded:
                    if exclude in filename:
                        toKeep = False

                if toKeep:
                    matches.append(os.path.join(root, filename))
    matches.sort()
    return matches


# Run the above function and store its results in a variable.
# full_file_paths = get_filepaths("/Users/johnny/Desktop/TEST")


def depickler(path):
    """
    Extracts a pickled Python object and returns it
    """
    with open(path, "rb") as pfile:
        data = pkl.load(pfile)
    return data


def circles(x, y, s, c='b', ax=None, vmin=None, vmax=None, **kwargs):
    """
    Make a scatter of circles plot of x vs y, where x and y are sequence
    like objects of the same lengths. The size of circles are in data scale.

    Parameters
    ----------
    x,y : scalar or array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, )
        Radius of circle in data unit.
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)
        `c` can be a 2-D array in which the rows are RGB or RGBA, however.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
        norm, cmap, transform, etc.

    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`

    Examples
    --------
    a = np.arange(11)
    circles(a, a, a*0.2, c=a, alpha=0.5, edgecolor='none')
    plt.colorbar()

    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection

    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None
    if 'fc' in kwargs: kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs: kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs: kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs: kwargs.setdefault('linewidth', kwargs.pop('lw'))

    patches = [Circle((x_, y_), s_) for x_, y_, s_ in np.broadcast(x, y, s)]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        collection.set_array(np.asarray(c))
        collection.set_clim(vmin, vmax)
    if ax is not None:
        ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    if c is not None:
        plt.sci(collection)
    return collection


def bag2pickle(paths=None, defaultPath="/home/rhagoletis/catkin/src/World/bags/", toHdf5=True, toPickle=False):
    '''
    Load  bag filesto make into respective dataframes
    '''
    if paths is None:
        paths = easygui.fileopenbox(title="Bags to Dataframes"
                                    , default=defaultPath,
                                    multiple=True, filetypes=["*traj.bag"])

    print paths, "\n"

    metadata = None
    tic()
    i = 1
    for path in paths:
        tic()
        print "starting analysis of file %s , %s / %s files" % (path.split('/')[-1], i, len(paths))

        #read the trajectory topic from bag
        try:
            df = rosbag_pandas.bag_to_dataframe(path, include=['/trajectory'])
        except Exception as e:

            #If error, dump none
            print "Bag has some nonetype error", path, e
            df = None
            parameters = None
            picklepath = None
            continue

        #try to recover parameters dict from the json loaded in later
        bag = rosbag.Bag(path)

        try:
            for topic, msg, t in bag.read_messages(topics='/metadata'):
                a = msg
                #         parameters=json.loads(a.data)
                #         metadata={"meta":parameters}

            metadata = json.loads(a.data)
            parameters = metadata['parameters']

        except:

            #if header recovery fails, try the pickle file stored in the same directory
            print "no such file!, trying the pickle"
            try:
                metadata = depickler(paths[0].split('.bag')[0])
                parameters = metadata['parameters']

            except IOError:
                #try the blank pickle if all fails
                metadata = depickler(paths[0].split('.bag')[0] + '.pickle')
                parameters = metadata['parameters']


            except Exception as e:
                print ("tried all forms of recovery, but no avail",e)
                parameters = None

        #store the parameters in a dict
        df.parameters = parameters
        obj = dict(df=df, metadata=metadata)

        # legacy naming
        # picklepath=path+"_df.pickle"
        # new naming
        if toPickle:
            picklepath = path + ".dfpickle"
            pickler(obj, picklepath)

        # bagStamp=('_'.join(df.parameters['bagFileName'].split('/')[-1].split('_')[:3]))
        # print "stamp is",bagStamp

        if toHdf5:
            picklepath = path + '.h5'
            df.to_hdf(picklepath, 'df')
            #     df.to_pickle(picklepath,)

        i += 1
        toc()
    print "\nanalysis of %s files complete" % len(paths)
    return df, parameters, picklepath


def bagDir2pickle(foldPath=None, defaultPath="/home/rhagoletis/catkin/src/World/bags/", postProcFunc=None,
                  included=['*.bag'], excluded=[]):
    tstart = datetime.now()

    if foldPath is None:
        foldPath = easygui.diropenbox(default=defaultPath)

    matches = fileList(foldPath, included=included, excluded=excluded)

    #sort reverse chrology to analyse the recentmost files
    matches.sort(reverse=True)
    print matches

    for i, path in enumerate(matches):
        print "\n\n\n\nFile {0} of {1}".format(i, len(matches))
        try:
            loopStart = datetime.now()
            df, parameters, picklepath = bag2pickle(paths=[path])
            #if df empty, move on
            if df is None:
                continue
            #if postprocfunc given, apply
            if postProcFunc is not None:
                postProcFunc(df, parameters, picklepath)

            print('\n\n Currently anal on {}. \n It is {} / {} file'.format(path.split('/')[-1], i, len(matches)))
            print "Time taken for this file :", (datetime.now() - loopStart)
            print '\n'

        except Exception as e:
            print "something odd in ", path, e

    tend = datetime.now()
    print "Total time", tend - tstart


def pickle2df(path=None, hdf5=False, defaultPath="/home/rhagoletis/catkin/src/World/bags/"):
    '''Load a dataframe from pickle'''

    if hdf5:
        filetypes = ["*.h5"]
    else:
        filetypes = ["*dfpickle"]

    if path is None:
        path = easygui.fileopenbox(title="Bags to Dataframes"
                                   , default=defaultPath,
                                   multiple=False, filetypes=filetypes)

    # print path, "\n"
    #     for path in paths:
    #     print "\n\ncurrentl;y analysisnh",path

    # picklepath=path+"_df.pickle"

    if hdf5:
        df = pd.read_hdf(path)
        fnjson = '.'.join(path.split('.')[:-2]) + '.json'
        with open(fnjson) as f:
            parameters = (json.load(f))['parameters']
            # print parameters

    else:
        data = depickler(path)
        parameters = data['metadata']['parameters']
        df = data["df"]

    df.parameters = parameters

    return df, parameters, path


def bokehPlot(df=None, case=None, parameters=None, fig=None,
              TOOLS="pan,crosshair,wheel_zoom\
              ,box_zoom,reset,box_select,lasso_select,undo,redo,save",
              x_range=(506, 526), y_range=(506, 526),
              output_backend="webgl", plot_width=500, plot_height=500,
              s=4, title=None, addOdour=True,
              addSmallTit=True, addStart=True, addArrow=True,
              xw=20, yw=20, xi=506., yi=503.,showPlot=True,
              reallign=False,xc=513, yc=513,decimation=1):
    '''
    
    :param df: Dataframe which containall trajectories 
    :param case: the case to be plotted, if none, plots all
    :param parameters: the dict having all run conditions
    :param fig: fig object to plot into, if none will create new
    :param TOOLS: The tools to be included in the html page
    :param x_range: the x range tuple
    :param y_range: the y range tuple
    :param output_backend: ideally webgl to handle so many points
    :param plot_width: px width of plot
    :param plot_height: px height og plot
    :param s: size of the scatter
    :param xc: x re center 
    :param yc: y re center
    :param xw: 
    :param yw: 
    :param xi: 
    :param yi: 
    :param showPlot: 
    :param title: Plot title
    :param addOdour: add odour overlay
    :param addSmallTit: add small title to sub text
    :param addStart: add start triangle
    :param addArrow: add wind direction arrow
    :param reallign: realligh to new center
    :param xre: recenter x
    :param yre: recenter y
    :param decimation: decimation factor
    :return: 
    '''
    if df is None:
        df,parameters,title=pickle2df(hdf5=True)

    if case is None:
        dfc = df.ix[::decimation]
    else:
        dfc = df[df.trajectory__case == case].ix[::decimation]

    x = dfc.trajectory__pPos_x
    y = dfc.trajectory__pPos_y
    h = np.deg2rad(dfc.trajectory__pOri_x)

    if 'haw' in parameters['fly']:
        dfcv = dfc.trajectory__valve2
        dfcvP = dfc.trajectory__valve1  # other odour incase of error in gui


    elif 'apple' in parameters['fly']:
        dfcv = dfc.trajectory__valve1
        dfcvP = dfc.trajectory__valve2  # other odour incase of error in gui

    ox = dfc[dfcv == True].trajectory__pPos_x
    oy = dfc[dfcv == True].trajectory__pPos_y

    oxP = dfc[dfcvP == True].trajectory__pPos_x
    oyP = dfc[dfcvP == True].trajectory__pPos_y


    try:
        cm = bp.viridis(max(parameters['odourQuad']) + 1)
        fc = cm[parameters["odourQuad"][case]]
    except TypeError:
        fc = bp.viridis(10)[5]

    if not addSmallTit:
        title = ''

    if fig is None:
        fig = figure(tools=TOOLS, x_range=x_range, y_range=y_range, output_backend=output_backend,
                     plot_width=plot_width, plot_height=plot_height,
                     active_scroll='wheel_zoom', title=title)

    fig.title.align = 'center'
    fig.title.text_font_size = '14pt'

    r = 2
    try:
        xs = parameters['playerInitPos'][0]
        ys = parameters['playerInitPos'][1]
        theta = parameters['windQuadOpen'][case] + 180
    except KeyError:
        print ('key error, old version?')

    if reallign:
        xoffs = xs - xc
        yoffs = ys - yc
    else:
        xoffs = 0
        yoffs = 0


    # Pos and heading of fly
    fig.triangle(x - xoffs, y - yoffs, size=s, angle=h, fill_alpha=0.5, line_color=None)

    if addOdour:
        # circle at odour pos, with pf encoded in color
        fig.circle(ox - xoffs, oy - yoffs, size=2 * s, fill_alpha=0.8, line_color=None, fill_color=fc)
        fig.circle(oxP - xoffs, oyP - yoffs, size=2 * s, fill_alpha=0.8, line_color=None, fill_color=(255, 0, 0))

    if addStart:
        # triangle at init pos
        fig.triangle(parameters['playerInitPos'][0], parameters['playerInitPos'][1],
                     size=3 * s, angle=0, fill_alpha=0.9, line_color=None, color='firebrick')

    # arrow

    if parameters["windQuad"][case] == -3:
        lw = 0
        la = 0.1
    elif parameters["windQuad"][case] == -2:
        lw = 4
        la = 1
    else:
        lw = 10
        la = 1
    # print 'le is',lw
    if addArrow:
        fig.add_layout(Arrow(end=VeeHead(size=20), line_color="red", line_alpha=la,

                             x_start=xs, y_start=ys, line_width=lw,
                             x_end=xs + r * np.cos(np.deg2rad(theta)), y_end=ys + r * np.sin(np.deg2rad(theta))))


    return fig


def bokehQuadPlot(df=None,  parameters=None, fig=None,
              TOOLS="pan,crosshair,wheel_zoom\
              ,box_zoom,reset,box_select,lasso_select,undo,redo,save",
              x_range=(506, 526), y_range=(506, 526),
              output_backend="webgl", plot_width=500, plot_height=500,
              s=4, title=None, addOdour=True,
              addSmallTit=True, addStart=True, addArrow=True,
              xw=20, yw=20, xi=506., yi=503.,showPlot=True,
              reallign=False,xc=513, yc=513,decimation=1,

              path=None, tr=None, tl=None, bl=None, br=None,
              exposeReturn=False,addBigTit=True, ):
    '''
        plot a single df

    :param df: Dataframe which containall trajectories 
    :param case: the case to be plotted, if none, plots all
    :param parameters: the dict having all run conditions
    :param fig: fig object to plot into, if none will create new
    :param TOOLS: The tools to be included in the html page
    :param x_range: the x range tuple
    :param y_range: the y range tuple
    :param output_backend: ideally webgl to handle so many points
    :param plot_width: px width of plot
    :param plot_height: px height og plot
    :param s: size of the scatter
    :param xc: x re center 
    :param yc: y re center
    :param xw: 
    :param yw: 
    :param xi: 
    :param yi: 
    :param showPlot: 
    :param title: Plot title
    :param addOdour: add odour overlay
    :param addSmallTit: add small title to sub text
    :param addStart: add start triangle
    :param addArrow: add wind direction arrow
    :param reallign: realligh to new center
    :param xre: recenter x
    :param yre: recenter y
    :param decimation: decimation factor
    :param path:path of the file 
    :param tr: top right ax
    :param tl: top left ax
    :param bl: bottom left ax
    :param br: bottom right ax
    :param exposeReturn: expose return all the sub ax
    :param addBigTit: add big title
    :return: 
    '''
    # output to static HTML file (with CDN resources)
    if path is None:
        path = 'blank'
    output_file(path + ".html", title="Trajectory Quad", mode="cdn")

    # # create a new plot with the tools above, and explicit ranges
    # w = 500
    # h = 500
    # s = 6
    # xc = 516
    # yc = 513
    # xw = 20
    # yw = 20
    # xi = xc - xw / 2.
    # yi = yc - yw / 2.

    try:
        ls = parameters['loadingString']
        wq = parameters['windQuad']
        oq = parameters['odourQuad']
        wqo = parameters['windQuadOpen']

        def quadTitGen(quad, ls=ls, wq=wq, oq=oq, wqo=wqo):
            tit = 'odour : ' + str(oq[quad]) + '\t wind : ' + str(wq[quad]) + '\t windDir : ' + str(wqo[quad])
            #         tit='ls : '+str(ls)+'\t odour : '+str(oq[quad])+'\t wind : '+str(wq[quad])+'\t windDir : '+str(wqo[quad])
            return tit

    except KeyError:
        print ('old version key error?')
        def quadTitGen(quad):
            return str(quad)

    # df = df, case = 0, parameters = parameters, fig = bl,
    # TOOLS = TOOLS,x_range = x_range, y_range = y_range,
    # output_backend = output_backend, plot_width = plot_width, plot_height = plot_height,
    # s = s, title = title, addOdour = addOdour,
    # addSmallTit = addSmallTit, addStart = addStart, addArrow = addArrow,
    # xw = xw, yw = yw, xi = xi, yi = yi, showPlot = showPlot,
    # reallign = reallign, xc = xc, yc = yc, decimation = decimation,
    #
    # path = path, tr = tr, tl = tl, bl = bl, br = br,
    # exposeReturn = exposeReturn, addBigTit = addBigTit,

    bl = bokehPlot(df = df, case = 2,fig = bl,title = quadTitGen(2),
                   parameters = parameters,
                   TOOLS = TOOLS,x_range = x_range, y_range = y_range,
                   output_backend = output_backend, plot_width = plot_width, plot_height = plot_height,
                   s = s, addOdour = addOdour,
                   addSmallTit = addSmallTit, addStart = addStart, addArrow = addArrow,
                   xw = xw, yw = yw, xi = xi, yi = yi, showPlot = showPlot,
                   reallign = reallign, xc = xc, yc = yc, decimation = decimation,
                   )

    tr =bokehPlot(df = df, case = 0,fig = tr,title = quadTitGen(0),
                   parameters = parameters,
                   TOOLS = TOOLS,x_range = bl.x_range, y_range = bl.y_range,
                   output_backend = output_backend, plot_width = plot_width, plot_height = plot_height,
                   s = s, addOdour = addOdour,
                   addSmallTit = addSmallTit, addStart = addStart, addArrow = addArrow,
                   xw = xw, yw = yw, xi = xi, yi = yi, showPlot = showPlot,
                   reallign = reallign, xc = xc, yc = yc, decimation = decimation,
                   )
    tl =bokehPlot(df = df, case = 1,fig = tl,title = quadTitGen(1),
                   parameters = parameters,
                   TOOLS = TOOLS,x_range = bl.x_range, y_range = bl.y_range,
                   output_backend = output_backend, plot_width = plot_width, plot_height = plot_height,
                   s = s, addOdour = addOdour,
                   addSmallTit = addSmallTit, addStart = addStart, addArrow = addArrow,
                   xw = xw, yw = yw, xi = xi, yi = yi, showPlot = showPlot,
                   reallign = reallign, xc = xc, yc = yc, decimation = decimation,
                   )
    br =bokehPlot(df = df, case = 3,fig = br,title = quadTitGen(3),
                   parameters = parameters,
                   TOOLS = TOOLS,x_range = bl.x_range, y_range = bl.y_range,
                   output_backend = output_backend, plot_width = plot_width, plot_height = plot_height,
                   s = s, addOdour = addOdour,
                   addSmallTit = addSmallTit, addStart = addStart, addArrow = addArrow,
                   xw = xw, yw = yw, xi = xi, yi = yi, showPlot = showPlot,
                   reallign = reallign, xc = xc, yc = yc, decimation = decimation,
                   )
    # bl = bokehPlot(df, 2, parameters, TOOLS=TOOLS, title=quadTitGen(2),
    #                fig=bl, addSmallTit=addSmallTit, addStart=addStart,
    #                addArrow=addArrow, addOdour=addOdour
    #                , x_range=x_range, y_range=y_range)
    # tr = bokehPlot(df, 0, parameters, x_range=bl.x_range, y_range=bl.y_range, TOOLS=TOOLS, title=quadTitGen(0),
    #                fig=tr, addSmallTit=addSmallTit, addStart=addStart,
    #                addArrow=addArrow, addOdour=addOdour)
    # tl = bokehPlot(df, 1, parameters, x_range=bl.x_range, y_range=bl.y_range, TOOLS=TOOLS, title=quadTitGen(1),
    #                fig=tl, addSmallTit=addSmallTit, addStart=addStart
    #                , addArrow=addArrow, addOdour=addOdour)
    # br = bokehPlot(df, 3, parameters, x_range=bl.x_range, y_range=bl.y_range, TOOLS=TOOLS, title=quadTitGen(3),
    #                fig=br, addSmallTit=addSmallTit, addStart=addStart,
    #                addArrow=addArrow, addOdour=addOdour)

    p = gridplot([[tl, tr], [bl, br]])
    #     export_png(p, filename=path+".png")
    if addBigTit:
        p = column(Div(text=path.split('/')[-1]), p)

    if exposeReturn:  # if you want access to all plots
        return p, tr, tl, bl, br
    else:

        if showPlot:
            show(p)
        else:
            save(p)

        return p, None, None, None, None


def pickle2bokeh(path=None,hdf5=False,
                 fig=None,
                 TOOLS="pan,crosshair,wheel_zoom\
              ,box_zoom,reset,box_select,lasso_select,undo,redo,save",
                 x_range=(506, 526), y_range=(506, 526),
                 output_backend="webgl", plot_width=500, plot_height=500,
                 s=4, title=None, addOdour=True,
                 addSmallTit=True, addStart=True, addArrow=True,
                 xw=20, yw=20, xi=506., yi=503., showPlot=True,
                 reallign=False, xc=513, yc=513, decimation=1,

                 tr=None, tl=None, bl=None, br=None,
                 exposeReturn=False, addBigTit=True,

                 ):

    df, parameters, path = pickle2df(path=path, hdf5=hdf5)
    try:
        plot, tr, tl, bl, br = bokehQuadPlot(df = df,fig = fig,title = title,
                   parameters = parameters,
                   TOOLS = TOOLS,x_range = x_range, y_range = y_range,
                   output_backend = output_backend, plot_width = plot_width, plot_height = plot_height,
                   s = s, addOdour = addOdour,
                   addSmallTit = addSmallTit, addStart = addStart, addArrow = addArrow,
                   xw = xw, yw = yw, xi = xi, yi = yi, showPlot = showPlot,
                   reallign = reallign, xc = xc, yc = yc, decimation = decimation,
                    exposeReturn=exposeReturn,addBigTit=addBigTit)
    except SyntaxError as e:
        print "\n\n\n Exception", e
        return None, None, None, None

    if exposeReturn:  # if you want access to all plots
        return plot, df, parameters, path, tr, tl, bl, br
    else:
        return plot, df, parameters, path


def pickleDir2bokeh(foldPath=None,
                    defaultPath='/media/rhagoletis/6db8b2b2-ebe2-4555-9b79-93b10ef9dec31/agg/odour+wind/', hdf5=False,
                    fig=None,TOOLS="pan,crosshair,wheel_zoom\
              ,box_zoom,reset,box_select,lasso_select,undo,redo,save",
                 x_range=(506, 526), y_range=(506, 526),
                 output_backend="webgl", plot_width=500, plot_height=500,
                 s=4, title=None, addOdour=True,
                 addSmallTit=True, addStart=True, addArrow=True,
                 xw=20, yw=20, xi=506., yi=503., showPlot=True,
                 reallign=False, xc=513, yc=513, decimation=1,

                 tr=None, tl=None, bl=None, br=None,
                 exposeReturn=False, addBigTit=True,

                    ):
    tstart = datetime.now()

    if foldPath is None:
        foldPath = easygui.diropenbox(default=defaultPath)

    if hdf5:
        include = ['*.h5']
    else:
        include = ['*bag_df.pickle']
    matches = fileList(foldPath, included=include, excluded=['pf'])
    matches.sort(reverse=True)
    print matches

    for i,path in enumerate(matches):
        tic()
        plot, df, parameters, path = pickle2bokeh(path=path,hdf5=hdf5,
                    fig=fig,title = title,

                   TOOLS = TOOLS,x_range = x_range, y_range = y_range,
                   output_backend = output_backend, plot_width = plot_width, plot_height = plot_height,
                   s = s, addOdour = addOdour,
                   addSmallTit = addSmallTit, addStart = addStart, addArrow = addArrow,
                   xw = xw, yw = yw, xi = xi, yi = yi, showPlot = showPlot,
                   reallign = reallign, xc = xc, yc = yc, decimation = decimation,
                    exposeReturn=exposeReturn,addBigTit=addBigTit)

        #         df,parameters=pickle2df(path)

        #         try:
        #             bokehQuadPlot(df=df,parameters=parameters,path=path)
        #         except Exception as e:
        #             print "\n\n\n Exception",e
        #             pass

        print('\n\n Currently anal on {}. \n It is {} / {} file'.format(path.split('/')[-1], i, len(matches)))
        toc()

    tend = datetime.now()
    print "Total time", tend - tstart


def bagDir2pickle(foldPath=None, defaultPath="/home/rhagoletis/catkin/src/World/bags/", postProcFunc=None,
                  included=['*.bag'], excluded=[]):
    tstart = datetime.now()

    if foldPath is None:
        foldPath = easygui.diropenbox(default=defaultPath)

    matches = fileList(foldPath, included=included, excluded=excluded)
    matches.sort(reverse=True)
    print matches

    for i, path in enumerate(matches):
        print "\n\n\n\nFile {0} of {1}".format(i, len(matches))
        try:
            loopStart = datetime.now()
            df, parameters, picklepath = bag2pickle(paths=[path])
            if df is None:
                continue

            if postProcFunc is not None:
                postProcFunc(df, parameters, picklepath)

            print('\n\n Currently anal on {}. \n It is {} / {} file'.format(path.split('/')[-1], i, len(matches)))
            print "Time taken for this file :", (datetime.now() - loopStart)
            print '\n'
            i += 1

        except Exception as e:
            print "something odd in ", path, e

    tend = datetime.now()
    print "Total time", tend - tstart
