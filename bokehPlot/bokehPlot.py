# %matplotlib inline
# import mpld3
# mpld3.enable_notebook()
import rosbag_pandas
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
from opterator import opterate

import os


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
    :return data of the serialized pickle
    """
    with open(path, "rb") as pfile:
        data = pkl.load(pfile)
    return data


def fileList(foldPath=None, included=['*traj.bag_df.pickle'], excluded=[]):
    '''
    
    :param foldPath: Folder path to get the list of files, will ask if given none 
    :param included: list of filetypes to be excluded ex. ['*traj.bag_df.pickle']
    :param excluded: list of string to exclude ex. ['pf] 
    :return: matches: a list of filepaths in that the dir with the filtering done and sorted
    '''
    matches = []
    foldPath=easygui.diropenbox()
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


def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    :return list of absolute file paths
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.


# Run the above function and store its results in a variable.
full_file_paths = get_filepaths("/Users/johnny/Desktop/TEST")


def bokehPlot(df, case, parameters, fig=None,
              TOOLS="pan,crosshair,wheel_zoom\
              ,box_zoom,reset,box_select,lasso_select,undo,redo,save",
              x_range=(506, 526), y_range=(506, 526),
              output_backend="webgl", plot_width=1000, plot_height=1000,
              s=4, xc=513, yc=505,
              xw=20, yw=20, xi=506., yi=503.,
              showPlot=True, title=None, addOdour=True,
              addSmallTit=True, addStart=True, addArrow=True,reallign=False):

    '''
    
    
    :param df: dataframe to be plotted
    :param case: the case of the df to be plotted
    :param parameters: parameters of the run
    :param fig: the figure to which to be overlayed, if None will make a new figure
    :param TOOLS: Tools of the bokeh plot
    :param x_range: x range of the plot (xmin,xmax)
    :param y_range: y range of the plot (ymin,ymax)
    :param output_backend: preferbale webgl
    :param plot_width: width of the plot in px
    :param plot_height: height of plot in px
    :param s: the spot size of the triangle
    :param xc: x center of 
    :param yc: y center
    :param xw: x width range
    :param yw: y width range
    :param xi: z initialx    
    :param yi: y initial 
    :param showPlot: save and show plot or only save plot
    :param title: title of the plot
    :param addOdour: add odour overaly
    :param addSmallTit: add small title
    :param addStart: add start marker
    :param addArrow: arr arrow of wind direction
    :param reallign: reallign for frame shifts between trials
    :return: figure
    '''
    dfc = df[df.trajectory__case == case]
    x = dfc.trajectory__pPos_x
    y = dfc.trajectory__pPos_y
    h = np.deg2rad(dfc.trajectory__pOri_x)

    if 'haw' in parameters['fly']:
        dfcv = dfc.trajectory__valve2

    elif 'apple' in parameters['fly']:
        dfcv = dfc.trajectory__valve1

    ox = dfc[dfcv == True].trajectory__pPos_x
    oy = dfc[dfcv == True].trajectory__pPos_y

    #     opf=dfc[dfc.trajectory__valve1==True].trajectory__pOri_x

    #     import matplotlib as mpl
    #     colors = [
    #         "#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b, _ in 255*mpl.cm.viridis(mpl.colors.Normalize()(opf))
    #     ]


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
    xs = parameters['playerInitPos'][0]
    ys = parameters['playerInitPos'][1] + 8
    theta = parameters['windQuadOpen'][case] + 180
    if reallign:
        xoffs = xs - 513
        yoffs = ys - 513
    else:
        xoffs=0
        yoffs=0
    # Pos and heading of fly
    fig.triangle(x - xoffs, y - yoffs, size=s, angle=h, fill_alpha=0.5, line_color=None)

    if addOdour:
        # circle at odour pos, with pf encoded in color
        fig.circle(ox, oy, size=2 * s, fill_alpha=0.8, line_color=None, fill_color=fc)

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


def bokehQuadPlot(df, parameters, path, tr=None, tl=None, bl=None, br=None,
                  TOOLS="pan,crosshair,wheel_zoom,box_zoom,\
                  reset,box_select,lasso_select,undo,redo,save",
                  x_range=(506, 526), y_range=(506, 526), plot_width=1000,
                  plot_height=1000,
                  output_backend="webgl", showPlot=False, exposeReturn=False,
                  addBigTit=True, addSmallTit=True,
                  addArrow=True, addStart=True, addOdour=True):
    '''
    plot a single df
    '''

    # output to static HTML file (with CDN resources)
    output_file(path + ".html", title="Trajectory Quad", mode="cdn")

    # create a new plot with the tools above, and explicit ranges
    w = 500
    h = 500
    s = 6
    xc = 516
    yc = 513
    xw = 20
    yw = 20
    xi = xc - xw / 2.
    yi = yc - yw / 2.

    ls = parameters['loadingString']
    wq = parameters['windQuad']
    oq = parameters['odourQuad']
    wqo = parameters['windQuadOpen']

    def quadTitGen(quad, ls=ls, wq=wq, oq=oq, wqo=wqo):
        tit = 'odour : ' + str(oq[quad]) + '\t wind : ' + str(wq[quad]) + '\t windDir : ' + str(wqo[quad])
        #         tit='ls : '+str(ls)+'\t odour : '+str(oq[quad])+'\t wind : '+str(wq[quad])+'\t windDir : '+str(wqo[quad])

        return tit

    bl = bokehPlot(df, 2, parameters, TOOLS=TOOLS, title=quadTitGen(2),
                   fig=bl, addSmallTit=addSmallTit, addStart=addStart,
                   addArrow=addArrow, addOdour=addOdour
                   , x_range=x_range, y_range=y_range)
    tr = bokehPlot(df, 0, parameters, x_range=bl.x_range, y_range=bl.y_range, TOOLS=TOOLS, title=quadTitGen(0),
                   fig=tr, addSmallTit=addSmallTit, addStart=addStart,
                   addArrow=addArrow, addOdour=addOdour)
    tl = bokehPlot(df, 1, parameters, x_range=bl.x_range, y_range=bl.y_range, TOOLS=TOOLS, title=quadTitGen(1),
                   fig=tl, addSmallTit=addSmallTit, addStart=addStart
                   , addArrow=addArrow, addOdour=addOdour)
    br = bokehPlot(df, 3, parameters, x_range=bl.x_range, y_range=bl.y_range, TOOLS=TOOLS, title=quadTitGen(3),
                   fig=br, addSmallTit=addSmallTit, addStart=addStart,
                   addArrow=addArrow, addOdour=addOdour)

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


def pickle2df(path=None,defaultPath = "/home/rhagoletis/catkin/src/World/bags/"):
    '''
    
    :param path: 
    :return: df, parameters, path 
    '''

    if path is None:
        path = easygui.fileopenbox(title="Bags to Dataframes"
                                   , default=defaultPath,
                                   multiple=False, filetypes=["*df.pickle"])

    # print path, "\n"
    #     for path in paths:
    #     print "\n\ncurrentl;y analysisnh",path
    # picklepath=path+"_df.pickle"
    data = depickler(path)
    df = data["df"]
    parameters = data['metadata']['parameters']

    return df, parameters, path


def pickle2bokeh(path=None, showPlot=True, exposeReturn=False, tr=None, tl=None, bl=None, br=None,
                 addBigTit=True, addSmallTit=True, addArrow=True, addStart=True, addOdour=True,
                 x_range=(506, 526), y_range=(506, 526)):
    df, parameters, path = pickle2df(path=path)
    try:
        plot, tr, tl, bl, br = bokehQuadPlot(df=df, parameters=parameters, path=path, showPlot=showPlot,
                                             exposeReturn=exposeReturn, tr=tr, tl=tl, bl=bl, br=br,
                                             addBigTit=addBigTit, addSmallTit=addSmallTit, addArrow=addArrow,
                                             addStart=addStart, addOdour=addOdour,
                                             x_range=x_range, y_range=y_range)
    except SyntaxError as e:
        print "\n\n\n Exception", e
        return None, None, None, None

    if exposeReturn:  # if you want access to all plots
        return plot, df, parameters, path, tr, tl, bl, br
    else:
        return plot, df, parameters, path



def pickleDir2bokeh(foldPath=None, showPlot=False):
    tstart = datetime.now()

    if foldPath is None:
        foldPath = easygui.diropenbox()

    matches = fileList(foldPath, included=['*traj.bag_df.pickle'], excluded=['pf'])
    #     matches = []
    #     for root, dirnames, filenames in os.walk(foldPath):
    #         for filename in fnmatch.filter(filenames, '*traj.bag_df.pickle'):
    #             if 'pf' in filename:
    #                 continue
    #             matches.append(os.path.join(root, filename))

    matches.sort(reverse=True)
    print matches

    i = 1  # file progress index
    for path in matches:
        tic()
        plot, df, parameters, path = pickle2bokeh(path, showPlot=showPlot)

        #         df,parameters=pickle2df(path)

        #         try:
        #             bokehQuadPlot(df=df,parameters=parameters,path=path)
        #         except Exception as e:
        #             print "\n\n\n Exception",e
        #             pass

        print('\n\n Currently anal on {}. \n It is {} / {} file'.format(path.split('/')[-1], i, len(matches)))

        i += 1
        toc()

    tend = datetime.now()
    print "Total time", tend - tstart


# def dfDir2bokeh():

#     from datetime import datetime

#     tstart = datetime.now()



#     foldPath= easygui.diropenbox(    default="/home/rhagoletis/catkin/src/World/bags/")
#     matches = []
#     for root, dirnames, filenames in os.walk(foldPath):
#         for filename in fnmatch.filter(filenames, '*traj.bag_df.pickle'):
#             if 'pf' in filename:
#                 continue
#             matches.append(os.path.join(root, filename))
#     matches.sort(reverse=True)
#     print matches

#     i=1
#     for path in matches:

#         tic()

# #         df,parameters=pickle2bokeh(path)
#         df,parameters,path2=pickle2df(path)

#         try:
#             bokehQuadPlot(df=df,parameters=parameters,path=path,showPlot=False)
#         except Exception as e:
#             print "\n\n\n Exception",e
#             pass

#         print('\n\n Currently anal on {}. \n It is {} / {} file'.format(path.split('/')[-1], i,len(matches)))

# #         print "Currently analysing", path
#         i+=1
#         toc()

#     tend = datetime.now()
#     print "Total time",tend - tstart


def bag2pickle(paths=None, defaultPath="/home/rhagoletis/catkin/src/World/bags/"):
    '''
    Load  bag filesto make into respective dataframes
    '''
    # defaultPath="/home/rhagoletis/catkin/src/World/bags/"
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
        try:
            df = rosbag_pandas.bag_to_dataframe(path, include=['/trajectory'])
        except Exception as e:
            print "Bag has some nonetype error", path, e
            df = None
            parameters = None
            picklepath = None
            continue
        bag = rosbag.Bag(path)

        try:
            for topic, msg, t in bag.read_messages(topics='/metadata'):
                a = msg
                #         parameters=json.loads(a.data)
                #         metadata={"meta":parameters}

            metadata = json.loads(a.data)
            parameters = metadata['parameters']

        except:
            print "no such file!, trying the pickle"
            try:
                metadata = depickler(paths[0].split('.bag')[0])
            except IOError:
                metadata = depickler(paths[0].split('.bag')[0] + '.pickle')

            parameters = metadata['parameters']

        obj = dict(df=df, metadata=metadata)

        picklepath = path + "_df.pickle"
        pickler(obj, picklepath)
        #     df.to_pickle(picklepath)

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
    #     matches = []
    #     for root, dirnames, filenames in os.walk(foldPath):
    #         for filename in fnmatch.filter(filenames, '*traj.bag_df.pickle'):
    #             if 'pf' in filename:
    #                 continue
    #             matches.append(os.path.join(root, filename))

    matches.sort(reverse=True)
    print matches

    i = 1  # file progress index
    for path in matches:

        tic()
        df, parameters, picklepath = bag2pickle(paths=[path])
        if df is None:
            continue

        if postProcFunc is not None:
            postProcFunc(df, parameters, picklepath)

        print('\n\n Currently anal on {}. \n It is {} / {} file'.format(path.split('/')[-1], i, len(matches)))

        i += 1
        toc()

    tend = datetime.now()
    print "Total time", tend - tstart


def bagDir2bokeh(foldPath=None, defaultPath="/home/rhagoletis/catkin/src/World/bags/", showPlot=False,
                 excluded=['full']):
    bagDir2pickle(foldPath, defaultPath=defaultPath, postProcFunc=bokehQuadPlot, excluded=excluded)



def plotAggreg(foldPath=None,
               defaultPath='/media/rhagoletis/6db8b2b2-ebe2-4555-9b79-93b10ef9dec31/agg/windNull',
              x_range=(493,533), y_range=(485,525),):
    if foldPath is None:
        foldPath=easygui.diropenbox(default=defaultPath)
    files=fileList(foldPath,included=['*.bag_df.pickle'])
    tr=None
    tl=None
    bl=None
    br=None
    iTot=len(files)
    i=1
    for item in files:
        plot,df,parameters,path,tr,tl,bl,br=pickle2bokeh(item,exposeReturn=True,tr=tr,tl=tl,bl=bl,br=br,
                     addBigTit=False,addSmallTit=False,addArrow=False,addStart=False,
                        x_range=x_range, y_range=y_range,)
        print ("currently analysing {}, {}/{} files ").format(item.strip('/')[-1],i,iTot)
        i+=1
    show(plot)
    return plot
# def xDir2y(x2y,included=['*.bag'],excluded=[],defaultPath="/home/rhagoletis/catkin/src/World/bags/"):

#     tstart = datetime.now()

#     foldPath= easygui.diropenbox(default=defaultPath)

#     matches=fileList(foldPath,included=included,excluded=excluded)
# #     matches = []
# #     for root, dirnames, filenames in os.walk(foldPath):
# #         for filename in fnmatch.filter(filenames, '*traj.bag_df.pickle'):
# #             if 'pf' in filename:
# #                 continue
# #             matches.append(os.path.join(root, filename))

#     matches.sort(reverse=True)
#     print matches

#     i=1 #file progress index
#     for path in matches:

#         tic()
#         x2y(paths=[path])

#         print('\n\n Currently anal on {}. \n It is {} / {} file'.format(path.split('/')[-1], i,len(matches)))

#         i+=1
#         toc()

#     tend = datetime.now()
#     print "Total time",tend - tstart

# bag2pickle()
# bagDir2pickle()
# pickle2bokeh()
# pickle2df()
# dfDir2bokeh()
# xc=513
# yc=505
# w=19
# xw=w
# yw=w
# xr=(xc-xw,xc+xw)
# yr=(yc-yw,yc+yw)
#
# out=plotAggreg(x_range=xr,y_range=yr)

if __name__ =='__main__':
    xc = 514
    yc = 510

    try:
        w = int(raw_input('Enter the  half width of plot\n'))
    except ValueError:
        w=20
        print "you entered nothing, using default value",w
    xw = w
    yw = w
    xr = (xc - xw, xc + xw)
    yr = (yc - yw, yc + yw)

    pickleDir2bokeh(addBigTit=False, addSmallTit=False, addArrow=False, addStart=False,x_range=xr, y_range=yr)
    # plotAggreg(x_range=xr, y_range=yr)
