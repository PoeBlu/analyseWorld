from ipy_notebooks.ipyimports import *
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

# tic()
# bagDir2bokeh(foldPath=None,defaultPath="/media/rhagoletis/Traj",showPlot=False,)
# toc()


# filenames=fileList()
# # inputs = range(10)
# tic()
#

# def processInput(filename,bins=40, bottom=0, kind='scatterStack', toShow=True,
#                     start=500, stepSize=130, alpha=0.6, s=50):
#     dfDump=pickle2df(path=filename)
#     df=dfDump[0]
#     fname=dfDump[-1]
#     a=quadPolarHist(df,bins,bottom,kind,fname,toShow)
#     plt.close()
# results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in filenames)
#
# toc()
prog=0
totSize=0
def processInput(path, postProcFunc=None):
    global prog,totSize
    # print "\n\n\n\nFile {0} of {1}".format(i, len(matches))
    try:
        loopStart = datetime.now()
        df, parameters, picklepath = bag2pickle(paths=[path],toPickle=False,toHdf5=True)
        if df is None:
            return

        if postProcFunc is not None:
            postProcFunc(df, parameters, picklepath)

        # print('\n\n Currently anal on {}. \n It is {} file'.format(path.split('/')[-1], len(matches)))
        print "Time taken for this file :", (datetime.now() - loopStart)
        print "Finished file {0} of {1}".format(prog,totSize)
        print '\n'
        prog+=1

    except Exception as e:
        print "something odd in ", path, e


def bagDir2pickle_(foldPath=None,defaultPath="/home/rhagoletis/catkin/src/World/bags/",postProcFunc=None,
                  included=['*.bag'],excluded=[]):
    global totSize
    tstart = datetime.now()

    if foldPath is None:
        foldPath = easygui.diropenbox(default=defaultPath)

    matches = fileList(foldPath, included=included, excluded=excluded)
    totSize=len(matches)

    matches.sort(reverse=True)
    print matches


    results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in matches)

    tend = datetime.now()
    print "Total time", tend - tstart




# def pickleDir2bokeh_(foldPath=None, showPlot=False,
#                     addBigTit=True, addSmallTit=True, addArrow=True,
#                     addStart=True, addOdour=False,
#                     x_range=(506, 526), y_range=(506, 526),
#                     defaultPath='/media/rhagoletis/6db8b2b2-ebe2-4555-9b79-93b10ef9dec31/agg/odour+wind/', hdf5=False):
#     tstart = datetime.now()
#
#     if foldPath is None:
#         foldPath = easygui.diropenbox(default=defaultPath)
#
#     if hdf5:
#         include = ['*.h5']
#     else:
#         include = ['*bag_df.pickle']
#     matches = fileList(foldPath, included=include, excluded=['pf'])
#
#     matches.sort(reverse=True)
#     print matches
#
#     Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in matches)
#     plot, df, parameters, path = Parallel(n_jobs=num_cores)()pickle2bokeh(path, showPlot=showPlot,
#                                               addBigTit=addBigTit, addSmallTit=addSmallTit,
#                                               addArrow=addArrow, addStart=addStart,
#                                               x_range=x_range, y_range=y_range, addOdour=addOdour, hdf5=hdf5)
#
#
#     tend = datetime.now()
#     print "Total time", tend - tstart
#

tic()
bagDir2pickle_(foldPath=None,defaultPath="",postProcFunc=bokehQuadPlot,excluded=['full'])
toc()

