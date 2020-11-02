import numpy as np
import xarray as xr
import re
from pathlib import Path
import collections

def distance(val, ref):
    return abs(ref - val)
vectDistance = np.vectorize(distance)

def cmap_xmap(function, cmap):
    """ Applies function, on the indices of colormap cmap. Beware, function
    should map the [0, 1] segment to itself, or you are in for surprises.
    See also cmap_xmap.
    """
    cdict = cmap._segmentdata
    function_to_map = lambda x : (function(x[0]), x[1], x[2])
    for key in ('red','green','blue'):
        cdict[key] = map(function_to_map, cdict[key])
#        cdict[key].sort()
#        assert (cdict[key][0]<0 or cdict[key][-1]>1), "Resulting indices extend out of the [0, 1] segment."
    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)

def getClosest(sortedMatrix, column, val):
    while len(sortedMatrix) > 3:
        half = int(len(sortedMatrix) / 2)
        sortedMatrix = sortedMatrix[-half - 1:] if sortedMatrix[half, column] < val else sortedMatrix[: half + 1]
    if len(sortedMatrix) == 1:
        result = sortedMatrix[0].copy()
        result[column] = val
        return result
    else:
        safecopy = sortedMatrix.copy()
        safecopy[:, column] = vectDistance(safecopy[:, column], val)
        minidx = np.argmin(safecopy[:, column])
        safecopy = safecopy[minidx, :].A1
        safecopy[column] = val
        return safecopy

def convert(column, samples, matrix):
    return np.matrix([getClosest(matrix, column, t) for t in samples])

def valueOrEmptySet(k, d):
    return (d[k] if isinstance(d[k], set) else {d[k]}) if k in d else set()

def mergeDicts(d1, d2):
    """
    Creates a new dictionary whose keys are the union of the keys of two
    dictionaries, and whose values are the union of values.
    Parameters
    ----------
    d1: dict
        dictionary whose values are sets
    d2: dict
        dictionary whose values are sets
    Returns
    -------
    dict
        A dict whose keys are the union of the keys of two dictionaries,
    and whose values are the union of values
    """
    res = {}
    for k in d1.keys() | d2.keys():
        res[k] = valueOrEmptySet(k, d1) | valueOrEmptySet(k, d2)
    return res

def extractCoordinates(filename):
    """
    Scans the header of an Alchemist file in search of the variables.
    Parameters
    ----------
    filename : str
        path to the target file
    mergewith : dict
        a dictionary whose dimensions will be merged with the returned one
    Returns
    -------
    dict
        A dictionary whose keys are strings (coordinate name) and values are
        lists (set of variable values)
    """
    with open(filename, 'r') as file:
#        regex = re.compile(' (?P<varName>[a-zA-Z._-]+) = (?P<varValue>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),?')
        regex = r"(?P<varName>[a-zA-Z._-]+) = (?P<varValue>[^,]*),?"
        dataBegin = r"\d"
        is_float = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
        for line in file:
            match = re.findall(regex, line)
            if match:
                return {
                    var : float(value) if re.match(is_float, value)
                        else bool(re.match(r".*?true.*?", value.lower())) if re.match(r".*?(true|false).*?", value.lower())
                        else value
                    for var, value in match
                }
            elif re.match(dataBegin, line[0]):
                return {}

def extractVariableNames(filename):
    """
    Gets the variable names from the Alchemist data files header.
    Parameters
    ----------
    filename : str
        path to the target file
    Returns
    -------
    list of list
        A matrix with the values of the csv file
    """
    with open(filename, 'r') as file:
        dataBegin = re.compile('\d')
        lastHeaderLine = ''
        for line in file:
            if dataBegin.match(line[0]):
                break
            else:
                lastHeaderLine = line
        if lastHeaderLine:
            regex = re.compile(' (?P<varName>\S+)')
            return regex.findall(lastHeaderLine)
        return []

def openCsv(path):
    """
    Converts an Alchemist export file into a list of lists representing the matrix of values.
    Parameters
    ----------
    path : str
        path to the target file
    Returns
    -------
    list of list
        A matrix with the values of the csv file
    """
    regex = re.compile('\d')
    with open(path, 'r') as file:
        lines = filter(lambda x: regex.match(x[0]), file.readlines())
        return [[float(x) for x in line.split()] for line in lines]

def beautifyValue(v):
    """
    Converts an object to a better version for printing, in particular:
        - if the object converts to float, then its float value is used
        - if the object can be rounded to int, then the int value is preferred
    Parameters
    ----------
    v : object
        the object to try to beautify
    Returns
    -------
    object or float or int
        the beautified value
    """
    try:
        v = float(v)
        if v.is_integer():
            return int(v)
        return v
    except:
        if type(v) == np.str_:
            v = v.replace('\n', '').replace(' ', '_')
        return v
    
def beautifyFigName(figname):
    for symbol in r".[]\/@:":
        figname = figname.replace(symbol, '_')
    return figname

if __name__ == '__main__':
    # CONFIGURE SCRIPT
    # Where to find Alchemist data files
    directory = 'data'
    # Where to save charts
    output_directory = 'charts'
    # How to name the summary of the processed data
    pickleOutput = 'data_summary'
    # Experiment prefixes: one per experiment (root of the file name)
    experiments = ['sim']
    floatPrecision = '{: 0.2f}'
    # Number of time samples 
    timeSamples = 200
    # time management
    minTime = 0.0
    maxTime = 345600.0
    timeColumnName = 'instant[SECONDS]'
    logarithmicTime = False
    # One or more variables are considered random and "flattened"
    seedVars = ['seed']
    # Label mapping
    class Measure:
        def __init__(self, description, unit = None):
            self.__description = description
            self.__unit = unit
        def description(self):
            return self.__description
        def unit(self):
            return '' if self.__unit is None else f'({self.__unit})'
        def derivative(self, new_description = None, new_unit = None):
            def cleanMathMode(s):
                return s[1:-1] if s[0] == '$' and s[-1] == '$' else s
            def deriveString(s):
                return r'$d ' + cleanMathMode(s) + r'/{dt}$'
            def deriveUnit(s):
                return f'${cleanMathMode(s)}' + '/{s}$' if s else None
            result = Measure(
                new_description if new_description else deriveString(self.__description),
                new_unit if new_unit else deriveUnit(self.__unit),
            )
            return result
        def __str__(self):
            return f'{self.description()} {self.unit()}'
    
    centrality_label = 'H_a(x)'
    def expected(x):
        return r'\mathbf{E}[' + x + ']'
    def stdev_of(x):
        return r'\sigma{}[' + x + ']'
    def mse(x):
        return 'MSE[' + x + ']'
    def cardinality(x):
        return r'\|' + x + r'\|'

    labels = {
        'instant[SECONDS]': Measure('time', 's'),
        'commCountUploadPartial': Measure('commCountUploadPartial'),
        'delaySumUploadPartial[SECONDS]': Measure('delaySumUploadPartial', 's'),
        'delayMaxUploadPartial[SECONDS]': Measure('delayMaxUploadPartial', 's'),
        'commCountUploadTot': Measure('commCountUploadTot'),
        'delaySumUploadTot[SECONDS]': Measure('delaySumUploadTot', 's'),
        'delayMaxUploadTot[SECONDS]': Measure('delayMaxUploadTot', 's'),
        'commCountDownloadPartial': Measure('commCountDownloadPartial'),
        'delaySumDownloadPartial[SECONDS]': Measure('delaySumDownloadPartial', 's'),
        'delayMaxDownloadPartial[SECONDS]': Measure('delayMaxDownloadPartial', 's'),
        'commCountDownloadTot': Measure('commCountDownloadTot'),
        'delaySumDownloadTot[SECONDS]': Measure('delaySumDownloadTot', 's'),
        'delayMaxDownloadTot[SECONDS]': Measure('delayMaxDownloadTot', 's'),
        'commCountDownloadMaxPartial': Measure('commCountDownloadMaxPartial'),
        'delaySumDownloadMaxPartial[SECONDS]': Measure('delaySumDownloadMaxPartial', 's'),
        'delayMaxDownloadMaxPartial[SECONDS]': Measure('delayMaxDownloadMaxPartial', 's'),
        'commCountDownloadMaxTot': Measure('commCountDownloadMaxTot'),
        'delaySumDownloadMaxTot[SECONDS]': Measure('delaySumDownloadMaxTot', 's'),
        'delayMaxDownloadMaxTot[SECONDS]': Measure('delayMaxDownloadMaxTot', 's'),
        'runOnCloudPartial': Measure('runOnCloudPartial'),
        'runOnEdgePartial': Measure('runOnEdgePartial'),
        'runOnSmartphonePartial': Measure('runOnSmartphonePartial'),
        'runOnCloudTot': Measure('runOnCloudTot'),
        'runOnEdgeTot': Measure('runOnEdgeTot'),
        'runOnSmartphoneTot': Measure('runOnSmartphoneTot'),
    }
    def derivativeOrMeasure(variable_name):
        if variable_name.endswith('dt'):
            return labels.get(variable_name[:-2], Measure(variable_name)).derivative()
        return Measure(variable_name)
    def label_for(variable_name):
        return labels.get(variable_name, derivativeOrMeasure(variable_name)).description()
    def unit_for(variable_name):
        return str(labels.get(variable_name, derivativeOrMeasure(variable_name)))
    
    # Setup libraries
    np.set_printoptions(formatter={'float': floatPrecision.format})
    # Read the last time the data was processed, reprocess only if new data exists, otherwise just load
    import pickle
    import os
    separator = os.path.sep
    if os.path.exists(directory):
        newestFileTime = max([os.path.getmtime(directory + separator + file) for file in os.listdir(directory)], default=0.0)
        try:
            lastTimeProcessed = pickle.load(open('timeprocessed', 'rb'))
        except:
            lastTimeProcessed = -1
        shouldRecompute = not os.path.exists(".skip_data_process") and newestFileTime != lastTimeProcessed
        if not shouldRecompute:
            try:
                means = pickle.load(open(pickleOutput + '_mean', 'rb'))
            except: 
                shouldRecompute = True
        if shouldRecompute:
            timefun = np.logspace if logarithmicTime else np.linspace
            means = {}
            for experiment in experiments:
                # Collect all files for the experiment of interest
                import fnmatch
                allfiles = filter(lambda file: fnmatch.fnmatch(file, experiment + '_*.txt'), os.listdir(directory))
                allfiles = [directory + separator + name for name in allfiles]
                allfiles.sort()
                # From the file name, extract the independent variables
                dimensions = {}
                for file in allfiles:
                    dimensions = mergeDicts(dimensions, extractCoordinates(file))
                dimensions = {k: sorted(v) for k, v in dimensions.items()}
                # Add time to the independent variables
                dimensions[timeColumnName] = range(0, timeSamples)
                # Compute the matrix shape
                shape = tuple(len(v) for k, v in dimensions.items())
                # Prepare the Dataset
                dataset = xr.Dataset()
                for k, v in dimensions.items():
                    dataset.coords[k] = v
                if len(allfiles) == 0:
                    print("WARNING: No data for experiment " + experiment)
                    means[experiment] = dataset
                else:
                    varNames = extractVariableNames(allfiles[0])
                    # NB these two vars are specific for this type of data and they represent two aggregate metrics
                    varNames.append('commCount')
                    varNames.append('delaySum')
                    for v in varNames:
                        if v != timeColumnName:
                            novals = np.ndarray(shape)
                            novals.fill(float('nan'))
                            dataset[v] = (dimensions.keys(), novals)
                    # Compute maximum and minimum time, create the resample
                    timeColumn = varNames.index(timeColumnName)
                    allData = { file: np.matrix(openCsv(file)) for file in allfiles }
                    computeMin = minTime is None
                    computeMax = maxTime is None
                    if computeMax:
                        maxTime = float('-inf')
                        for data in allData.values():
                            maxTime = max(maxTime, data[-1, timeColumn])
                    if computeMin:
                        minTime = float('inf')
                        for data in allData.values():
                            minTime = min(minTime, data[0, timeColumn])
                    timeline = timefun(minTime, maxTime, timeSamples)
                    # Resample
                    for file in allData:
    #                    print(file)
                        allData[file] = convert(timeColumn, timeline, allData[file])
                    # Populate the dataset
                    for file, data in allData.items():
                        dataset[timeColumnName] = timeline
                        for idx, v in enumerate(varNames):
                            if v != timeColumnName:
                                darray = dataset[v]
                                experimentVars = extractCoordinates(file)
                                # NB this 'if' is needed to populate also the two metric added before 
                                # ('commCount' and 'delaySum'), if you remove them or you modify the number 
                                # of metrics you hae also to modify this 'if'
                                if idx > 28:
                                    test = data[:, (idx - 25)].A1 + data[:, (idx - 19)].A1
                                    darray.loc[experimentVars] = test
                                else:
                                    darray.loc[experimentVars] = data[:, idx].A1
                    # Fold the dataset along the seed variables, producing the mean and stdev datasets
                    mergingVariables = [seed for seed in seedVars if seed in dataset.coords]
                    means[experiment] = dataset.mean(dim = mergingVariables, skipna=True)
            # Save the datasets
            pickle.dump(means, open(pickleOutput + '_mean', 'wb'), protocol=-1)
            pickle.dump(newestFileTime, open('timeprocessed', 'wb'))
    else:
        means = { experiment: xr.Dataset() for experiment in experiments }

    # QUICK CHARTING

    import matplotlib
    from matplotlib import rc
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmx
    from mpl_toolkits.mplot3d import Axes3D # needed for 3d projection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    matplotlib.rcParams.update({'axes.titlesize': 12})
    matplotlib.rcParams.update({'axes.labelsize': 10})
#    Custom charting
#    colors = ['#33a02c','#e31a1c','#1f78b4', '#000000FF']
    colors_length = 3
    colors = [cmx.viridis(float(i)/(colors_length - 1)) for i in range(colors_length)]

    def add_data_to_chart_delays(ax, x, y, z, what, idxColor):
        c = ax.plot_trisurf(x,y,z, label=what, linewidth=2, antialiased=False, shade=True, alpha=0.5, color=colors[idxColor])
        c._facecolors2d=c._facecolors3d
        c._edgecolors2d=c._edgecolors3d
    def make_delay_chart2D(datas, dimension, xLabel, title='', filename=''):
        bx = []
        ey = []
        cy = []
        for b in datas[dimension].values:
            filterd = datas.sel({dimension:b})
            delays = filterd['delaySum'].values
            communications = filterd['commCount'].values
            bx.append(b)
            cy.append((delays[0, -1] / communications[0, -1] * 1000))
            ey.append((delays[1, -1] / communications[1, -1] * 1000))
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(title)
        ax.set_xlabel(f"{xLabel}")
        ax.set_ylabel("Mean delay (ms)")
        ax.set_xlim([0.0, 1.0])
#        ax.set_ylim([0,max(max(cy), max(ey)) + 50])
        ax.plot(bx, cy, label='$B$=cloud', color=colors[2], linewidth=1.0)
        ax.plot(bx, ey, label='$B$=edge', color=colors[1], linewidth=1.0)
        ax.legend()
        plt.tight_layout()
        fig.savefig(filename)
        plt.close(fig)
    
    def make_delay_chart3D(datas, xaxis, xlabel, title='', filename='', orientation_axis=0, invert_axis = False):
        beta_label = '$P_{l}$'
        other_axis = []
        beta_axis = []
        ez = []
        cz = []
        for g in datas[xaxis].values:
            for b in datas['beta'].values:
                filterd = datas.sel({xaxis:g}).sel(beta=b)
                delays = filterd['delaySum'].values
                communications = filterd['commCount'].values
                other_axis.append(g)
                beta_axis.append(b)
                cz.append((delays[0, -1] / communications[0, -1] * 1000))
                ez.append((delays[1, -1] / communications[1, -1] * 1000))
        fig = plt.figure(figsize=(7,4))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_title(title)
        ax.set_zlabel("Mean delay (ms)")
        if invert_axis:
            x = beta_axis
            y = other_axis
            x_label = beta_label
            y_label = xlabel
        else:
            x = other_axis
            y = beta_axis
            x_label = xlabel
            y_label = beta_label
            
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if orientation_axis == 0:
            ax.set_xlim([min(x), max(x)])
            ax.set_ylim([min(y), max(y)])
        if orientation_axis == 1:
            ax.set_xlim([max(x), min(x)])
            ax.set_ylim([min(y), max(y)])
        if orientation_axis == 2:
            ax.set_xlim([min(x), max(x)])
            ax.set_ylim([max(y), min(y)])
        if orientation_axis == 3:
            ax.set_xlim([max(x), min(x)])
            ax.set_ylim([max(y), min(y)])

        add_data_to_chart_delays(ax, x, y, cz, '$B$=cloud', len(colors) - 1)
        add_data_to_chart_delays(ax, x, y, ez, '$B$=edge', 1)
        ax.legend()
        plt.tight_layout()
        fig.savefig(filename)
        plt.close(fig)

    def make_double_line_chart(xdata, ydata1, ydata1Label, ydata1Color, ydata2, ydata2Label, ydata2Color, xlabel = '', ylabel = '', title = '', filename = ''):
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim([min(xdata), max(xdata)])
        ax.set_ylim([0,max(max(ydata1), max(ydata2))])
        ax.plot(xdata, ydata1, label=ydata1Label, color=ydata1Color, linewidth=1.0)
        ax.plot(xdata, ydata2, label=ydata2Label, color=ydata2Color, linewidth=1.0)
        ax.legend()
        plt.tight_layout()
        fig.savefig(filename)
        plt.close(fig)
        
    def make_log_double_line_chart(xdata, ydata1, ydata1Label, ydata1Color, ydata2, ydata2Label, ydata2Color, xlabel = '', ylabel = '', title = '', filename = ''):
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        plt.yscale('symlog')
        ax.set_ylabel(ylabel)
        ax.set_xlim([min(xdata), max(xdata)])
        ax.set_ylim([0,max(max(ydata1), max(ydata2))])
        ax.plot(xdata, ydata1, label=ydata1Label, color=ydata1Color, linewidth=1.0)
        ax.plot(xdata, ydata2, label=ydata2Label, color=ydata2Color, linewidth=1.0)
        ax.legend()
        plt.tight_layout()
        fig.savefig(filename)
        plt.close(fig)

    def generate_round_charts(datas, basedir=''):
        numSmartphoneNode = 300
        g_axis = []
        b_axis = []
        ez = []
        cz = []
        sz = []
        b_axis_with_g1 = []
        cz_with_g1 = []
        sz_with_g1_all = []
        sz_with_g1_each = []
        g_axis_with_b1 = []
        cz_with_b1 = []
        ez_with_b1 = []
        for g in datas['gamma'].values:
            for b in datas['beta'].values:
                filterd = datas.sel(gamma=g).sel(beta=b)
                cz.append(filterd['runOnCloudTot'].values[0, -1])
                ez.append(filterd['runOnEdgeTot'].values[0, -1])
                sz.append(filterd['runOnSmartphoneTot'].values[0, -1])
                g_axis.append(g)
                b_axis.append(b)
                if b == 1.0:
                    g_axis_with_b1.append(g)
                    cz_with_b1.append(filterd['runOnCloudTot'].values[0, -1])
                    ez_with_b1.append(filterd['runOnEdgeTot'].values[0, -1])
                if g == 1.0:
                    b_axis_with_g1.append(b)
                    cz_with_g1.append(filterd['runOnCloudTot'].values[0, -1])
                    sz_with_g1_all.append(filterd['runOnSmartphoneTot'].values[0, -1])
                    sz_with_g1_each.append(filterd['runOnSmartphoneTot'].values[0, -1] / (numSmartphoneNode * b))
        # 3D chart
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_title("Num of rounds (proxy of cost)")
        ax.set_xlabel("$P_{e}$")
        ax.set_ylabel("$P_{l}$")
        plt.zscale('symlog')
        ax.set_zlabel("num of round")
        ax.set_xlim([1.0, 0.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_zlim([0,max(max(cz), max(ez))])
        add_data_to_chart_delays(ax, g_axis, b_axis, cz, 'OnCloud', 2)
        add_data_to_chart_delays(ax, g_axis, b_axis, ez, 'OnEdge', 0)
        add_data_to_chart_delays(ax, g_axis, b_axis, sz, 'OnThermostats', 1)
        ax.legend()
        plt.tight_layout()
        fig.savefig(f'{basedir}{separator}rounds3D.pdf')
        plt.close(fig)
        # 2D by beta all smartphone
        filename = f'{basedir}{separator}roundsByPl-allThermostats.pdf'
        make_log_double_line_chart(b_axis_with_g1, cz_with_g1, 'OnCloud', colors[2], sz_with_g1_all, 'OnAllThermostats', colors[1], xlabel='$P_{l}$', ylabel='num of round', title='Num of rounds (cost proxy)', filename=filename)
        # 2D by beta each smartphoneplt.tight_layout()
        filename = f'{basedir}{separator}roundsByPl-eachThermostat.pdf'
        make_log_double_line_chart(b_axis_with_g1, cz_with_g1, 'OnCloud', colors[2], sz_with_g1_each, 'OnEachThermostat', colors[1], xlabel='$P_{l}$', ylabel='num of round', title='Num of rounds (cost proxy)', filename=filename)
        # 2D by gamma
        filename = f'{basedir}{separator}roundsByPe.pdf'
        make_log_double_line_chart(g_axis_with_b1, cz_with_b1, 'OnCloud', colors[2], ez_with_b1, 'OnEdge', colors[0], xlabel='$P_{e}$', ylabel='num of round', title='Num of rounds (cost proxy)', filename=filename)

    def generate_delay_charts(datas, errors = None, basedir=''):
        dir3D = f'{basedir}{separator}3D{separator}Pl-Pe'
        Path(dir3D).mkdir(parents=True, exist_ok=True)
        dir3Dbeta_delay = f'{basedir}{separator}3D{separator}Pl-delay'
        Path(dir3Dbeta_delay).mkdir(parents=True, exist_ok=True)
        dir2Dbeta = f'{basedir}{separator}2D{separator}Pl'
        Path(dir2Dbeta).mkdir(parents=True, exist_ok=True)
        dir2Dgamma = f'{basedir}{separator}2D{separator}Pe'
        Path(dir2Dgamma).mkdir(parents=True, exist_ok=True)
        for dee in datas.dee.values:
            for dcc in datas.dcc.values:
                for dec in datas.dec.values:
                    for dsCnd in datas.dsCnd.values:
                        title = '$d_{ee}=' + f'{dee}$ (ms)' + ' $d_{cc}=' + f'{dcc}$ (ms)' + ' $d_{ec}=' + f'{dec}$ (ms)' + ' $d_{a}=' + f'{dsCnd}$ (ms)'
                        figname = beautifyFigName(f'dee-{dee}-dcc-{dcc}-dec-{dec}-da-{dsCnd}')
                        filename3D = f'{dir3D}{separator}{figname}.pdf'
                        filename2Dbeta = f'{dir2Dbeta}{separator}{figname}.pdf'
                        filename2Dgamma = f'{dir2Dgamma}{separator}{figname}.pdf'
                        selectedData = datas.sel(dee=dee).sel(dcc=dcc).sel(dec=dec).sel(dsCnd=dsCnd)
                        make_delay_chart3D(selectedData, 'gamma', '$P_{e}$', title, filename3D)
                        make_delay_chart2D(selectedData.mean(dim = 'gamma', skipna = True), 'beta', '$P_{l}$', title, filename2Dbeta)
                        make_delay_chart2D(selectedData.mean(dim = 'beta', skipna = True), 'gamma', '$P_{e}$', title, filename2Dgamma)
        deeSecondValue = datas.dee.values[1]
        decSecondValue = datas.dec.values[1]
        dccSecondValue = datas.dcc.values[1]
        dscndSecondValue = datas.dsCnd.values[1]
        gammaSelected = 1.0
        for inverted in [True, False]:
            for i in [0, 1, 2, 3]:
                # dec
                title = '$d_{ee}=' + f'{deeSecondValue}$ (ms)' + ' $d_{cc}=' + f'{dccSecondValue}$ (ms)' + ' $d_{a}=' + f'{dscndSecondValue}$ (ms)'
                figname = beautifyFigName(f'{inverted}-{i}-dee-{deeSecondValue}-dcc-{dccSecondValue}-da-{dscndSecondValue}')
                filename3D = f'{dir3Dbeta_delay}{separator}{figname}.pdf'
                selectedData = datas.sel(gamma=gammaSelected).sel(dcc=dccSecondValue).sel(dee=deeSecondValue).sel(dsCnd=dscndSecondValue)
                make_delay_chart3D(selectedData, 'dec', '$d_{ec}$ (ms)', title, filename3D, i, inverted)
                # dee
                title = '$d_{cc}=' + f'{dccSecondValue}$ (ms)' + ' $d_{ec}=' + f'{decSecondValue}$ (ms)' + ' $d_{a}=' + f'{dscndSecondValue}$ (ms)'
                figname = beautifyFigName(f'{inverted}-{i}-dcc-{dccSecondValue}-dec-{decSecondValue}-da-{dscndSecondValue}')
                filename3D = f'{dir3Dbeta_delay}{separator}{figname}.pdf'
                selectedData = datas.sel(gamma=gammaSelected).sel(dcc=dccSecondValue).sel(dec=decSecondValue).sel(dsCnd=dscndSecondValue)
                make_delay_chart3D(selectedData, 'dee', '$d_{ee}$ (ms)', title, filename3D, i, inverted)
                # dcc
                title = '$d_{ee}=' + f'{deeSecondValue}$ (ms)' + ' $d_{ec}=' + f'{decSecondValue}$ (ms)' + ' $d_{a}=' + f'{dscndSecondValue}$ (ms)'
                figname = beautifyFigName(f'{inverted}-{i}-dee-{deeSecondValue}-dec-{decSecondValue}-da-{dscndSecondValue}')
                filename3D = f'{dir3Dbeta_delay}{separator}{figname}.pdf'
                selectedData = datas.sel(gamma=gammaSelected).sel(dee=deeSecondValue).sel(dec=decSecondValue).sel(dsCnd=dscndSecondValue)
                make_delay_chart3D(selectedData, 'dcc', '$d_{cc}$ (ms)', title, filename3D, i, inverted)
                # dsCnd
                title = '$d_{ee}=' + f'{deeSecondValue}$ (ms)' + ' $d_{cc}=' + f'{dccSecondValue}$ (ms)' + ' $d_{ec}=' + f'{decSecondValue}$ (ms)'
                figname = beautifyFigName(f'{inverted}-{i}-dee-{deeSecondValue}-dcc-{dccSecondValue}-dec-{decSecondValue}')
                filename3D = f'{dir3Dbeta_delay}{separator}{figname}.pdf'
                selectedData = datas.sel(gamma=gammaSelected).sel(dcc=dccSecondValue).sel(dec=decSecondValue).sel(dee=deeSecondValue)
                make_delay_chart3D(selectedData, 'dsCnd', '$d_{a}$ (ms)', title, filename3D, i, inverted)
            

    def generate_application_chart(means, errors = None, basedir=''):
        Path(basedir).mkdir(parents=True, exist_ok=True)
        datas = means.sel(dee=1.0).sel(dcc=25).sel(dec=50).sel(dsCnd=150).sel(gamma=0.33).sel(beta=0.33)
        time = datas[timeColumnName].values / 3600 # from seconds to hours
        varianceTemp = np.sqrt(datas['varianceTemp'].values[0, :])
        avgTemp = datas['avgTemp'].values[0, :]
        minTemp = datas['minTemp'].values[0, :]
        maxTemp = datas['maxTemp'].values[0, :]
        # generate complete chart
        fig = plt.figure(figsize = (6, 3))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('time (h)')
        ax.set_ylabel('temperature (°C)')
        line_width = 2.0
        ax.plot(time, avgTemp, label='mean$\pm$st.dev', color=colors[1], linewidth=line_width)
        ax.plot(time, avgTemp + varianceTemp, label=None, color=colors[1], linewidth=line_width / 2)
        ax.plot(time, avgTemp - varianceTemp, label=None, color=colors[1], linewidth=line_width / 2)
        ax.plot(time, minTemp, label='min', color=colors[2], linewidth=line_width)
        ax.plot(time, maxTemp, label='max', color=colors[0], linewidth=line_width)
        ax.set_xlim(minTime, maxTime / 3600)
        ax.axvline(25, color='#878787', linestyle='dashed', linewidth=line_width)
        ax.axvline(31, color='#878787', linestyle='dashed', linewidth=line_width)
        ax.axvline(61, color='#878787', linestyle='dashed', linewidth=line_width)
        ax.legend(frameon = False)
        fig.tight_layout()
        fig.savefig(f'{basedir}{separator}application.pdf')
        plt.close(fig)
        # generate chart with stdev
        fig = plt.figure(figsize = (6, 3))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('time (h)')
        ax.set_ylabel('temperature (°C)')
        line_width = 2.0
        ax.plot(time, avgTemp, label='mean$\pm$st.dev', color=colors[1], linewidth=line_width)
        ax.plot(time, avgTemp + varianceTemp, label=None, color=colors[1], linewidth=line_width / 2)
        ax.plot(time, avgTemp - varianceTemp, label=None, color=colors[1], linewidth=line_width / 2)
        ax.set_xlim(minTime, maxTime / 3600)
        ax.axvline(25, color='#878787', linestyle='dashed', linewidth=line_width)
        ax.axvline(31, color='#878787', linestyle='dashed', linewidth=line_width)
        ax.axvline(61, color='#878787', linestyle='dashed', linewidth=line_width)
        ax.legend(frameon = False)
        fig.tight_layout()
        fig.savefig(f'{basedir}{separator}application-with-stdev.pdf')
        plt.close(fig)
        # generate chart with min-max
        fig = plt.figure(figsize = (6, 3))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('time (h)')
        ax.set_ylabel('temperature (°C)')
        line_width = 2.0
        ax.plot(time, avgTemp, label='mean', color=colors[1], linewidth=line_width)
        ax.plot(time, minTemp, label='min', color=colors[2], linewidth=line_width)
        ax.plot(time, maxTemp, label='max', color=colors[0], linewidth=line_width)
        ax.set_xlim(minTime, maxTime / 3600)
        ax.axvline(25, color='#878787', linestyle='dashed', linewidth=line_width)
        ax.axvline(31, color='#878787', linestyle='dashed', linewidth=line_width)
        ax.axvline(61, color='#878787', linestyle='dashed', linewidth=line_width)
        ax.legend(frameon = False)
        fig.tight_layout()
        fig.savefig(f'{basedir}{separator}application-with-min-max.pdf')
        plt.close(fig)

    def generate_charts(means, errors = None, basedir=''):
        applicationDir = f'{basedir}{separator}application'
        roundDir = f'{basedir}{separator}protelis-rounds'
        delayDir = f'{basedir}{separator}delays'
        Path(applicationDir).mkdir(parents=True, exist_ok=True)
        Path(roundDir).mkdir(parents=True, exist_ok=True)
        Path(delayDir).mkdir(parents=True, exist_ok=True)
        data = means.sel(dLocalhost=0.02)
        generate_application_chart(data, basedir=applicationDir)
        generate_delay_charts(data, basedir=delayDir)
        generate_round_charts(data.sel(dee=1.0).sel(dcc=25).sel(dec=50).sel(dsCnd=150), basedir=roundDir)

    for experiment in experiments:
        current_experiment_means = means[experiment]
        generate_charts(current_experiment_means, basedir=f'{output_directory}{separator}custom')
        
