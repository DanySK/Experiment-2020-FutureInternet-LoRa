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
    directory = 'dataRound'
    # Where to save charts
    output_directory = 'charts'
    # How to name the summary of the processed data
    pickleOutput = 'round_data_summary'
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
            lastTimeProcessed = pickle.load(open('round_timeprocessed', 'rb'))
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
            pickle.dump(newestFileTime, open('round_timeprocessed', 'wb'))
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
    colors_length = 4
    colors = [cmx.viridis(float(i)/(colors_length - 1)) for i in range(colors_length)]
    
    def compute_protelis_broker_cost():
        node_count = 310
        neighboor = 49
        daily_connection_minutes = 24 * 60
        daily_round = 24 * 4
        daily_connection_cost = daily_connection_minutes * 0.096 / 1_000_000
        daily_message_cost = (daily_round + daily_round * neighboor) * 1.20 / 1_000_000
        daily_broker_cost = (daily_connection_cost + daily_message_cost) * node_count
        return daily_broker_cost

    def compute_lora_gateway_broker_cost():
        gateway_count = 9
        mote_per_gateway = 10
        daily_connection_minutes = 24 * 60
        daily_lora_per_gateway = 24 * mote_per_gateway
        daily_connection_cost = daily_connection_minutes * 0.096 / 1_000_000
        daily_message_cost = daily_lora_per_gateway * 2 * 1.20 / 1_000_000
        daily_broker_cost = (daily_connection_cost + daily_message_cost) * gateway_count
        return daily_broker_cost
    
    def compute_lora_server_broker_cost():
        gateway_count = 9
        mote_per_gateway = 10
        daily_connection_minutes = 24 * 60
        daily_lora_per_gateway = 24 * mote_per_gateway
        daily_connection_cost = daily_connection_minutes * 0.096 / 1_000_000
        daily_message_cost = (daily_lora_per_gateway * gateway_count + daily_lora_per_gateway) * 1.20 / 1_000_000
        daily_broker_cost = daily_connection_cost + daily_message_cost
        return daily_broker_cost
    
    def compute_broker_cost():
        return compute_protelis_broker_cost() + compute_lora_gateway_broker_cost() + compute_lora_server_broker_cost()
    
    def num_of_instance(node_count):
        if node_count == 0:
            return 0
        if node_count <= 100:
            return 1
        if node_count <= 200:
            return 2
        return 3
    
    def compute_instance_cost(node_count):
        return num_of_instance(node_count) * 0.0291 * 24
    
    def compute_cloud_cost(node_count, withBroker):
        num_of_day = 1
        if withBroker:
            return (compute_broker_cost() + compute_instance_cost(node_count)) * num_of_day
        return compute_instance_cost(node_count) * num_of_day
    
    def compute_energy_cost(proportion_device_on_thermostat, withBroker):
        device_count = 300
        daily_round_per_device = 4 * 24
        daily_round = device_count * proportion_device_on_thermostat * daily_round_per_device
        time = 24 * 60 * 60
        time_per_round = 0.1
        idle_consumption = 0.25
        round_consumption = 1.25 - idle_consumption
        daily_joule = device_count * idle_consumption * time + daily_round * round_consumption * time_per_round
        price_per_joule = 9.3e-8
        energy_cost = daily_joule * price_per_joule
        if withBroker:
            return compute_broker_cost() + energy_cost
        return energy_cost 
    
    def make_four_line_chart(xdata, ydata1, ydata1Label, ydata1Color, ydata2, ydata2Label, ydata2Color, ydata3, ydata3Label, ydata3Color, ydata4, ydata4Label, ydata4Color, xlabel = '', ylabel = '', ylabel3 = '', title = '', filename = ''):
        fig = plt.figure(figsize=(6,3))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim([0, 1])
        # ax.set_yscale('symlog', linthreshy=6)
        # ax.set_ylim([0,max(max(ydata1), max(ydata2), max(ydata3)) + 2])
        ax.plot(xdata, ydata1, label=ydata1Label, color=ydata1Color, linewidth=2.0)
        ax.plot(xdata, ydata2, label=ydata2Label, color=ydata2Color, linewidth=2.0)
        ax.plot(xdata, ydata3, label=ydata3Label, color=ydata3Color, linewidth=2.0)
        ax.plot(xdata, ydata4, label=ydata4Label, color=ydata4Color, linewidth=2.0)
        
        ax.legend()
        plt.tight_layout()
        fig.savefig(filename)
        plt.close(fig)
    
    def make_three_line_chart(xdata, ydata1, ydata1Label, ydata1Color, ydata2, ydata2Label, ydata2Color, ydata3, ydata3Label, ydata3Color, xlabel = '', ylabel = '', ylabel3 = '', title = '', filename = ''):
        fig = plt.figure(figsize=(7,4))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim([0, 1])
        # ax.set_yscale('symlog', linthreshy=6)
        ax.set_ylim([0,max(max(ydata1), max(ydata2), max(ydata3)) + 0.5])
        ax.plot(xdata, ydata1, label=ydata1Label, color=ydata1Color, linewidth=2.0)
        ax.plot(xdata, ydata2, label=ydata2Label, color=ydata2Color, linewidth=2.0)
        ax.plot(xdata, ydata3, label=ydata3Label, color=ydata3Color, linewidth=2.0)
        
        ax.legend()
        plt.tight_layout()
        fig.savefig(filename)
        plt.close(fig)
    
    def make_two_line_chart(xdata, ydata1, ydata1Label, ydata1Color, ydata2, ydata2Label, ydata2Color, xlabel = '', ylabel = '', ylabel3 = '', title = '', filename = ''):
        fig = plt.figure(figsize=(7,4))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim([0, 1])
        # ax.set_yscale('symlog', linthreshy=6)
        # ax.set_ylim([0,max(max(ydata1), max(ydata2), max(ydata3)) + 2])
        ax.plot(xdata, ydata1, label=ydata1Label, color=ydata1Color, linewidth=2.0)
        ax.plot(xdata, ydata2, label=ydata2Label, color=ydata2Color, linewidth=2.0)
        
        ax.legend()
        plt.tight_layout()
        fig.savefig(filename)
        plt.close(fig)
        
    def make_log_double_line_chart(xdata1, ydata1, ydata1Label, ydata1Color, xdata2, ydata2, ydata2Label, ydata2Color, xlabel = '', ylabel = '', title = '', filename = ''):
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        plt.yscale('symlog')
        ax.set_ylabel(ylabel)
        ax.set_xlim([min(xdata1), max(xdata1)])
        ax.set_ylim([0,max(max(ydata1), max(ydata2))])
        ax.plot(xdata1, ydata1, label=ydata1Label, color=ydata1Color, linewidth=1.0)
        ax.plot(xdata2, ydata2, label=ydata2Label, color=ydata2Color, linewidth=1.0)
        ax.legend()
        plt.tight_layout()
        fig.savefig(filename)
        plt.close(fig)

    def generate_round_charts(datas, basedir=''):
        numSmartphoneNode = 300
        g_axis = []
        b_axis = []
        b_axis_with_g1 = []
        cz_with_g1 = []
        sz_with_g1_all = []
        sz_with_g1_each = []
        for g in datas['gamma'].values:
            for b in datas['beta'].values:
                filterd = datas.sel(gamma=g).sel(beta=b)
                g_axis.append(g)
                b_axis.append(b)
                if g == 1.0:
                    b_axis_with_g1.append(b)
                    cz_with_g1.append(filterd['runOnCloudTot'].values[0, -1])
                    runOnSmartphoneTot = filterd['runOnSmartphoneTot'].values[0, -1]
                    sz_with_g1_all.append(runOnSmartphoneTot)
                    if b > 0.0:
                        sz_with_g1_each.append(runOnSmartphoneTot / int(numSmartphoneNode * b))
        cost_with_cloud = []
        cost_without_cloud = []
        energy_cost = []
        energy_cost_with_broker = []
        broker_cost = []
        total_cost_with_broker = []
        total_cost_without_broker = []
        
        x = np.linspace(0.0, 1.0, 301)
        for b in x:
            cost_with_cloud.append(compute_cloud_cost(int(numSmartphoneNode * (1 - b)), True))
            cost_without_cloud.append(compute_cloud_cost(int(numSmartphoneNode * (1 - b)), False))
            energy_cost_with_broker.append(compute_energy_cost(b, True))
            energy_cost.append(compute_energy_cost(b, False))
            broker_cost.append(compute_broker_cost())
            total_cost_with_broker.append(compute_cloud_cost(int(numSmartphoneNode * (1 - b)), True) + compute_energy_cost(b, False))
            total_cost_without_broker.append(compute_cloud_cost(int(numSmartphoneNode * (1 - b)), False) + compute_energy_cost(b, False))
        # 2D by beta all smartphone
        filename = f'{basedir}{separator}roundsByPl-allThermostats.pdf'
        make_log_double_line_chart(b_axis_with_g1, cz_with_g1, 'OnCloud', colors[2], b_axis_with_g1, sz_with_g1_all, 'OnAllThermostats', colors[1], xlabel='$P_{l}$', ylabel='rounds count', title='Rounds count (cost proxy)', filename=filename)
        # 2D by beta each smartphoneplt.tight_layout()
        filename = f'{basedir}{separator}roundsByPl-eachThermostat.pdf'
        make_log_double_line_chart(b_axis_with_g1, cz_with_g1, 'OnCloud', colors[2], b_axis_with_g1[1: ], sz_with_g1_each, 'OnEachThermostat', colors[1], xlabel='$P_{l}$', ylabel='rounds count', title='Rounds count (cost proxy)', filename=filename)
        # cost
#        filename = f'{basedir}{separator}deplyment-cost-four-line.pdf'
#        make_four_line_chart(x, cost_with_cloud, 'cost with B=cloud', colors[2], cost_without_cloud, 'cost with B=edge', colors[1], energy_cost, 'energy cost with B=edge', colors[0], energy_cost_with_broker, 'energy cost with B=cloud', colors[3], xlabel='$P_{l}$', ylabel=r'Upkeep($\$/day$)', ylabel3=r'$\$/$', title='Upkeep cost estimation', filename=filename)   
        filename = f'{basedir}{separator}upkeep-cost-breakdown.pdf'
        make_three_line_chart(x, cost_without_cloud, 'cloud infrastructure', colors[2], broker_cost, 'broker cloud hosting', colors[1], energy_cost, 'end device electricity', colors[0], xlabel='$P_{l}$', ylabel=r'Upkeep ($\$/day$)', ylabel3=r'$\$/$', title='Upkeep cost breakdown', filename=filename)        
        filename = f'{basedir}{separator}estimated-upkeep-cost.pdf'
        make_two_line_chart(x, total_cost_with_broker, 'B=cloud', colors[3], total_cost_without_broker, 'B=edge', colors[1], xlabel='$P_{l}$', ylabel=r'Upkeep ($\$/day$)', ylabel3=r'$\$/$', title='Estimated upkeep cost', filename=filename)        
        
    def generate_charts(means, errors = None, basedir=''):
        roundDir = f'{basedir}{separator}protelis-rounds-new'
        Path(roundDir).mkdir(parents=True, exist_ok=True)
        data = means.sel(dLocalhost=0.02)
        generate_round_charts(data.sel(dee=1.0).sel(dcc=25).sel(dec=50).sel(dsCnd=150), basedir=roundDir)

    for experiment in experiments:
        current_experiment_means = means[experiment]
        generate_charts(current_experiment_means, basedir=f'{output_directory}{separator}custom')
        
