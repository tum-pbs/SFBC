import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
st.set_page_config(layout="wide")
st.title('Ablation Study Visualizer')

import os

# @st.cache_data
def file_selector(folder_path='/ablationStudies'):
    folder_path = os.path.dirname(__file__) + folder_path
    # st.write(folder_path)
    filenames = os.listdir(folder_path)
    # st.write(filenames)
    filenames = [f for f in filenames if f.endswith('.csv')]
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

with st.sidebar:
    filename = file_selector()

loadAll = st.sidebar.radio("Load All Files",[False, True])


ablationStudy = pd.read_csv(filename)
if loadAll:
    ablationStudy = pd.DataFrame()
    filenames = os.listdir('./ablationStudies/')
    filenames = ['./ablationStudies/' + f for f in filenames if f.endswith('.csv')]
    for f in filenames:
        ablationStudy = pd.concat((ablationStudy, pd.read_csv(f)), ignore_index=True)



import seaborn as sns

def getBasisLabel(b):
    if b == 'fourier even':
        return 'Fourier (even)'
    if b == 'fourier odd':
        return 'Fourier (odd)'
    if b == 'fourier':
        return 'SFBC'
    if b == 'ffourier':
        return 'SFBC'
    if b == 'linear':
        return 'LinCConv'
    if b == 'abf cubic_spline':
        return 'SplineConv'
    if b == 'dmcf':
        return 'DMCF'
    if b == 'rbf square':
        return 'Nearest Neighbor'
    if b == 'chebyshev':
        return 'Chebyshev'
    st.write('unknown basis function', b)
    return 'UNKNOWN'
def getWindowLabel(w):
    if w == 'poly6':
        return 'Spiky'
    if w == 'Spiky':
        return 'Spiky Kernel'
    if w == 'Parabola':
        return r'$1-x^2$'
    if w == 'cubicSpline':
        return 'Cubic Spline'
    if w == 'quarticSpline':
        return 'Quartic Spline'
    if w == 'Linear':
        return r'$1-x$'
    if w == 'None':
        return 'None'
    if w == 'Mueller':
        return 'Müller'
    if np.isnan(w):
        return 'None'
    st.write('unknown window function', w)
    return 'UNKNOWN'
def getMappingLabel(w):
    if w == 'cartesian':
        return 'Identity'
    if w == 'polar':
        return 'Polar Coordinates'
    if w == 'preserving':
        return 'Ummenhofer et al'
    st.write('unknown window function', w)
    return 'UNKNOWN'
def getFeatureLabel(f):
    # if f == 'normalizedVolume':
    #     return r'$\hat{V_i}$'
    # if f == 'rho':
    #     return r'$\rho_i$'
    # if f == 'volumesOverRho':


    if f == 'one':
        return r'$1$'
    elif f == 'zero':
        return r'$0$'
    elif f == 'volume':
        return r'$V_i$'
    elif f == 'normalizedVolume':
        return r'$\hat{V}_i$'
    elif f == 'ni':
        return r'$\#\mathcal{N}_i$'
    elif f == 'rho':
        return r'$\rho_i$'
    elif f == 'rhoTimesVolume':
        return r'$\rho_i\cdot V_i$'
    elif f == 'rhoTimesNormalizedVolume':
        return r'$\rho_i\cdot \hat{V}_i$'
    elif f == 'rhoOverVolume':
        return r'$\frac{rho_i}{{V}_i}$'
    elif f == 'rhoOverNormalizedVolume':
        return r'$\frac{rho_i}{\hat{V}_i}$'
    elif f == 'volumeOverRho':
        return r'$\frac{{V}_i}{\rho_i}$'
    elif f == 'normalizedVolumeOverRho':
        return r'$\frac{\hat{V}_i}{\rho_i}$'
    elif f == 'frequency':
        return r'$f$'
    elif f == 'seed':
        return r'seed'

    # st.write('unknown features', w)
    raise Exception("Unknown Features %s" % f)
    return 'UNKNOWN'


def getFeaturesLabel(w):
    features = w.split(' ')
    # st.write(features)
    s = [getFeatureLabel(f) for f in features]
    return '[%s]'% ', '.join(s)

def getTargetLabel(w):
    if w == 'rho':
        return r'$\rho$'
    if w == 'gradRhoNaive':
        return r'$\nabla\rho$ [Naïve]'
    if w == 'gradRhoDifference':
        return r'$\nabla\rho$ [Difference]'
    if w == 'gradRhoSymmetric':
        return r'$\nabla\rho$ [Symmetric]'
    st.write('unknown target', w)
    return 'UNKNOWN'

ablationStudy['base'] = ablationStudy.base.apply(getBasisLabel)
ablationStudy['window'] = ablationStudy.window.apply(getWindowLabel)
ablationStudy['mapping'] = ablationStudy.mapping.apply(getMappingLabel)
ablationStudy['Configuration'] = ablationStudy[['window','mapping']].apply(lambda x: 'Window: ' + x[0] + ' @ Map: ' + x[1], axis = 1)
ablationStudy['features'] = ablationStudy.features.apply(getFeaturesLabel)
ablationStudy['target'] = ablationStudy.target.apply(getTargetLabel)

ablationStudy = ablationStudy.rename(columns = {
    'n' : 'Base Function Count',
    'base': 'Base Function',
    'features': "Features",
    'target': 'Target',
    'mapping': 'Coordinate Mapping',
    'window': 'Window Function',
    'seed':'Dataset Seed',
    'networkSeed': 'Netork Seed',
    'arch':'Network Architecture',
    'meanLoss':'RMSE'})



relevantHeaders = [     'Base Function Count',
                        'Base Function',
                        "Features",
                        'Target',
                        'Coordinate Mapping',
                        'Window Function',
                        'Dataset Seed',
                        'Netork Seed',
                        'Network Architecture']

filterDict = {}
with st.sidebar.expander("Filters"):
    for h in relevantHeaders:
        filterDict[h] = st.multiselect(
        'Filter for %s' % h,
        [str(s) for s in ablationStudy[h].unique()],
        [str(s) for s in ablationStudy[h].unique()])

# st.write(ablationStudy.keys())
methods = ['Nearest Neighbor', 'LinCConv', 'SplineConv', 'Gaussian Kernel', 'Müller Kernel', 'Wendland-2', 'Quartic Spline', 'Chebyshev', 'Chebyshev (2nd kind)', 'Fourier (even)', 'Fourier (odd)', 'Fourier', 'DMCF', 'PointNet', 'DPCConv', 'GNS', 'MP-PDE', 'Spiky', 'Bump RBF']


# g = sns.catplot(data = ablationStudy[ablationStudy['base']!='fourier'], x = 'n', y = 'meanLoss', hue = 'base', aspect = 2.0, capsize = .05, kind = 'bar')




def translateVar(varSelect):
    # if varSelect == 'Base Function Count':
    #     return 'n'
    # if varSelect == 'Base Function':
    #     return 'base'
    # if varSelect == 'Features':
    #     return 'features'
    # if varSelect == 'Target':
    #     return 'target'
    # if varSelect == 'Coordinate Mapping':
    #     return 'mapping'
    # if varSelect == 'Window Function':
    #     return 'window'
    # if varSelect == 'Dataset Seed':
    #     return 'seed'
    # if varSelect == 'Network Seed':
    #     return 'networkSeed'
    # if varSelect == 'Network Architecture':
    #     return 'arch'
    # if varSelect == 'Configuration':
    #     return 'Configuration'
    # if varSelect == 'RMSE':
    #     return 'meanLoss'
    if varSelect == 'Percentile':
        return 'pi'
    if varSelect == 'Confidence':
        return 'ci'
    return varSelect


xVar = translateVar(st.sidebar.selectbox(
    "x-Variable",
    ("Base Function Count", "Base Function", "Features", "Target", "Coordinate Mapping", 'Window Function', 'Dataset Seed', 'Network Seed', 'Network Architecture')
))
yVar = translateVar(st.sidebar.selectbox(
    "y-Variable",
    ("RMSE", None)
))

hueVar = translateVar(st.sidebar.selectbox(
    "Hue",
    ("Base Function","Base Function Count",  "Features", "Target", "Coordinate Mapping", 'Window Function', 'Dataset Seed', 'Network Seed', 'Network Architecture', None)
))
rowVar = translateVar(st.sidebar.selectbox(
    "row-Variable",
    (None, "Base Function Count", "Base Function", "Features", "Target", "Coordinate Mapping", 'Window Function', 'Dataset Seed', 'Network Seed', 'Network Architecture')
))
colVar = translateVar(st.sidebar.selectbox(
    "col-Variable",
    (None, "Base Function Count", "Base Function", "Features", "Target", "Coordinate Mapping", 'Window Function', 'Dataset Seed', 'Network Seed', 'Network Architecture')
))

styleVar = translateVar(st.sidebar.selectbox(
    "style-Variable",
    (None, "Base Function Count", "Base Function", "Features", "Target", "Coordinate Mapping", 'Window Function', 'Dataset Seed', 'Network Seed', 'Network Architecture')
))

errorVar = translateVar(st.sidebar.selectbox(
    "Error Bar",
    ("Percentile", "Confidence", None)
))

yLog = st.sidebar.radio("Log Y-Axis",[True, False])

plotVar = st.sidebar.selectbox(
    "Plot Style",
    ("Bar", "Boxen")
)


if hueVar == 'Base Function':
    # palette = sns.color_palette('tab20', len(ablationStudy[hueVar].unique()))
    actualBases = ablationStudy['Base Function'].unique()
    actualBases = [b for b in actualBases if b in methods]
    if 'SFBC' in ablationStudy['Base Function'].unique():
        actualBases.append('Fourier')
    actualBases.sort(key = lambda i: methods.index(i))
    # st.write(actualBases)
    # if 'SFBC' in ablationStudy['Base Function'].unique():
        # actualBases[actualBases.index('Fourier')] = 'SFBC'

    palette = sns.color_palette('tab20', len(methods))
    # palette[0] = (0,1,0)
    # st.write(actualBases)
    # st.write(actualBases.index('LinCConv'))
    actualPalette = [None] * len(actualBases)
    for b in actualBases:
        if b == 'SFBC': continue
        actualPalette[actualBases.index(b)] = palette[methods.index(b)]
    if 'Fourier' in actualBases:
        actualPalette[actualBases.index('Fourier')] = (0,0,1)
    if 'SFBC' in actualBases:
        actualPalette[actualBases.index('SFBC')] = (0,0,1)
    if 'SFBC' in ablationStudy['Base Function'].unique():
        actualBases[actualBases.index('Fourier')] = 'SFBC'
    if 'LinCConv' in actualBases:
        actualPalette[actualBases.index('LinCConv')] = (1,0,0)
    palette = actualPalette
else:
    palette = sns.color_palette('tab20', len(ablationStudy[hueVar].unique()))

titanic = sns.load_dataset("titanic")

# print('Filtering')
for k in relevantHeaders:
    # st.write(k, filterDict[k])
    # print(filterDict[k])
    ablationStudy = ablationStudy[ablationStudy[k].apply(lambda x: str(x)).isin(filterDict[k])]
    # print(ablationStudy)
    # st.write(ablationStudy)


fig = plt.figure(figsize=(10, 4))
sns.countplot(x="class", data=titanic)

if plotVar == 'Boxen':
    g = sns.catplot(data = ablationStudy, 
            x = xVar, y = yVar, hue = hueVar, hue_order = None if hueVar !='Base Function' else actualBases,
            aspect = 1.5, #capsize = .15, 
            kind = 'boxen', errorbar = (errorVar), row = rowVar, col = colVar, 
            palette = palette)
if plotVar == 'Bar':
    g = sns.catplot(data = ablationStudy, 
        x = xVar, y = yVar, hue = hueVar, hue_order = None if hueVar !='Base Function' else actualBases,
        aspect = 1.5, capsize = .15, 
        kind = 'bar', errorbar = (errorVar), row = rowVar, col = colVar, 
        palette = palette)
if plotVar == 'Line':
    g = sns.relplot(data = ablationStudy, 
        x = xVar, y = yVar, hue = hueVar, hue_order = None if hueVar !='Base Function' else actualBases,
        aspect = 1., style = styleVar,
        kind = 'line', errorbar = (errorVar), row = rowVar, col = colVar, 
        palette = palette, facet_kws = {'sharey': False})
    

for ax in g.axes.flatten():
    if yLog:
        ax.set_yscale('log')
    ax.set_axisbelow(True)
    ax.grid(which = 'major', axis = 'y', ls= '--')


g.fig.subplots_adjust(bottom = 0.15, left = 0.05, right = 0.98, top = 0.95)
sns.move_legend(g, "lower center", bbox_to_anchor=(.5,0), title = '', ncol = 12)
plt.savefig('ablation.pdf', format = 'pdf')
st.pyplot(g)

# g.axes[0,0].set_yscale('log')
# # g.axes[0,1].set_title('R2')
# # g.axes[0,2].set_title('PSNR')
# g.axes[0,0].grid(which = 'major', axis = 'y')
# g.axes[0,1].grid(which = 'major', axis = 'y')
# g.axes[0,2].grid(which = 'major', axis = 'y')
# g.axes[0,0].set_axisbelow(True)
# g.axes[0,1].set_axisbelow(True)
# g.axes[0,2].set_axisbelow(True)
# g.set_xticklabels(rotation=40, ha = 'right') 

# g.axes[0,0].set_title(r'Target: $\rho$, Coordinate Mapping: Identity')
# g.axes[0,0].set_title(r'Target: $\rho$, Coordinate Mapping: Polar')
# g.axes[0,0].set_title(r'Target: $\rho$, Coordinate Mapping: Preserving')
# g.axes[0,0].set_ylabel('Mean Testing Loss')
# g.axes[0,0].set_xlabel('Input Feature')
# g.axes[0,1].set_xlabel('Input Feature')
# g.axes[0,2].set_xlabel('Input Feature')
# g.set_xticklabels(g.get_xticklabels(), rotation=30)
# g.tight_layout()

# print(ablationFiles)


df = pd.DataFrame(
[{"Key" : k, "Count": len(ablationStudy[k].unique()), "Values" : ', '.join([str(s) for s in ablationStudy[k].unique()])} for k in relevantHeaders])


st.write('Dataset Columns: [%s]' % (', '.join([str(s) for s in ablationStudy.keys()])))
st.table(df.set_index(df.columns[0]))