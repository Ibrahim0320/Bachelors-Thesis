import openpyxl
import pandas
import numbers
import numpy as np
from bcolors import bcolors

print(f"{bcolors.WARNING}Welcome to data.py{bcolors.ENDC}")

def cellval(cell):
    if isinstance(cell.value, numbers.Number):
        return cell.value
    elif isinstance(cell.value, str) and cell.value.startswith("="):
        return eval(cell.value[1:])
    elif type(cell.value) == type(None):
        return None
    else:
        raise TypeError(f"Expected number or addition expression, got '{cell.value}' in {cell}")

fname = "HiggsData.xlsx"
diff = 12
cols = ['ggF','VBF','VH','ttH']
rows = ['bb','cc','tt','mumu','gamgam','gg','WW','ZZ','Zgam']

print(f"Attempting to read {fname}...")
book = openpyxl.load_workbook(fname)

### 7+8 TeV
shname = '7 and 8 TeV'
print(f"Loading data for {shname}...")
sheet = book[shname]

mu_atlas_78 = pandas.DataFrame(sheet['B2:E10'],index=rows, columns=cols).applymap(cellval)
unc_atlas_78 = pandas.DataFrame(sheet[f'B{2+diff}:E{10+diff}'],index=rows, columns=cols).applymap(cellval)
mu_cms_78 = pandas.DataFrame(sheet[f'B{2+2*diff}:E{10+2*diff}'],index=rows, columns=cols).applymap(cellval)
unc_cms_78 = pandas.DataFrame(sheet[f'B{2+3*diff}:E{10+3*diff}'],index=rows, columns=cols).applymap(cellval)

shname = '13 TeV'
print(f"Loading data for {shname}...")
sheet = book[shname]

mu_atlas_13 = pandas.DataFrame(sheet['B2:E10'],index=rows, columns=cols).applymap(cellval)
unc_atlas_13 = pandas.DataFrame(sheet[f'B{2+diff}:E{10+diff}'],index=rows, columns=cols).applymap(cellval)
mu_cms_13 = pandas.DataFrame(sheet[f'B{2+2*diff}:E{10+2*diff}'],index=rows, columns=cols).applymap(cellval)
unc_cms_13 = pandas.DataFrame(sheet[f'B{2+3*diff}:E{10+3*diff}'],index=rows, columns=cols).applymap(cellval)

m_78 = pandas.DataFrame(((mu_atlas_78, unc_atlas_78),(mu_cms_78, unc_cms_78)), index=['atlas', 'cms'], columns=['mu','unc'])
m_13 = pandas.DataFrame(((mu_atlas_13, unc_atlas_13),(mu_cms_13, unc_cms_13)), index=['atlas', 'cms'], columns=['mu','unc'])

print("Succesfully loaded signal strengths! :)")
print("Now loading branching ratios...")
sheet = book['Branching Ratios']
br = pandas.DataFrame(sheet['C2:C10'], index=rows).applymap(cellval).to_dict()[0]

print(f"{bcolors.OKGREEN}Loaded all data!!!{bcolors.ENDC}")


def hd(data_type, prodmode, finalmode, energy=''):
    """
    hd creates arrays of LHC data

    :param data_type: 'mu' (averages) or 'unc' (uncertainties)
    :param prodmode: production mode in LHC
    :param finalmode: final decay mode in LHC
    :param energy: 78 (7 and 8 TeV) or 13 (13 TeV). Returns both energies if left empty
    :return: Returns an array
    """

    ret = []
    if energy != '13':
        [ret.append(mat[prodmode][finalmode]) for mat in m_78[data_type] if not np.isnan(mat[prodmode][finalmode])]
    if energy != '78':
        [ret.append(mat[prodmode][finalmode]) for mat in m_13[data_type] if not np.isnan(mat[prodmode][finalmode])]
        
    return ret
