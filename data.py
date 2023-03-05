import openpyxl
import pandas

fname = "HiggsData.xlsx"
diff = 12
print(f"Attempting to read {fname}...")
book = openpyxl.load_workbook(fname)

### 7+8 TeV
shname = '7och8 TeV'
print(f"Loading data for {shname}")
sh78 = book[shname]

cols = ['ggF','VBF','VH','ttH']
rows = ['bb','cc','tt','mumu','gamgam','gg','WW','ZZ','Zgam']

mu_atlas = sh78['B2:E10']

df = pandas.DataFrame(mu_atlas,index=rows, columns=cols)
print(df['VBF']['bb'].value)

# mu_ggF_bb = [sh78[f'B2'].value, sh78[f'B{2+2*diff}'].value]
# unc_ggF_bb = [sh78[f'B{2+diff}'].value, sh78[f'B{2+3*diff}'].value]
mu_VBF_bb = [sh78[f'C2'].value]
unc_VBF_bb = [sh78[f'C{2+diff}'].value]
mu_VH_bb = [sh78[f'D2'].value, sh78[f'D{2+2*diff}'].value]
unc_VH_bb = [sh78[f'D{2+diff}'].value, sh78[f'D{2+3*diff}'].value]
mu_ttH_bb = [sh78[f'E2'].value, sh78[f'E{2+2*diff}'].value]
unc_ttH_bb = [sh78[f'E{2+diff}'].value, sh78[f'E{2+3*diff}'].value]

mu_ggF_tt = [sh78[f'B4'].value, sh78[f'B{4+2*diff}'].value]
unc_ggF_tt = [sh78[f'B{4+diff}'].value, sh78[f'B{4+3*diff}'].value]
mu_VBF_tt = [sh78[f'C4'].value, sh78[f'C{4+2*diff}'].value]
unc_VBF_tt = [sh78[f'C{4+diff}'].value, sh78[f'C{4+3*diff}'].value]
mu_VH_tt = [sh78[f'D4'].value, sh78[f'D{4+2*diff}'].value]
unc_VH_tt = [sh78[f'D{4+diff}'].value, sh78[f'D{4+3*diff}'].value]
mu_ttH_tt = [sh78[f'E4'].value, sh78[f'E{4+2*diff}'].value]
unc_ttH_tt = [sh78[f'E{4+diff}'].value, sh78[f'E{4+3*diff}'].value]

mu_ggF_gg = [sh78[f'B6'].value, sh78[f'B{6+2*diff}'].value]
unc_ggF_gg = [sh78[f'B{6+diff}'].value, sh78[f'B{6+3*diff}'].value]
mu_VBF_gg = [sh78[f'C6'].value, sh78[f'C{6+2*diff}'].value]
unc_VBF_gg = [sh78[f'C{6+diff}'].value, sh78[f'C{6+3*diff}'].value]
mu_VH_gg = [sh78[f'D6'].value, sh78[f'D{6+2*diff}'].value]
unc_VH_gg = [sh78[f'D{6+diff}'].value, sh78[f'D{6+3*diff}'].value]
mu_ttH_gg = [sh78[f'E6'].value, sh78[f'E{6+2*diff}'].value]
unc_ttH_gg = [sh78[f'E{6+diff}'].value, sh78[f'E{6+3*diff}'].value]

mu_ggF_ww = [sh78[f'B8'].value]
unc_ggF_ww = [sh78[f'B{8+diff}'].value]
mu_VBF_ww = [sh78[f'C8'].value, sh78[f'C{8+2*diff}'].value]
unc_VBF_ww = [sh78[f'C{6+diff}'].value, sh78[f'C{6+3*diff}'].value]
mu_VH_ww = [sh78[f'D8'].value, sh78[f'D{8+2*diff}'].value]
unc_VH_ww = [sh78[f'D{6+diff}'].value, sh78[f'D{6+3*diff}'].value]
mu_ttH_ww = [sh78[f'E8'].value]
unc_ttH_ww = [sh78[f'E{8+diff}'].value]

mu_ggF_zz = [sh78[f'B9'].value, sh78[f'B{9+2*diff}'].value]
unc_ggF_zz = [sh78[f'B{9+diff}'].value, sh78[f'B{9+3*diff}'].value]
mu_VBF_zz = [sh78[f'C9'].value, sh78[f'C{9+2*diff}'].value]
unc_VBF_zz = [sh78[f'C{9+diff}'].value, sh78[f'C{9+3*diff}'].value]
mu_VH_zz = [sh78[f'D{9+2*diff}'].value]
unc_VH_zz = [sh78[f'D{9+3*diff}'].value]
mu_ttH_zz = [sh78[f'E9'].value, sh78[f'E{9+2*diff}'].value]
unc_ttH_zz = [sh78[f'E{9+diff}'].value, sh78[f'E{9+3*diff}'].value]
