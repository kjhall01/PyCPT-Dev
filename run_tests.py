from pycpt_functions_seasonal import *
import sys
import time

workdir = "/Users/kylehall/Projects/PYCPT/PyCPT-Dev/"
cptdir='/Users/kylehall/CPT/16.5.8/'
shp_file = 'None' #path from pycpt_functions_seasonal.py to your shape file
use_default = 'True' #Make False to turn off default country boundaries
station = False
xmodes_min = 1
xmodes_max = 8
ymodes_min = 2
ymodes_max = 10
ccamodes_min = 1
ccamodes_max = 5
eofmodes = 3 #number of eof modes to compute
tinis = [1998, 1998, 1998,1982]
tends =  [2010, 2009, 2009, 2009]
force_download = True



all_mos = ['None', 'PCR', 'CCA', 'None']
all_predictands = ['PRCP', 'RFREQ', 'PRCP', 'RFREQ']
all_predictors = ['PRCP', 'PRCP', 'VQ', 'UQ']
all_obs = ['TRMM', 'CHIRPS', 'Chilestations', 'ENACTS-BD']
all_stations = [False, False, True, False]

all_obs_nonstation = ['ENACTS-BD', 'CPC-CMAP-URD', 'CHIRPS', 'TRMM', 'CPC','GPCC']
all_obs_stations = ['CHIRPS', 'Chilestations', 'GPCC']
all_models = ['COLA-RSMAS-CCSM4', 'GFDL-CM2p5-FLOR-A06', 'GFDL-CM2p5-FLOR-B01','GFDL-CM2p1-aer04','NASA-GEOSS2S','NCEP-CFSv2']
all_met = ['Pearson','Spearman','2AFC','RocAbove','RocBelow', 'RMSE', 'Ignorance', 'RPSS', 'GROC']
monf=['May', 'Sep'] 	# Initialization month
mons=[ 'May', 'Sep']
tgti=['1.5', '1.5']  #S: start for the DL
tgtf=['4.5', '3.5']   #S: end for the DL
tgts=['Jun-Sep', 'Oct-Dec']

fyr=2019 	# Forecast year
predictors = [ (10, -10, 90, 135), #indonesia predictors
				(0, -30, 10,40), #zambia predictor
				(30, -1, -120, -45),  #Chile predictors
				(35, 15, 80, 100)] #bangladesh predictor

predictands = [(5, -5, 93, 125), #indonesia predictand
				(-8, -18, 20,34), #zambia predictand
				(30, -1, -100, -65), #Chile Predictand
				(28, 20, 87, 94)] #bangladesh predictand

areas = ['Indonesia', 'Zambia', 'Chile', 'Bangladesh']




def run_test(pycpt, showPlot):
	pycpt.pltdomain()

	for model in range(len(pycpt.models)):
		for tgt in range(len(pycpt.mons)):
			pycpt.setupParams(tgt)
			pycpt.prepFiles(tgt, model)
			pycpt.CPTscript(tgt, model)
			pycpt.run(tgt, model)

	for ime in range(len(pycpt.met)):
		pycpt.pltmap(ime)

	for imod in range(pycpt.eof_modes):
		pycpt.plteofs(imod)

	models = pycpt.models[0:int(len(pycpt.models)/2)]
	pycpt.setNextGenModels(models)

	for tgt in range(len(pycpt.tgts)):
		pycpt.NGensemble(tgt)
		pycpt.CPTscript(tgt)
		pycpt.run(tgt)

	for ime in range(len(pycpt.met)):
		pycpt.pltmap(ime, isNextGen=1)

	pycpt.plt_ng_deterministic()
	pycpt.plt_ng_probabilistic()

	pycpt.ensemblefiles(['NextGen'], work)


for i in range(4):
	nla1, sla1, wlo1, elo1 = predictors[i]
	print(predictors[i], predictands[i])
	nla2, sla2, wlo2, elo2 = predictands[i]
	work = areas[i]
	models = all_models[2*i:2*i+2]
	print(work, models)
	MOS = all_mos[i]
	PREDICTAND = all_predictands[i]
	PREDICTOR = all_predictors[i]
	obs = all_obs[i]
	station = all_stations[i]
	met = all_met
	print(MOS, PREDICTAND, PREDICTOR)
	print(obs, station)
	print(met)
	tini = tinis[i]
	tend = tends[i]
	pycpt = PyCPT_Args(cptdir, models, met, obs, station, MOS, xmodes_min, xmodes_max, ymodes_min, ymodes_max, ccamodes_min, ccamodes_max, eofmodes, PREDICTAND, PREDICTOR, mons, tgti, tgtf, tgts, tini, tend, monf, fyr, force_download, nla1, sla1, wlo1, elo1, nla2, sla2, wlo2, elo2, shp_file, use_default, showPlot=False)
	setup_directories(work, workdir, force_download, cptdir, showPlot=False)

	run_test(pycpt, False)
	print("Error for run: \n {}  {}  {}  {} {} {}".format(work, MOS, models, obs, station, met ))
