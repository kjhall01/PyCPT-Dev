import os
import sys
import warnings
import platform
import struct
import xarray as xr
import numpy as np
import pandas as pd
import copy
from scipy.stats import t
from scipy.stats import invgamma
import cartopy.crs as ccrs
from cartopy import feature
import cartopy.mpl.ticker as cticker
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import fileinput
#import pycurl
import subprocess as sp
import numpy as np
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import netCDF4 as ns

warnings.filterwarnings("ignore")

#Supporting class for custom pyplot/matplotlib colormaps
class MidpointNormalize(colors.Normalize):
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# Ignoring masked values and all kinds of edge cases to make a
		# simple example..
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y))

#class for handling the absurd number of parameters for running PyCPT and passing them to various functions
class PyCPT_Args():
	def __init__(self,  working_directory, workdir, cptdir, models, met, obs, station, MOS, xmodes_min, xmodes_max, ymodes_min, ymodes_max, ccamodes_min, ccamodes_max, nmodes, PREDICTAND, PREDICTOR, mons, tgti, tgtf, tgts, tini, tend, monf, fyr, force_download, nla1, sla1, wlo1, elo1, nla2, sla2, wlo2, elo2, localobs, lonkey, latkey, timekey, datakey, shp_file, use_default):
		#These are the variables set by the user
		self.models = models
		self.working_directory = working_directory
		self.workdir = workdir
		self.shp_file = shp_file
		self.use_default = use_default
		self.met = met
		self.obs = obs
		self.cptdir = cptdir
		self.station = station
		self.MOS = MOS
		self.xmodes_min = xmodes_min
		self.xmodes_max = xmodes_max
		self.ymodes_min = ymodes_min
		self.ymodes_max = ymodes_max
		self.ccamodes_min = ccamodes_min
		self.ccamodes_max = ccamodes_max
		self.eof_modes = nmodes
		self.PREDICTAND = PREDICTAND
		self.PREDICTOR = PREDICTOR
		self.mons = mons
		self.tgti = tgti
		self.tgtf = tgtf
		self.tgts = tgts
		self.tini = tini
		self.tend = tend
		self.monf = monf
		self.fyr = fyr
		self.force_download = force_download
		self.nla1 = nla1
		self.sla1 = sla1
		self.wlo1 = wlo1
		self.elo1 = elo1
		self.nla2 = nla2
		self.sla2 = sla2
		self.wlo2 = wlo2
		self.elo2 = elo2
		self.localobs = localobs
		self.lonkey =  lonkey
		self.latkey = latkey
		self.timekey = timekey
		self.datakey = datakey
		self.L = ['1']
		if self.working_directory[-1] == '/' or self.working_directory[-1] == '\\':
			self.cwd_work = self.working_directory + self.workdir
		else:
			if '/' in self.working_directory:
				self.cwd_work = self.working_directory + '/' +  self.workdir
			else:
				self.cwd_work = self.working_directory + '\\' +  self.workdir



		#Following are to be updated during 'setup_params'
		self.ndays = 0
		self.nmonths = 0
		self.rainfall_frequency = 0
		self.wetday_threshold = 0
		self.threshold_pctle = False
		self.obs_source = ''
		self.hdate_last = 0
		self.mpref = ''
		self.ntrain = 0
		self.fprefix=''

		#a dictionary to use for dynamic formatting in the 'get data' function with the url dictionary
		self.arg_dict = {
			#set by the user, others depend on which tgt were looking at or which model or obs source
			'tini': self.tini,
			'tend': self.tend,
			'nla1': self.nla1,
			'sla1': self.sla1,
			'wlo1': self.wlo1,
			'elo1': self.elo1,
			'nla2': self.nla2,
			'sla2': self.sla2,
			'wlo2': self.wlo2,
			'elo2': self.elo2,
			'fyr': self.fyr
		}

		self.url_dict = {
		  'Hindcasts': {
		    'PRCP': {	'CanSIPSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.CanSIPSv2/.HINDCAST/.MONTHLY/.prec/SOURCES/.Models/.NMME/.CanSIPSv2/.FORECAST/.MONTHLY/.prec/appendstream/S/%280000%201%20{mon}%20{tini}-{tend}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'CMC1-CanCM3': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.CMC1-CanCM3/.HINDCAST/.MONTHLY/.prec/SOURCES/.Models/.NMME/.CMC1-CanCM3/.FORECAST/.MONTHLY/.prec/appendstream/S/%280000%201%20{mon}%20{tini}-{tend}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'CMC2-CanCM4': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.CMC2-CanCM4/.HINDCAST/.MONTHLY/.prec/SOURCES/.Models/.NMME/.CMC2-CanCM4/.FORECAST/.MONTHLY/.prec/appendstream/S/%280000%201%20{mon}%20{tini}-{tend}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'COLA-RSMAS-CCSM4': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.COLA-RSMAS-CCSM4/.MONTHLY/.prec/S/%280000%201%20{mon}%20{tini}-{tend}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'GFDL-CM2p5-FLOR-A06': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.GFDL-CM2p5-FLOR-A06/.MONTHLY/.prec/S/%280000%201%20{mon}%20{tini}-{tend}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'GFDL-CM2p5-FLOR-B01': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.GFDL-CM2p5-FLOR-B01/.MONTHLY/.prec/S/%280000%201%20{mon}%20{tini}-{tend}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'NASA-GEOSS2S': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NASA-GEOSS2S/.HINDCAST/.MONTHLY/.prec/S/%280000%201%20{mon}%20{tini}-{tend}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'NCEP-CFSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NCEP-CFSv2/.HINDCAST/.PENTAD_SAMPLES/.MONTHLY/.prec/SOURCES/.Models/.NMME/.NCEP-CFSv2/.FORECAST/.PENTAD_SAMPLES/.MONTHLY/.prec/appendstream/S/%280000%201%20{mon}%20{tini}-{tend}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/M/%281%29%2824%29RANGE/%5BM%5D/average/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv'},
		    'UQ': {'NCEP-CFSv2': 'http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.ENSEMBLE/.PGBF/.pressure_level/.VGRD/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.ENSEMBLE/.PGBF/.pressure_level/.SPFH/mul/P/850/VALUE/S/%2812%20{mon}%20{tini}-{tend}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv'},
		    'VQ': {'NCEP-CFSv2': 'http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.ENSEMBLE/.PGBF/.pressure_level/.VGRD/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.ENSEMBLE/.PGBF/.pressure_level/.SPFH/mul/P/850/VALUE/S/%281%20{mon}%201982-2009%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv'},
		    'RFREQ': {	'CanSIPSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.CanSIPSv2/.HINDCAST/.MONTHLY/.prec/S/%280000%201%20{mon}%201982-2009%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{ndays}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'CMC1-CanCM3': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.CMC1-CanCM3/.HINDCAST/.MONTHLY/.prec/S/%280000%201%20{mon}%201982-2009%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{ndays}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'CMC2-CanCM4': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.CMC2-CanCM4/.HINDCAST/.MONTHLY/.prec/S/%280000%201%20{mon}%201982-2009%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{ndays}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'COLA-RSMAS-CCSM4': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.COLA-RSMAS-CCSM4/.MONTHLY/.prec/S/%280000%201%20{mon}%201982-2009%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{ndays}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'GFDL-CM2p5-FLOR-A06': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.GFDL-CM2p5-FLOR-A06/.MONTHLY/.prec/S/%280000%201%20{mon}%201982-2009%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{ndays}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'GFDL-CM2p5-FLOR-B01': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.GFDL-CM2p5-FLOR-B01/.MONTHLY/.prec/S/%280000%201%20{mon}%201982-2009%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{ndays}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'NASA-GEOSS2S': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NASA-GEOSS2S/.HINDCAST/.MONTHLY/.prec/S/%280000%201%20{mon}%201982-2009%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{ndays}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'NCEP-CFSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NCEP-CFSv2/.HINDCAST/.MONTHLY/.prec/S/%280000%201%20{mon}%201982-2009%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{ndays}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'GFDL-CM2p1-aer04': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.GFDL-CM2p1-aer04/.MONTHLY/.prec/S/%280000%201%20{mon}%201982-2009%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{ndays}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv'}},
		  'Obs': {
		    'RFREQ': {
		        True: 'https://iridl.ldeo.columbia.edu/{obs_source}/Y/{sla2}/{nla2}/RANGE/X/{wlo2}/{elo2}/RANGE/T/(days%20since%201960-01-01)/streamgridunitconvert/T/(1%20Jan%201982)/(31%20Dec%202010)/RANGEEDGES/%5BT%5Dpercentileover/{wetday_threshold}/flagle/T/{ndays}/runningAverage/{ndays}/mul/T/2/index/.T/SAMPLE/nip/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/dup/I/5/splitstreamgrid/%5BI2%5Daverage/sub/I/3/-1/roll/.T/replaceGRID/-999/setmissing_value/grid%3A//name/(T)/def//units/(months%20since%201960-01-01)/def//standard_name/(time)/def//pointwidth/1/def/16/Jan/1901/ensotime/12./16/Jan/3001/ensotime/%3Agrid/use_as_grid//name/(fp)/def//units/(unitless)/def//long_name/(rainfall_freq)/def/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv.gz',
		        False:'http://datoteca.ole2.org/SOURCES/.UEA/.CRU/.TS4p0/.monthly/.wet/lon/%28X%29/renameGRID/lat/%28Y%29/renameGRID/time/%28T%29/renameGRID/T/%28Jan%201982%29/%28Dec%202010%29/RANGE/T/%28{tar}%29/seasonalAverage/Y/{sla2}/{nla2}/RANGEEDGES/X/{wlo2}/{elo2}/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv'},
		    'PRCP': {   'Chilestations': 'http://iridl.ldeo.columbia.edu/{obs_source}/T/%28{tar}%29/seasonalAverage/-999/setmissing_value/%5B%5D%5BT%5Dcptv10.tsv',
		                'ENACTS-BD':'https://datalibrary.bmd.gov.bd/{obs_source}/T/%28Jan%201982%29/%28Dec%202010%29/RANGE/T/%28{tar}%29/seasonalAverage/Y/%28{sla2}%29/%28{nla2}%29/RANGEEDGES/X/%28{wlo2}%29/%28{elo2}%29/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv',
		                'CPC-CMAP-URD': 'https://iridl.ldeo.columbia.edu/{obs_source}/T/%28Jan%201982%29/%28Dec%202010%29/RANGE/T/%28{tar}%29/seasonalAverage/Y/%28{sla2}%29/%28{nla2}%29/RANGEEDGES/X/%28{wlo2}%29/%28{elo2}%29/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv',
		                'TRMM': 'https://iridl.ldeo.columbia.edu/{obs_source}/T/%28Jan%201982%29/%28Dec%202010%29/RANGE/T/%28{tar}%29/seasonalAverage/Y/%28{sla2}%29/%28{nla2}%29/RANGEEDGES/X/%28{wlo2}%29/%28{elo2}%29/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv',
		                'CPC': 'https://iridl.ldeo.columbia.edu/{obs_source}/T/%28Jan%201982%29/%28Dec%202010%29/RANGE/T/%28{tar}%29/seasonalAverage/Y/%28{sla2}%29/%28{nla2}%29/RANGEEDGES/X/%28{wlo2}%29/%28{elo2}%29/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv',
		                'CHIRPS': 'https://iridl.ldeo.columbia.edu/{obs_source}/T/%28Jan%201982%29/%28Dec%202010%29/RANGE/T/%28{tar}%29/seasonalAverage/Y/%28{sla2}%29/%28{nla2}%29/RANGEEDGES/X/%28{wlo2}%29/%28{elo2}%29/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv',
		                'GPCC': 'https://iridl.ldeo.columbia.edu/{obs_source}/T/%28Jan%201982%29/%28Dec%202010%29/RANGE/T/%28{tar}%29/seasonalAverage/Y/%28{sla2}%29/%28{nla2}%29/RANGEEDGES/X/%28{wlo2}%29/%28{elo2}%29/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BT%5Dcptv10.tsv'
		    }
		  },
		  'Forecasts': {
		    'PRCP': {	'CanSIPSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.CanSIPSv2/.FORECAST/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'CMC1-CanCM3': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.CMC1-CanCM3/.FORECAST/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
					    'CMC2-CanCM4': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.CMC2-CanCM4/.FORECAST/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'COLA-RSMAS-CCSM4': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.COLA-RSMAS-CCSM4/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'GFDL-CM2p5-FLOR-A06': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.GFDL-CM2p5-FLOR-A06/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'GFDL-CM2p5-FLOR-B01': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.GFDL-CM2p5-FLOR-B01/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'NASA-GEOSS2S': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NASA-GEOSS2S/.FORECAST/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'NCEP-CFSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NCEP-CFSv2/.FORECAST/.EARLY_MONTH_SAMPLES/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'GFDL-CM2p1-aer04': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.GFDL-CM2p1-aer04/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/{nmonths30}/mul/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv'},
		    'UQ': {'NCEP-CFSv2': 'http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.REALTIME_ENSEMBLE/.PGBF/.pressure_level/.VGRD/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.REALTIME_ENSEMBLE/.PGBF/.pressure_level/.SPFH/mul/P/850/VALUE/S/%281%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv'},
		    'VQ': {'NCEP-CFSv2': 'http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.REALTIME_ENSEMBLE/.PGBF/.pressure_level/.VGRD/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.REALTIME_ENSEMBLE/.PGBF/.pressure_level/.SPFH/mul/P/850/VALUE/S/%281%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv'},
		    'RFREQ': {	'CMC1-CanCM3': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.CMC1-CanCM3/.FORECAST/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
					    'CMC2-CanCM4': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.CMC2-CanCM4/.FORECAST/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'COLA-RSMAS-CCSM4': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.COLA-RSMAS-CCSM4/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'GFDL-CM2p5-FLOR-A06': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.GFDL-CM2p5-FLOR-A06/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'GFDL-CM2p5-FLOR-B01': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.GFDL-CM2p5-FLOR-B01/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'NASA-GEOSS2S': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NASA-GEOSS2S/.HINDCAST/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'NCEP-CFSv2': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NCEP-CFSv2/.HINDCAST/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv',
						'GFDL-CM2p1-aer04': 'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.GFDL-CM2p1-aer04/.MONTHLY/.prec/S/%280000%201%20{monf}%20{fyr}%29/VALUES/L/{tgti}/{tgtf}/RANGEEDGES/%5BL%5D//keepgrids/average/%5BM%5D/average/Y/{sla1}/{nla1}/RANGEEDGES/X/{wlo1}/{elo1}/RANGEEDGES/-999/setmissing_value/%5BX/Y%5D%5BL/S/add%5D/cptv10.tsv'},
		  }
		}

	def setupParams(self,tar_ndx):
		"""PyCPT setup"""
		#global rainfall_frequency,threshold_pctle,wetday_threshold,obs_source,hdate_last,mpref,L,ntrain,fprefix, nmonths, ndays
		days_in_month_dict = {"Jan": 31, "Feb": 28, "Mar": 31, "Apr": 30, "May": 31, "Jun": 30, "Jul": 31, "Aug": 31, "Sep": 30, "Oct": 31, "Nov": 30, "Dec": 31}
		months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
		if '-' in self.tgts[tar_ndx]:
			mon_ini, mon_fin = self.tgts[tar_ndx].split('-')
			self.nmonths, self.ndays, flag = 0, 0, 0
			found_end, count = 0, 0
			while found_end == 0 and count < 24:
				if flag == 1:
					self.nmonths += 1
					self.ndays += days_in_month_dict[months[count % 12]]
					if months[count % 12] == mon_fin:
						flag, found_end = 0, 1

				if months[count % 12] == mon_ini:
					flag = 1
					self.nmonths += 1
					self.ndays += days_in_month_dict[months[count % 12]]
				count += 1

		else:
			mon_ini = self.tgts[tar_ndx]
			self.nmonths, self.ndays, self.flag = 0, 0, 0
			for i in months:
				if i == mon_ini:
					flag = 1
					self.nmonths += 1
					self.ndays += days_in_month_dict[i]

		# Predictor switches
		if self.PREDICTOR=='PRCP' or self.PREDICTOR=='UQ' or self.PREDICTOR=='VQ':
			self.rainfall_frequency = False  #False uses total rainfall for forecast period, True uses frequency of rainy days
			self.threshold_pctle = False
			self.wetday_threshold = -999 #WET day threshold (mm) --only used if rainfall_frequency is True!
		elif self.PREDICTOR=='RFREQ':
			self.rainfall_frequency = True  #False uses total rainfall for forecast period, True uses frequency of rainy days
			self.wetday_threshold = 3 #WET day threshold (mm) --only used if rainfall_frequency is True!
			self.threshold_pctle = False    #False for threshold in mm; Note that if True then if counts DRY days!!!

		if self.rainfall_frequency:
			print('Predictand is Rainfall Frequency; wet day threshold = '+str(self.wetday_threshold)+' mm')
		else:
			print('Predictand is Rainfall Total (mm)')

		########Observation dataset URLs
		if self.obs == 'CPC-CMAP-URD':
		    self.obs_source = 'SOURCES/.Models/.NMME/.CPC-CMAP-URD/prate'
		    self.hdate_last = 2010
		elif self.obs == 'TRMM':
		    self.obs_source = 'SOURCES/.NASA/.GES-DAAC/.TRMM_L3/.TRMM_3B42/.v7/.daily/.precipitation/X/-180./1.5/180./GRID/Y/-50/1.5/50/GRID'
		    self.hdate_last = 2014
		elif self.obs == 'CPC':
		    self.obs_source = 'SOURCES/.NOAA/.NCEP/.CPC/.UNIFIED_PRCP/.GAUGE_BASED/.GLOBAL/.v1p0/.extREALTIME/.rain/X/-180./1.5/180./GRID/Y/-90/1.5/90/GRID'
		    self.hdate_last = 2018
		elif self.obs == 'CHIRPS':
		    self.obs_source = 'SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p25/.prcp/'+str(self.ndays)+'/mul'
		    self.hdate_last = 2018
		elif self.obs == 'Chilestations':
		    self.obs_source = 'home/.xchourio/.ACToday/.CHL/.prcp'
		    self.hdate_last = 2019
		elif self.obs == 'GPCC':
		    self.obs_source = 'SOURCES/.WCRP/.GCOS/.GPCC/.FDP/.version7/.0p5/.prcp/'+str(self.nmonths)+'/mul'
		    self.hdate_last = 2013
		elif self.obs == 'Chilestations':
		    self.obs_source = 'home/.xchourio/.ACToday/.CHL/.prcp'
		    self.hdate_last = 2019
		elif self.obs == 'ENACTS-BD':
			self.obs_source = 'SOURCES/.Bangladesh/.BMD/.monthly/.rainfall/.rfe_merged/'+str(self.nmonths)+'/mul'
			self.hdate_last = 2020
		else:
		    print ("Obs option is invalid")

		########MOS-dependent parameters
		if self.MOS=='None':
		    self.mpref='noMOS'
		elif self.MOS=='CCA':
		    self.mpref='CCA'
		elif self.MOS=='PCR':
		    self.mpref='PCR'
		elif self.MOS=='ELR':
		    self.mpref='ELRho'

		self.L=['1'] #lead for file name (TO BE REMOVED --requested by Xandre)
		if self.tgts[tar_ndx] in ['Nov-Jan', 'Dec-Feb', 'Jan-Mar']:
			self.ntrain= self.tend-self.tini # length of training period
		else:
			self.ntrain= self.tend-self.tini + 1# length of training period

		self.fprefix = self.PREDICTOR

		self.arg_dict['mon'] = self.mons[tar_ndx]
		self.arg_dict['tar'] = self.tgts[tar_ndx]
		self.arg_dict['tgti'] = self.tgti[tar_ndx]
		self.arg_dict['tgtf'] = self.tgtf[tar_ndx]
		self.arg_dict['monf'] = self.monf[tar_ndx]
		self.arg_dict['ndays'] = self.ndays
		self.arg_dict['nmonths30'] = self.nmonths*30
		self.arg_dict['obs_source'] = self.obs_source
		self.file='FCST_xvPr'
		self.NGMOS = 'None'

	def prepFiles(self, tar_ndx, model_ndx):
		"""Function to download (or not) the needed files"""
		print('Preparing CPT files for '+self.models[model_ndx]+' and initialization '+self.mons[tar_ndx]+'...')
		self.getData( tar_ndx, model_ndx, 'Hindcasts')
		self.getData( tar_ndx, model_ndx, 'Obs')
		self.getData( tar_ndx, model_ndx, 'Forecasts')

	def cutOutLocalData(self, tar_ndx):

		with open(self.localobs[tar_ndx], 'r') as fp:
			counter = 0
			data = []
			years = 0
			flag = 0

			for line in fp:
				if counter == 0:
					header = line
				if counter == 1:
					nfields = line
				else:
					if line[0] == 'c' or line[0] == 'x':
						years += 1
						if flag==2:
							lats, lons = [], []
							data.append(np.asarray(year_data))
							year_data = []
							flag = 1
						else:
							lats, lons = [], []
							flag = 1
							year_data = []
					elif flag == 1:
						flag = 2
						lons = [float(num) for num in line.strip().split('\t')]
					else:
						line = [float(num) for num in line.strip().split('\t') ]
						lats.append(line.pop(0))
						year_data.append(np.asarray(line))
				counter += 1
			data.append(np.asarray(year_data))
			data = np.asarray(data)
			lats, lons = np.asarray(lats), np.asarray(lons)
			print(data.shape)
			print(lats.shape)
			print(lons.shape)




		var = data
		vari = 'prec'
		varname = vari
		units = 'mm'
		var[np.isnan(var)]=-999. #use CPT missing value
		tar = self.tgts[tar_ndx]
		L=0.5*(float(self.tgtf[tar_ndx])+float(self.tgti[tar_ndx]))
		monthdic = {'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06','Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}
		S=monthdic[self.mons[tar_ndx]]
		if '-' in tar:
			mi=monthdic[tar.split("-")[0]]
			mf=monthdic[tar.split("-")[1]]

			#Read grads file to get needed coordinate arrays
			#W, Wi, XD, H, Hi, YD, T, Ti, TD = self.readGrADSctl(self.models,self.fprefix,self.PREDICTAND,self.mpref,self.file,tar,self.monf[tar_ndx],self.fyr)

			if tar=='Dec-Feb' or tar=='Nov-Jan':  #double check years are sync
				xyear=True  #flag a cross-year season
			else:
				#Ti=Ti+1
				xyear=False
		else:
			mi=monthdic[tar]
			mf=monthdic[tar]

			#Read grads file to get needed coordinate arrays
			#W, Wi, XD, H, Hi, YD, T, Ti, TD = self.readGrADSctl(self.models,self.fprefix,self.PREDICTAND,self.mpref,self.file,tar,self.monf[tar_ndx],self.fyr)

			if tar=='Dec-Feb' or tar=='Nov-Jan':  #double check years are sync
				xyear=True  #flag a cross-year season
			else:
				#Ti=Ti+1
				xyear=False

		latkeys = np.where(lats <= self.nla2  )
		latkeys = np.asarray(np.where(lats[latkeys] >= self.sla2))#.reshape(-1,1)

		lonkeys = np.asarray(np.where(lons <= self.elo2 ))
		lonkeys = np.asarray(np.where(lons[lonkeys] >= self.wlo2))#.reshape(-1,1)

		data = data[:,latkeys[0,0]:latkeys[0,latkeys.shape[1]-1], lonkeys[0,0]:lonkeys[0,lonkeys.shape[1]-1]]
		lats = lats[latkeys]
		lons = lons[lonkeys]

		Ti = self.tini
		T = data.shape[0]
		Tarr = np.arange(Ti, T+Ti)

		Wi = lons[0,0]
		W = lons.shape[0]
		XD = lons[0,1] - lons[0,0]

		Hi = lats[0,0]
		H = lats.shape[0]
		YD = lats[0,1] - lats[0,0]

		Xarr = np.linspace(Wi, Wi+W*XD,num=W+1)
		Yarr = np.linspace(Hi+H*YD, Hi,num=H+1)
		vari = 'prcp'

		Tarr = np.arange(Ti, Ti+T)
		Xarr = np.linspace(Wi, Wi+W*XD,num=W+1)
		Yarr = np.linspace(Hi+H*YD, Hi,num=H+1)
		outfile = './input/{}_{}_{}.tsv'.format('obs', self.PREDICTAND, self.tgts[tar_ndx])
		#Now write the CPT file
		f = open(outfile, 'w')
		f.write("xmlns:cpt=http://iri.columbia.edu/CPT/v10/{}".format(os.linesep))
		#f.write("xmlns:cf=http://cf-pcmdi.llnl.gov/documents/cf-conventions/1.4/\n")   #not really needed
		f.write("cpt:nfields=1{}".format(os.linesep))
		#f.write("cpt:T	" + str(Tarr)+"\n")  #not really needed
		for it in range(T):
			if xyear==True:
				f.write("cpt:field="+vari+", cpt:L="+str(L)+" months, cpt:S="+str(Ti)+"-"+S+"-01T00:00, cpt:T="+str(Tarr[it])+"-"+mi+"/"+str(Tarr[it]+1)+"-"+mf+", cpt:nrow="+str(H)+", cpt:ncol="+str(W)+", cpt:row=Y, cpt:col=X, cpt:units="+units+", cpt:missing=-999.{}".format(os.linesep))
			else:
				f.write("cpt:field="+vari+", cpt:L="+str(L)+" months, cpt:S="+str(Ti)+"-"+S+"-01T00:00, cpt:T="+str(Tarr[it])+"-"+mi+"/"+mf+", cpt:nrow="+str(H)+", cpt:ncol="+str(W)+", cpt:row=Y, cpt:col=X, cpt:units="+units+", cpt:missing=-999.{}".format(os.linesep))
			#f.write("\t")
			np.savetxt(f, Xarr[0:-1], fmt="%.3f",newline='\t') #f.write(str(Xarr)[1:-1])
			f.write("{}".format(os.linesep)) #next line
			for iy in range(H):
				#f.write(str(Yarr[iy]) + "\t" + str(var[it,iy,0:-1])[1:-1]) + "\n")
				np.savetxt(f,np.r_[Yarr[iy+1],var[it,iy,0:]],fmt="%.3f", newline='\t')  #excise extra line
				f.write("{}".format(os.linesep)) #next line
		f.close()

	def convertNCDF_CPT(self, tar_ndx):
		data = ns.Dataset(self.localobs[tar_ndx], 'r') #open .nc file for reading
		units='mm'
		L=0.5*(float(self.tgtf[tar_ndx])+float(self.tgti[tar_ndx]))

		Ti = self.tini
		T = data[self.timekey][:].filled().shape[0]
		Tarr = np.arange(Ti, T+Ti)

		Wi = data[self.lonkey][:].filled()[0]
		W = data[self.lonkey][:].filled().shape[0]
		XD = data[self.lonkey][:].filled()[1] - data[self.lonkey][:].filled()[0]

		Hi = data[self.latkey][:].filled()[0]
		H = data[self.latkey][:].filled().shape[0]
		YD = data[self.latkey][:].filled()[1] - data[self.latkey][:].filled()[0]

		Xarr = np.linspace(Wi, Wi+W*XD,num=W+1)
		Yarr = np.linspace(Hi+H*YD, Hi,num=H+1)
		vari = self.datakey

		monthdic = {'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06','Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}
		S=monthdic[self.mons[tar_ndx]]
		if '-' in self.tgts[tar_ndx]:
			mi=monthdic[self.tgts[tar_ndx].split("-")[0]]
			mf=monthdic[self.tgts[tar_ndx].split("-")[1]]

			if self.tgts[tar_ndx]=='Dec-Feb' or self.tgts[tar_ndx]=='Nov-Jan':  #double check years are sync
				xyear=True  #flag a cross-year season
			else:
				#Ti=Ti+1
				xyear=False
		else:
			mi=monthdic[self.tgts[tar_ndx]]
			mf=monthdic[self.tgts[tar_ndx]]

			if self.tgts[tar_ndx]=='Dec-Feb' or self.tgts[tar_ndx]=='Nov-Jan':  #double check years are sync
				xyear=True  #flag a cross-year season
			else:
				#Ti=Ti+1
				xyear=False

		var = data[self.datakey][:].filled()
		outfile = './input/{}_{}_{}.tsv'.format('obs', self.PREDICTAND, self.tgts[tar_ndx])
		#Now write the CPT file
		f = open(outfile, 'w')
		f.write("xmlns:cpt=http://iri.columbia.edu/CPT/v10/{}".format(os.linesep))
		#f.write("xmlns:cf=http://cf-pcmdi.llnl.gov/documents/cf-conventions/1.4/{}".format(os.linesep))   #not really needed
		f.write("cpt:nfields=1{}".format(os.linesep))
		#f.write("cpt:T	" + str(Tarr)+"{}".format(os.linesep))  #not really needed
		for it in range(T):
			if xyear==True:
				f.write("cpt:field="+vari+", cpt:L="+str(L)+" months, cpt:S="+str(Ti)+"-"+S+"-01T00:00, cpt:T="+str(Tarr[it])+"-"+mi+"/"+str(Tarr[it]+1)+"-"+mf+", cpt:nrow="+str(H)+", cpt:ncol="+str(W)+", cpt:row=Y, cpt:col=X, cpt:units="+units+", cpt:missing=-999.{}".format(os.linesep))
			else:
				f.write("cpt:field="+vari+", cpt:L="+str(L)+" months, cpt:S="+str(Ti)+"-"+S+"-01T00:00, cpt:T="+str(Tarr[it])+"-"+mi+"/"+mf+", cpt:nrow="+str(H)+", cpt:ncol="+str(W)+", cpt:row=Y, cpt:col=X, cpt:units="+units+", cpt:missing=-999.{}".format(os.linesep))
			#f.write("\t")
			np.savetxt(f, Xarr[0:-1], fmt="%.3f",newline='\t') #f.write(str(Xarr)[1:-1])
			f.write("{}".format(os.linesep)) #next line
			for iy in range(H):
				#f.write(str(Yarr[iy]) + "\t" + str(var[it,iy,0:-1])[1:-1]) + "{}".format(os.linesep))
				np.savetxt(f,np.r_[Yarr[iy+1],var[it,iy,0:]],fmt="%.3f", newline='\t')  #excise extra line
				f.write("{}".format(os.linesep)) #next line
		f.close()

	def getData(self,  tar_ndx, model_ndx, datatype):
		"""tar_ndx = index of the target period we are looking at within the list
		   model_ndx = index of model were looking at within the list
		   datatype = one of 'Obs', 'Hindcasts', or 'Forecasts'    """

		#set the model to the current focus
		self.arg_dict['model'] = self.models[model_ndx],
		found=0
		if datatype=='Obs' and False: #we are not doing local data at this time
			if os.path.isfile(self.localobs[tar_ndx]):
				print('Found Local data - trying to open as netCDF... ')
				try:
					self.convertNCDF_CPT(tar_ndx)
					print('Success! Converted to CPT for you')
					found = 1
				except:
					print('Could not open as NetCDF - We are trusting that it is correct CPT format')
					self.cutOutLocalData(tar_ndx)
					found = 1
			else:
				print("No local data by that name,, looking for previously downloaded")
				found = 0

		if datatype != 'Obs' or found == 0:
			if not self.force_download:
				try:
					if datatype == 'Hindcasts':
						ff=open("./input/"+self.models[model_ndx]+"_{}_".format(self.fprefix)+self.tgts[tar_ndx]+"_ini"+self.mons[tar_ndx]+".tsv",  'r')
						s = ff.readline()
					elif datatype == "Obs":
						if self.fprefix == 'RFREQ':
							fpre = 'RFREQ'
						else:
							fpre = 'PRCP'
						ff = open("./input/obs_"+fpre+"_"+self.tgts[tar_ndx]+".tsv", 'r')
						s = ff.readline()
					else:
						ff=open("./input/"+self.models[model_ndx]+"fcst_PRCP_"+self.tgts[tar_ndx]+"_ini"+self.monf[tar_ndx]+str(self.fyr)+".tsv")
						s = ff.readline()
				except OSError as err:
					print("\033[1mWarning:\033[0;0m {0}".format(err))
					print("{} precip file doesn't exist --\033[1mSOLVING: downloading file\033[0;0m".format(datatype))
					self.force_download = True

			if self.force_download:
				if datatype == 'Obs':
					url = self.url_dict[datatype][self.fprefix][self.threshold_pctle if self.fprefix == 'RFREQ' else self.obs ]
				else:
					url = self.url_dict[datatype][self.fprefix][self.models[model_ndx]]
				print("\n Obs (Freq) data URL: \n\n "+url.format(**self.arg_dict))
				if datatype == 'Hindcasts':
					get_ipython().system("curl -k "+url.format(**self.arg_dict)+" > ./input/"+self.models[model_ndx]+"_{}_".format(self.fprefix)+self.tgts[tar_ndx]+"_ini"+self.mons[tar_ndx]+".tsv")
				elif datatype == 'Obs':
					if self.fprefix == 'RFREQ':
						fpre = 'RFREQ'
					else:
						fpre = 'PRCP'
					print("curl -k "+url.format(**self.arg_dict)+" > ./input/obs_"+fpre+"_"+self.tgts[tar_ndx]+".tsv")
					get_ipython().system("curl -k "+url.format(**self.arg_dict)+" > ./input/obs_"+fpre+"_"+self.tgts[tar_ndx]+".tsv")
				else:
					get_ipython().system("curl -k "+url.format(**self.arg_dict)+" > ./input/"+self.models[model_ndx]+"fcst_{}_".format(self.fprefix)+self.tgts[tar_ndx]+"_ini"+self.monf[tar_ndx]+str(self.fyr)+".tsv")

		if self.obs_source=='home/.xchourio/.ACToday/.CHL/.prcp':   #weirdly enough, Ingrid sends the file with nfields=0. This is my solution for now. AGM
			replaceAll("obs_"+predictand+"_"+tar+".tsv","cpt:nfields=0","cpt:nfields=1")

		print('{} file ready to go'.format(datatype))
		print('----------------------------------------------')

	def CPTscript(self, tar_ndx, model_ndx=-1):
		"""Function to write CPT namelist file
		"""
		flag=0
		if model_ndx == -1:
			self._tempmods = copy.deepcopy(self.models)
			self.models = ['NextGen']
			self._tempmos = copy.deepcopy(self.MOS)
			self.MOS = self.NGMOS
			model_ndx=0
			flag=1

		# Set up CPT parameter file
		f=open("./scripts/params","w")
		if self.MOS=='CCA':
			# Opens CCA
			f.write("611{}".format(os.linesep))
		elif self.MOS=='PCR':
			# Opens PCR
			f.write("612{}".format(os.linesep))
		elif self.MOS=='PCR':
			# Opens GCM; because the calibration takes place via sklearn.linear_model (in the Jupyter notebook)
			f.write("614{}".format(os.linesep))
		elif self.MOS=='None':
			# Opens GCM (no calibration performed in CPT)
			f.write("614{}".format(os.linesep))
		else:
			print ("MOS option is invalid")

		# First, ask CPT to stop if error is encountered
		f.write("571{}".format(os.linesep))
		f.write("3{}".format(os.linesep))

		# Opens X input file
		f.write("1{}".format(os.linesep))
		file='{}/input/'.format(self.cwd_work)+self.models[model_ndx]+'_'+self.fprefix+'_'+self.tgts[tar_ndx]+'_ini'+self.mons[tar_ndx]+'.tsv{}'.format(os.linesep)
		f.write(os.path.normpath(file))
		# Nothernmost latitude
		f.write(str(self.nla1)+'{}'.format(os.linesep))
		# Southernmost latitude
		f.write(str(self.sla1)+'{}'.format(os.linesep))
		# Westernmost longitude
		f.write(str(self.wlo1)+'{}'.format(os.linesep))
		# Easternmost longitude
		f.write(str(self.elo1)+'{}'.format(os.linesep))

		if self.MOS=='CCA' or self.MOS=='PCR':
			# Minimum number of X modes
			f.write("{}{}".format(self.xmodes_min, os.linesep))
			# Maximum number of X modes
			f.write("{}{}".format(self.xmodes_max, os.linesep))

			# Opens forecast (X) file
			f.write("3{}".format(os.linesep))
			file='{}/input/'.format(self.cwd_work)+self.models[model_ndx]+'fcst_'+self.fprefix+'_'+self.tgts[tar_ndx]+'_ini'+self.monf[tar_ndx]+str(self.fyr)+'.tsv{}'.format(os.linesep)
			f.write(os.path.normpath(file))
			#Start forecast:
			f.write("223{}".format(os.linesep))
			if self.monf[tar_ndx]=="Dec":
				f.write(str(self.fyr+1)+"{}".format(os.linesep))
			else:
				f.write(str(self.fyr)+"{}".format(os.linesep))

		# Opens Y input file
		f.write("2{}".format(os.linesep))
		file='{}/input/obs_'.format(self.cwd_work)+self.PREDICTAND+'_'+self.tgts[tar_ndx]+'.tsv{}'.format(os.linesep)
		f.write(os.path.normpath(file))
		if self.station==False:
			# Nothernmost latitude
			f.write(str(self.nla2)+'{}'.format(os.linesep))
			# Southernmost latitude
			f.write(str(self.sla2)+'{}'.format(os.linesep))
			# Westernmost longitude
			f.write(str(self.wlo2)+'{}'.format(os.linesep))
			# Easternmost longitude
			f.write(str(self.elo2)+'{}'.format(os.linesep))
		if self.MOS=='CCA':
			# Minimum number of Y modes
			f.write("{}{}".format(self.ymodes_min, os.linesep))
			# Maximum number of Y modes
			f.write("{}{}".format(self.ymodes_max, os.linesep))

			# Minimum number of CCA modes
			f.write("{}{}".format(self.ccamodes_min, os.linesep))
			# Maximum number of CCAmodes
			f.write("{}{}".format(self.ccamodes_max, os.linesep))

		# X training period
		f.write("4{}".format(os.linesep))
		# First year of X training period
		if self.monf[tar_ndx] in ['Dec', 'Nov']:
			f.write("{}{}".format(self.tini+1, os.linesep))
		else:
			f.write("{}{}".format(self.tini, os.linesep))
		# Y training period
		f.write("5{}".format(os.linesep))
		# First year of Y training period
		if self.monf[tar_ndx] in ['Dec', 'Nov']:
			f.write("{}{}".format(self.tini+1, os.linesep))
		else:
			f.write("{}{}".format(self.tini, os.linesep))


		# Goodness index
		f.write("531{}".format(os.linesep))
		# Kendall's tau
		f.write("3{}".format(os.linesep))

		# Option: Length of training period
		f.write("7{}".format(os.linesep))
		# Length of training period
		f.write(str(self.ntrain)+'{}'.format(os.linesep))
		#	%store 55 >> params
		# Option: Length of cross-validation window
		f.write("8{}".format(os.linesep))
		# Enter length
		f.write("3{}".format(os.linesep))

		if self.MOS!="None":
			# Turn ON transform predictand data
			f.write("541{}".format(os.linesep))

		if self.fprefix=='RFREQ':
			# Turn ON zero bound for Y data	 (automatically on by CPT if variable is precip)
			f.write("542{}".format(os.linesep))
		# Turn ON synchronous predictors
		f.write("545{}".format(os.linesep))
		# Turn ON p-values for masking maps
		#f.write("561{}".format(os.linesep))

		### Missing value options
		f.write("544{}".format(os.linesep))
		# Missing value X flag:
		blurb='-999{}'.format(os.linesep)
		f.write(blurb)
		# Maximum % of missing values
		f.write("10{}".format(os.linesep))
		# Maximum % of missing gridpoints
		f.write("10{}".format(os.linesep))
		# Number of near-neighbors
		f.write("1{}".format(os.linesep))
		# Missing value replacement : best-near-neighbors
		f.write("4{}".format(os.linesep))
		# Y missing value flag
		blurb='-999{}'.format(os.linesep)
		f.write(blurb)
		# Maximum % of missing values
		f.write("10{}".format(os.linesep))
		# Maximum % of missing stations
		f.write("10{}".format(os.linesep))
		# Number of near-neighbors
		f.write("1{}".format(os.linesep))
		# Best near neighbor
		f.write("4{}".format(os.linesep))

		# Transformation settings
		#f.write("554{}".format(os.linesep))
		# Empirical distribution
		#f.write("1\n")

		#######BUILD MODEL AND VALIDATE IT	!!!!!

		# NB: Default output format is GrADS format
		# select output format
		f.write("131{}".format(os.linesep))
		# GrADS format
		f.write("3{}".format(os.linesep))

		# save goodness index
		f.write("112{}".format(os.linesep))
		file='{}/output/'.format(self.cwd_work)+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_Kendallstau_'+self.tgts[tar_ndx]+'_'+self.mons[tar_ndx]+'{}'.format(os.linesep)
		f.write(os.path.normpath(file))

		# Build cross-validated model
		f.write("311{}".format(os.linesep))

		# save EOFs
		if self.MOS=='CCA' or self.MOS=='PCR' :    #kjch092120
			f.write("111{}".format(os.linesep))
			#X EOF
			f.write("302{}".format(os.linesep))
			file= '{}/output/'.format(self.cwd_work)+self.models[model_ndx] +'_'+ self.fprefix+self.PREDICTAND+'_'+self.mpref+'_EOFX_'+self.tgts[tar_ndx]+'_'+self.mons[tar_ndx]+'{}'.format(os.linesep)
			f.write(os.path.normpath(file))
			#Exit submenu
			f.write("0{}".format(os.linesep))
		if self.MOS=='CCA' :        #kjch092120
			f.write("111{}".format(os.linesep))
			#Y EOF
			f.write("312{}".format(os.linesep))
			file='{}/output/'.format(self.cwd_work)+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_EOFY_'+self.tgts[tar_ndx]+'_'+self.mons[tar_ndx]+'{}'.format(os.linesep)
			f.write(os.path.normpath(file))
			#Exit submenu
			f.write("0{}".format(os.linesep))

		# cross-validated skill maps
		f.write("413{}".format(os.linesep))
		# save Pearson's Correlation
		f.write("1{}".format(os.linesep))
		file='{}/output/'.format(self.cwd_work)+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_Pearson_'+self.tgts[tar_ndx]+'_'+self.mons[tar_ndx]+'{}'.format(os.linesep)
		f.write(os.path.normpath(file))

		# cross-validated skill maps
		f.write("413{}".format(os.linesep))
		# save Spearmans Correlation
		f.write("2{}".format(os.linesep))
		file='{}/output/'.format(self.cwd_work)+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_Spearman_'+self.tgts[tar_ndx]+'_'+self.mons[tar_ndx]+'{}'.format(os.linesep)
		f.write(os.path.normpath(file))

		# cross-validated skill maps
		f.write("413{}".format(os.linesep))
		# save 2AFC score
		f.write("3{}".format(os.linesep))
		file='{}/output/'.format(self.cwd_work)+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_2AFC_'+self.tgts[tar_ndx]+'_'+self.mons[tar_ndx]+'{}'.format(os.linesep)
		f.write(os.path.normpath(file))

		# cross-validated skill maps
		f.write("413{}".format(os.linesep))
		# save RocBelow score
		f.write("15{}".format(os.linesep))
		file='{}/output/'.format(self.cwd_work)+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_RocBelow_'+self.tgts[tar_ndx]+'_'+self.mons[tar_ndx]+'{}'.format(os.linesep)
		f.write(os.path.normpath(file))

		# cross-validated skill maps
		f.write("413{}".format(os.linesep))
		# save RocAbove score
		f.write("16{}".format(os.linesep))
		file='{}/output/'.format(self.cwd_work)+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_RocAbove_'+self.tgts[tar_ndx]+'_'+self.mons[tar_ndx]+'{}'.format(os.linesep)
		f.write(os.path.normpath(file))

		# cross-validated skill maps
		f.write("413{}".format(os.linesep))
		# save RocAbove score
		f.write("7{}".format(os.linesep))
		file='{}/output/'.format(self.cwd_work)+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_RMSE_'+self.tgts[tar_ndx]+'_'+self.mons[tar_ndx]+'{}'.format(os.linesep)
		f.write(os.path.normpath(file))



		if self.MOS=='CCA' or self.MOS=='PCR' or self.MOS=="None" :  #kjch092120 #DO NOT USE CPT to compute probabilities if MOS='None' --use IRIDL for direct counting
			#######FORECAST(S)	!!!!!
			# Probabilistic (3 categories) maps
			f.write("455{}".format(os.linesep))
			# Output results
			f.write("111{}".format(os.linesep))
			# Forecast probabilities
			f.write("501{}".format(os.linesep))
			file='{}/output/'.format(self.cwd_work)+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'FCST_P_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'{}'.format(os.linesep)
			f.write(os.path.normpath(file))
			#502 # Forecast odds
			#Exit submenu
			f.write("0{}".format(os.linesep))

			# Compute deterministc values and prediction limits
			f.write("454{}".format(os.linesep))
			# Output results
			f.write("111{}".format(os.linesep))
			# Forecast values
			f.write("511{}".format(os.linesep))
			file='{}/output/'.format(self.cwd_work)+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'FCST_V_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'{}'.format(os.linesep)
			f.write(os.path.normpath(file))
			#502 # Forecast odds


			#######Following files are used to plot the flexible format
			# Save cross-validated predictions
			f.write("201{}".format(os.linesep))
			file='{}/output/'.format(self.cwd_work)+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'FCST_xvPr_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'{}'.format(os.linesep)
			f.write(os.path.normpath(file))
			# Save deterministic forecasts [mu for Gaussian fcst pdf]
			f.write("511{}".format(os.linesep))
			file='{}/output/'.format(self.cwd_work)+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'FCST_mu_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'{}'.format(os.linesep)
			f.write(os.path.normpath(file))
			# Save prediction error variance [sigma^2 for Gaussian fcst pdf]
			f.write("514{}".format(os.linesep))
			file='{}/output/'.format(self.cwd_work)+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'FCST_var_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'{}'.format(os.linesep)
			f.write(os.path.normpath(file))
			# Save z
			f.write("532{}".format(os.linesep))
			file='{}/output/'.format(self.cwd_work)+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'FCST_z_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'{}'.format(os.linesep)
			f.write(os.path.normpath(file))
			# Save predictand [to build predictand pdf]
			f.write("102{}".format(os.linesep))
			file='{}/output/'.format(self.cwd_work)+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'FCST_Obs_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'{}'.format(os.linesep)
			f.write(os.path.normpath(file))

			#Exit submenu
			f.write("0{}".format(os.linesep))

			# Change to ASCII format to send files to DL
			f.write("131{}".format(os.linesep))
			# ASCII format
			f.write("2{}".format(os.linesep))
			# Output results
			f.write("111{}".format(os.linesep))
			# Save cross-validated predictions
			f.write("201{}".format(os.linesep))
			file='{}/output/'.format(self.cwd_work)+self.models[model_ndx]+'_'+self.fprefix+'_'+self.mpref+'FCST_xvPr_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'{}'.format(os.linesep)
			f.write(os.path.normpath(file))
			# Save deterministic forecasts [mu for Gaussian fcst pdf]
			f.write("511{}".format(os.linesep))
			file='{}/output/'.format(self.cwd_work)+self.models[model_ndx]+'_'+self.fprefix+'_'+self.mpref+'FCST_mu_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'{}'.format(os.linesep)
			f.write(os.path.normpath(file))
			# Forecast probabilities
			f.write("501{}".format(os.linesep))
			file='{}/output/'.format(self.cwd_work)+self.models[model_ndx]+'_'+self.fprefix+'_'+self.mpref+'FCST_P_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'{}'.format(os.linesep)
			f.write(os.path.normpath(file))
			# Save prediction error variance [sigma^2 for Gaussian fcst pdf]
			f.write("514{}".format(os.linesep))
			file='{}/output/'.format(self.cwd_work)+self.models[model_ndx]+'_'+self.fprefix+'_'+self.mpref+'FCST_var_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'{}'.format(os.linesep)
			f.write(os.path.normpath(file))
			# Save z
			f.write("532{}".format(os.linesep))
			file='{}/output/'.format(self.cwd_work)+self.models[model_ndx]+'_'+self.fprefix+'_'+self.mpref+'FCST_z_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'{}'.format(os.linesep)
			f.write(os.path.normpath(file))
			# Save predictand [to build predictand pdf]
			f.write("102{}".format(os.linesep))
			file='{}/output/'.format(self.cwd_work)+self.models[model_ndx]+'_'+self.fprefix+'_'+self.mpref+'FCST_Obs_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'{}'.format(os.linesep)
			f.write(os.path.normpath(file))

			# cross-validated skill maps
			if self.MOS=="PCR" or self.MOS=="CCA" : #kjch092120
				f.write("0{}".format(os.linesep))

			# cross-validated skill maps
			f.write("413{}".format(os.linesep))
			# save 2AFC score
			f.write("3{}".format(os.linesep))
			file='{}/output/'.format(self.cwd_work)+self.models[model_ndx]+'_'+self.fprefix+'_'+self.mpref+'_2AFC_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'{}'.format(os.linesep)
			f.write(os.path.normpath(file))
			# Stop saving  (not needed in newest version of CPT)

		###########PFV --Added by AGM in version 1.5
		#Compute and write retrospective forecasts for prob skill assessment.
		#Re-define forecas file if PCR or CCA
		if self.MOS=="PCR" or self.MOS=="CCA" : #kjch092120
			f.write("3{}".format(os.linesep))
			file='./input/'+self.models[model_ndx]+'_'+self.fprefix+'_'+self.tgts[tar_ndx]+'_ini'+self.mons[tar_ndx]+'.tsv{}'.format(os.linesep)  #here a conditional should choose if rainfall freq is being used
			f.write(os.path.normpath(file))
		#Forecast period settings
		f.write("6{}".format(os.linesep))
		# First year to forecast. Save ALL forecasts (for "retroactive" we should only assess second half)
		if self.monf[tar_ndx]=="Oct" or self.monf[tar_ndx]=="Nov" or self.monf[tar_ndx]=="Dec":
			f.write(str(self.tini+1)+'{}'.format(os.linesep))
		else:
			f.write(str(self.tini)+'{}'.format(os.linesep))
		#Number of forecasts option
		f.write("9{}".format(os.linesep))
		# Number of reforecasts to produce
		if self.monf[tar_ndx]=="Oct" or self.monf[tar_ndx]=="Nov" or self.monf[tar_ndx]=="Dec":
			f.write(str(self.ntrain-1)+'{}'.format(os.linesep))
		else:
			f.write(str(self.ntrain)+'{}'.format(os.linesep))
		# Change to ASCII format
		f.write("131{}".format(os.linesep))
		# ASCII format
		f.write("2{}".format(os.linesep))
		# Probabilistic (3 categories) maps
		f.write("455{}".format(os.linesep))
		# Output results
		f.write("111{}".format(os.linesep))
		# Forecast probabilities --Note change in name for reforecasts:
		f.write("501{}".format(os.linesep))
		file='{}/output/'.format(self.cwd_work)+self.models[model_ndx]+'_RFCST_'+self.fprefix+'_'+self.tgts[tar_ndx]+'_ini'+self.monf[tar_ndx]+str(self.fyr)+'{}'.format(os.linesep)
		f.write(os.path.normpath(file))
		#502 # Forecast odds
		#Exit submenu
		f.write("0{}".format(os.linesep))

		# Close X file so we can access the PFV option
		f.write("121{}".format(os.linesep))
		f.write("Y{}".format(os.linesep))  #Yes to cleaning current results:# WARNING:
		#Select Probabilistic Forecast Verification (PFV)
		f.write("621{}".format(os.linesep))
		# Opens X input file
		f.write("1{}".format(os.linesep))
		file='{}/output/'.format(self.cwd_work)+self.models[model_ndx]+'_RFCST_'+self.fprefix+'_'+self.tgts[tar_ndx]+'_ini'+self.monf[tar_ndx]+str(self.fyr)+'.txt{}'.format(os.linesep)
		f.write(os.path.normpath(file))
		# Nothernmost latitude
		f.write(str(self.nla2)+'{}'.format(os.linesep))
		# Southernmost latitude
		f.write(str(self.sla2)+'{}'.format(os.linesep))
		# Westernmost longitude
		f.write(str(self.wlo2)+'{}'.format(os.linesep))
		# Easternmost longitude
		f.write(str(self.elo2)+'{}'.format(os.linesep))

		f.write("5{}".format(os.linesep))
		# First year of the PFV
		# for "retroactive" only first half of the entire training period is typically used --be wise, as sample is short)
		if self.monf[tar_ndx]=="Oct" or self.monf[tar_ndx]=="Nov" or self.monf[tar_ndx]=="Dec":
			f.write(str(self.tini+1)+'{}'.format(os.linesep))
		else:
			f.write(str(self.tini)+'{}'.format(os.linesep))

		#If these prob forecasts come from a cross-validated prediction (as it's coded right now)
		#we don't want to cross-validate those again (it'll change, for example, the xv error variances)
		#Forecast Settings menu
		f.write("552{}".format(os.linesep))
		#Conf level at 50% to have even, dychotomous intervals for reliability assessment (as per Simon suggestion)
		f.write("50{}".format(os.linesep))
		#Fitted error variance option  --this is the key option: 3 is 0-leave-out cross-validation, so no cross-validation!
		f.write("3{}".format(os.linesep))
		#-----Next options are required but not really used here:
		#Ensemble size
		f.write("10{}".format(os.linesep))
		#Odds relative to climo?
		f.write("N{}".format(os.linesep))
		#Exceedance probabilities: show as non-exceedance?
		f.write("N{}".format(os.linesep))
		#Precision options:
		#Number of decimal places (Max 8):
		f.write("3{}".format(os.linesep))
		#Forecast probability rounding:
		f.write("1{}".format(os.linesep))
		#End of required but not really used options ----

		#Verify
		f.write("313{}".format(os.linesep))

		#Reliability diagram
		f.write("431{}".format(os.linesep))
		f.write("Y{}".format(os.linesep)) #yes, save results to a file
		file='{}/output/'.format(self.cwd_work)+self.models[model_ndx]+'_RFCST_reliabdiag_'+self.fprefix+'_'+self.tgts[tar_ndx]+'_ini'+self.monf[tar_ndx]+str(self.fyr)+'.tsv{}'.format(os.linesep)
		f.write(os.path.normpath(file))

		# select output format -- GrADS, so we can plot it in Python
		f.write("131{}".format(os.linesep))
		# GrADS format
		f.write("3{}".format(os.linesep))

		# Probabilistic skill maps
		f.write("437{}".format(os.linesep))
		# save Ignorance (all cats)
		f.write("101{}".format(os.linesep))
		file='{}/output/'.format(self.cwd_work)+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_Ignorance_'+self.tgts[tar_ndx]+'_'+self.mons[tar_ndx]+'{}'.format(os.linesep)
		f.write(os.path.normpath(file))

		# Probabilistic skill maps
		f.write("437{}".format(os.linesep))
		# save Ranked Probability Skill Score (all cats)
		f.write("122{}".format(os.linesep))
		file='{}/output/'.format(self.cwd_work)+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_RPSS_'+self.tgts[tar_ndx]+'_'+self.mons[tar_ndx]+'{}'.format(os.linesep)
		f.write(os.path.normpath(file))

		# Probabilistic skill maps
		f.write("437{}".format(os.linesep))
		# save Ranked Probability Skill Score (all cats)
		f.write("131{}".format(os.linesep))
		file='{}/output/'.format(self.cwd_work)+self.models[model_ndx]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_GROC_'+self.tgts[tar_ndx]+'_'+self.mons[tar_ndx]+'{}'.format(os.linesep)
		f.write(os.path.normpath(file))


		# Exit
		f.write("0{}".format(os.linesep))
		f.write("0{}".format(os.linesep))
		f.close()
		if platform.system() == 'Windows':
			os.chdir('scripts')
			get_ipython().system("copy params "+self.models[model_ndx]+"_"+self.fprefix+"_"+self.mpref+"_"+self.tgts[tar_ndx]+"_"+self.mons[tar_ndx]+".cpt")
			os.chdir(self.working_directory)
			os.chdir(self.workdir)
		else:
			get_ipython().system("cp ./scripts/params ./scripts/"+self.models[model_ndx]+"_"+self.fprefix+"_"+self.mpref+"_"+self.tgts[tar_ndx]+"_"+self.mons[tar_ndx]+".cpt")

		if flag:
			self.models = self._tempmods

	def run(self, tar_ndx, model_ndx=-1):
		flag=0
		if model_ndx == -1:
			self._tempmods = copy.deepcopy(self.models)
			self.models=['NextGen']
			flag = 1
			model_ndx = 0

		print('Executing CPT for '+self.models[model_ndx]+' and initialization '+self.mons[tar_ndx]+'...')
		try:
			if platform.system() == "Windows":
				f2 = open('./cwd.txt', 'w')
				sp.call(['echo', '%cd%'], stdout=f2, shell=True)
				f2.close()
				f2 = open('./cwd.txt', 'r')
				cwd = f2.readline().strip()
				cwd = cwd.replace('\\', '/')
				f2.close()
				sp.check_output('cmd /c ' + os.path.normpath(self.cptdir + 'CPT_Batch.exe') + ' < ' + os.path.normpath('{}/scripts/params'.format(cwd)) + ' > ' + os.path.normpath('{}/scripts/CPT_stout_train_'.format( cwd)+self.models[model_ndx]+'_'+self.tgts[tar_ndx]+'_'+self.mons[tar_ndx]+'.txt') )
			else:
				sp.check_output(self.cptdir+'CPT.x < ./scripts/params > ./scripts/CPT_stout_train_'+self.models[model_ndx]+'_'+self.tgts[tar_ndx]+'_'+self.mons[tar_ndx]+'.txt',stderr=sp.STDOUT, shell=True)
		except sp.CalledProcessError as e:
			print('Unfortunately Windows Batch version doesnt do quite everything correctly- we get a memory access error here, but the rest of the notebook will still work fine! Everything for NextGen Skill analysis / forecasting works perfect')
			print(e.output.decode())
		print('----------------------------------------------')
		print('Calculations for '+self.mons[tar_ndx]+' initialization completed!')
		print('See output folder, and check scripts/CPT_stout_train_'+self.models[model_ndx]+'_'+self.tgts[tar_ndx]+'_'+self.mons[tar_ndx]+'.txt for errors')
		print('----------------------------------------------')
		print('----------------------------------------------\n\n{}'.format(os.linesep))

		if flag:
			self.MOS = self._tempmos
			self.models = self._tempmods

	def pltdomain(self):
		"""A simple plot function for the geographical domain

		PARAMETERS
		----------
			loni: western longitude
			lone: eastern longitude
			lati: southern latitude
			late: northern latitude
			title: title
		"""


		try:
			self.shape_feature = ShapelyFeature(Reader(self.shp_file).geometries(), ccrs.PlateCarree(), facecolor='none')
		except:
			pass #print('Failed to load custom shape file')
		#Create a feature for States/Admin 1 regions at 1:10m from Natural Earth
		states_provinces = feature.NaturalEarthFeature(
			category='cultural',
			#name='admin_1_states_provinces_shp',
			name='admin_0_countries',
			scale='10m',
			facecolor='none')
		#ax.add_feature(cfeature.LAND)
	    #ax.add_feature(cfeature.COASTLINE)
	    #ax.add_feature(states_provinces, edgecolor='gray')


		fig = plt.subplots(figsize=(15,15), subplot_kw=dict(projection=ccrs.PlateCarree()))
		loni = [self.wlo1,self.wlo2]
		lati = [self.nla1,self.nla2]
		lone = [self.elo1,self.elo2]
		late = [self.sla1,self.sla2]
		title = ['Predictor', 'Predictand']

		for i in range(2):

			ax = plt.subplot(1, 2, i+1, projection=ccrs.PlateCarree())
			ax.set_extent([loni[i],lone[i],lati[i],late[i]], ccrs.PlateCarree())

			# Put a background image on for nice sea rendering.
			#ax.stock_img()

			ax.add_feature(feature.LAND)
			#ax.add_feature(feature.COASTLINE)
			ax.add_feature(feature.OCEAN)
			try:
				ax.add_feature(self.shape_feature, edgecolor='black')
			except:
				pass
			ax.set_title(title[i]+" domain")
			pl=ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
					  linewidth=1, color='gray', alpha=0.5, linestyle=(0,(2,4)))
			pl.xlabels_top = False
			pl.ylabels_left = False
			pl.xformatter = LONGITUDE_FORMATTER
			pl.yformatter = LATITUDE_FORMATTER
			if self.use_default == 'True':
				ax.add_feature(states_provinces, edgecolor='black')
			#states_provinces = cfeature.NaturalEarthFeature(
	        #category='cultural',
	        #name='admin_1_states_provinces_lines',
	        #scale='50m',
	        #facecolor='none')
		plt.savefig("./images/domain_{}_{}_{}_{}.png".format(loni[0],lone[0],lati[0],late[0]),dpi=300, bbox_inches='tight') #SAVE_FILE 0_domain.png
		plt.show()

	def pltmap(self, score_ndx, isNextGen=-1):
		"""A simple function for ploting the statistical scores

		PARAMETERS
		----------
			score: the score - this is one of ['Pearson','Spearman','2AFC','RocAbove','RocBelow']
			loni: western longitude
			lone: eastern longitude
			lati: southern latitude
			late: northern latitude
		"""
		if isNextGen != -1:
			self.models = ['NextGen']

		try:
			shape_feature = ShapelyFeature(Reader(self.shp_file).geometries(), ccrs.PlateCarree(), facecolor='none')
		except:
			pass#print('Failed to load custom shape file')

		nmods=len(self.models)
		nsea=len(self.mons)
		if self.obs == 'ENACTS-BD':
			x_offset = 0.6
			y_offset = 0.4
		else:
			x_offset = 0
			y_offset = 0

		fig, ax = plt.subplots(nrows=nmods, ncols=nsea, figsize=(6*nsea, 6*nmods),sharex=False,sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})
		#fig = plt.figure(figsize=(20,40))
		#ax = [ plt.subplot2grid((nmods+1, nsea), (int(np.floor(nd / nsea)), int(nd % nsea)),rowspan=1, colspan=1, projection=ccrs.PlateCarree()) for nd in range(nmods*nsea) ]
		#ax.append(plt.subplot2grid((nmods+1, nsea), (nmods, 0), colspan=nsea ) )
		if nsea==1 and nmods == 1:
			ax = [[ax]]
		elif nsea == 1:
			ax = [[ax[q]] for q in range(nmods)]
		elif nmods == 1:
			ax = [ax]
		if self.met[score_ndx] not in ['Pearson','Spearman']:
			#urrent_cmap = plt.cm.get_cmap('RdYlBu', 10 )
			current_cmap = self.make_cmap(10)
		else:
			#current_cmap = plt.cm.get_cmap('RdYlBu', 14 )
			current_cmap = self.make_cmap(14)
		#current_cmap.set_bad('white',1.0)
		#current_cmap.set_under('white', 1.0)
		print()
		print(self.met[score_ndx])
		for i in range(nmods):
			for j in range(nsea):
				mon=self.mons[j]
				#Read grads binary file size H, W  --it assumes all files have the same size, and that 2AFC exists
				with open('{}/output/'.format(self.cwd_work)+self.models[i]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_2AFC_'+self.tgts[j]+'_'+mon+'.ctl', "r") as fp:
					for line in self.lines_that_contain("XDEF", fp):
						W = int(line.split()[1])
						XD= float(line.split()[4])
				with open('{}/output/'.format(self.cwd_work)+self.models[i]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_2AFC_'+self.tgts[j]+'_'+mon+'.ctl', "r") as fp:
					for line in self.lines_that_contain("YDEF", fp):
						H = int(line.split()[1])
						YD= float(line.split()[4])

	#			ax = plt.subplot(nwk/2, 2, wk, projection=ccrs.PlateCarree())

				#ax = plt.subplot(nmods,nsea, k, projection=ccrs.PlateCarree())
				if self.obs == 'ENACTS-BD':
				#	wlo2,elo2,sla2,nla2 => loni,lone,lati,late
					ax[i][j].set_extent([self.wlo2+x_offset,self.wlo2+W*XD+x_offset,self.sla2+y_offset,self.sla2+H*YD+y_offset], ccrs.PlateCarree())
				else:
					ax[i][j].set_extent([self.wlo2+x_offset,self.wlo2+W*XD+x_offset,self.sla2+y_offset,self.sla2+H*YD+y_offset], ccrs.PlateCarree())
				#Create a feature for States/Admin 1 regions at 1:10m from Natural Earth
				states_provinces = feature.NaturalEarthFeature(
					category='cultural',
	#				name='admin_1_states_provinces_shp',
					name='admin_0_countries',
					scale='10m',
					facecolor='none')

				ax[i][j].add_feature(feature.LAND)
				#ax[i][j].add_feature(feature.COASTLINE)
				pl=ax[i][j].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
					  linewidth=1, color='gray', alpha=0.5, linestyle=(0,(2,4)))
				pl.xlabels_top = False
				pl.ylabels_left = True
				pl.ylabels_right = False
				#pl.xlabels_bottom = False
				#if i == nmods - 1: change so long vals in every plot
				pl.xlabels_bottom = True
				pl.xformatter = LONGITUDE_FORMATTER
				pl.yformatter = LATITUDE_FORMATTER
				try:
					ax[i][j].add_feature(self.shape_feature, edgecolor='black')
				except:
					print('failed to add your shape file')
				if self.use_default == 'True':
					ax[i][j].add_feature(states_provinces, edgecolor='black')
				ax[i][j].set_ybound(lower=self.sla2, upper=self.nla2)

				if j == 0:
					ax[i][j].text(-0.42, 0.5, self.models[i],rotation='vertical', verticalalignment='center', horizontalalignment='center', transform=ax[i][j].transAxes)
				if i == 0:
					ax[i][j].set_title(self.tgts[j])
				#for i, axi in enumerate(axes):  # need to enumerate to slice the data
				#	axi.set_ylabel(model, fontsize=12)


				if self.met[score_ndx] == 'CCAFCST_V' or self.met[score_ndx] == 'PCRFCST_V':
					if platform.system() == "Windows":
						f=open('{}/output/'.format(self.cwd_work)+self.models[i]+'_'+self.fprefix+'_'+self.met[score_ndx]+'_'+training_season+'_'+mon+str(fday)+'_wk'+str(wk)+'.dat','rb')
						garbage=struct.unpack('s',f.read(1))[0]
						recl=struct.unpack('i', f.read(4))[0]
						A = np.fromfile(f, dtype='float32', count=W*H)
					else:
						f=open('{}/output/'.format(self.cwd_work)+self.models[i]+'_'+self.fprefix+'_'+self.met[score_ndx]+'_'+training_season+'_'+mon+str(fday)+'_wk'+str(wk)+'.dat','rb')
						recl=struct.unpack('i',f.read(4))[0]
						numval=int(recl/np.dtype('float32').itemsize)
						#Now we read the field
						A=np.fromfile(f,dtype='float32',count=numval)

					var = np.transpose(A.reshape((W, H), order='F'))
					var[var==-999.]=np.nan #only sensible values

					CS=ax[i][j].pcolormesh(np.linspace(self.wlo2+x_offset, self.wlo2+W*XD+x_offset,num=W), np.linspace(self.sla2+H*YD+y_offset, self.sla2+y_offset, num=H), var,
						#vmin=-max(np.max(var),np.abs(np.min(var))), #vmax=np.max(var),
						norm=MidpointNormalize(midpoint=0.),
						cmap=current_cmap,
						transform=ccrs.PlateCarree())
					ax[i][j].set_title("Deterministic forecast for Week "+str(wk))
					if self.fprefix == 'RFREQ':
						label ='Freq Rainy Days (days)'
					elif self.fprefix == 'PRCP':
						label = 'Rainfall anomaly (mm/week)'
						f.close()
					#current_cmap = plt.cm.get_cmap()
					#current_cmap.set_bad(color='white')
					#current_cmap.set_under('white', 1.0)
				else:
					#Since CPT writes grads files in sequential format, we need to excise the 4 bytes between records (recl)
					if platform.system() == "Windows":
						f=open('{}/output/'.format(self.cwd_work)+self.models[i]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_'+self.met[score_ndx]+'_'+self.tgts[j]+'_'+mon+'.dat','rb')
						garb=struct.unpack('s',f.read(1))[0]
						recl = struct.unpack('i', f.read(4))[0]
						A = np.fromfile(f, dtype='float32', count=W*H)
					else:
						f=open('{}/output/'.format(self.cwd_work)+self.models[i]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_'+self.met[score_ndx]+'_'+self.tgts[j]+'_'+mon+'.dat','rb')
						recl=struct.unpack('i', f.read(4))[0]
						numval=int(recl/np.dtype('float32').itemsize)
						#Now we read the field
						A=np.fromfile(f,dtype='float32',count=numval)

					var = np.transpose(A.reshape((W, H), order='F'))
					#define colorbars, depending on each score	--This can be easily written as a function
					if self.met[score_ndx] == '2AFC':
						var[var<0]=np.nan #only positive values
						CS=ax[i][j].pcolormesh(np.linspace(self.wlo2+x_offset, self.wlo2+W*XD+x_offset,num=W), np.linspace(self.sla2+H*YD+y_offset, self.sla2+y_offset, num=H), var,
						vmin=0,vmax=100,
						cmap=current_cmap,
						transform=ccrs.PlateCarree())
						label = '2AFC (%)'

					if self.met[score_ndx] == 'RMSE':
						var[var<0]=np.nan #only positive values
						CS=ax[i][j].pcolormesh(np.linspace(self.wlo2+x_offset, self.wlo2+W*XD+x_offset,num=W), np.linspace(self.sla2+H*YD+y_offset, self.sla2+y_offset, num=H), var,
						vmin=0,vmax=1000,
						cmap=plt.get_cmap('Reds', 10),
						transform=ccrs.PlateCarree())
						label = 'RMSE'
					if self.met[score_ndx] == 'RocAbove' or self.met[score_ndx]=='RocBelow':
						var[var<0]=np.nan #only positive values
						CS=ax[i][j].pcolormesh(np.linspace(self.wlo2+x_offset, self.wlo2+W*XD+x_offset,num=W), np.linspace(self.sla2+H*YD+y_offset, self.sla2+y_offset, num=H), var,
						vmin=0,vmax=1,
						cmap=current_cmap,
						transform=ccrs.PlateCarree())
						label = 'ROC area'

					if self.met[score_ndx] == 'Spearman' or self.met[score_ndx]=='Pearson':
						var[var<-1.]=np.nan #only sensible values
						CS=ax[i][j].pcolormesh(np.linspace(self.wlo2+x_offset, self.wlo2+W*XD+x_offset,num=W), np.linspace(self.sla2+H*YD+y_offset, self.sla2+y_offset, num=H), var,
						vmin=-1.05,vmax=1.05,
						cmap=current_cmap,
						transform=ccrs.PlateCarree())
						label = 'Correlation'


				#Make true if you want cbar on left, default is cbar on right
				is_left = False
				if is_left:
					axins = inset_axes(ax[i][j],
		                   width="5%",  # width = 5% of parent_bbox width
		                   height="100%",  # height : 50%
		                   loc='center left',
		                   bbox_to_anchor=(-0.22, 0., 1, 1),
		                   bbox_transform=ax[i][j].transAxes,
		                   borderpad=0.1,
		                   )
					if self.met[score_ndx] in ['Pearson','Spearman']:
						 bounds = [-0.9, -0.75, -0.6, -0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
						 cbar = fig.colorbar(CS, ax=ax[i][j], cax=axins,  orientation='vertical', pad=0.02, ticks=bounds)
					elif self.met[score_ndx] == '2AFC':
						bounds = [10*gt for gt in range(1,10, 2)]
						cbar = fig.colorbar(CS, ax=ax[i][j], cax=axins, orientation='vertical', pad=0.02, ticks=bounds)
					elif self.met[score_ndx] == 'RMSE':
						bounds = [10*gt for gt in range(1,10, 2)]
						cbar = fig.colorbar(CS, ax=ax[i][j], cax=axins, orientation='vertical', pad=0.02)#, ticks=bounds)
					else:
						bounds = [round(0.1*gt,1) for gt in range(1,10, 2)]
						cbar = fig.colorbar(CS, ax=ax[i][j], cax=axins, orientation='vertical', pad=0.02, ticks=bounds)
					#cbar.set_label(label) #, rotation=270)\
					#axins.yaxis.tick_left()
				else:
					axins = inset_axes(ax[i][j],
		                   width="5%",  # width = 5% of parent_bbox width
		                   height="100%",  # height : 50%
		                   loc='center right',
		                   bbox_to_anchor=(0., 0., 1.15, 1),
		                   bbox_transform=ax[i][j].transAxes,
		                   borderpad=0.1,
		                   )
					if self.met[score_ndx] in ['Pearson','Spearman']:
						 bounds = [-0.9, -0.75, -0.6, -0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
						 cbar = fig.colorbar(CS, ax=ax[i][j], cax=axins,  orientation='vertical', pad=0.02, ticks=bounds)
					elif self.met[score_ndx] == '2AFC':
						bounds = [10*gt for gt in range(1,10, 2)]
						cbar = fig.colorbar(CS, ax=ax[i][j], cax=axins, orientation='vertical', pad=0.02, ticks=bounds)
					elif self.met[score_ndx] == 'RMSE':
						bounds = [10*gt for gt in range(1,10, 2)]
						cbar = fig.colorbar(CS, ax=ax[i][j], cax=axins, orientation='vertical', pad=0.02)#, ticks=bounds)
					else:
						bounds = [round(0.1*gt,1) for gt in range(1,10, 2)]
						cbar = fig.colorbar(CS, ax=ax[i][j], cax=axins, orientation='vertical', pad=0.02, ticks=bounds)
					cbar.set_label(label) #, rotation=270)\
					#axins.yaxis.tick_left()
				f.close()
		if self.models[0] == 'NextGen':
			filename =  'NextGen_' + self.met[score_ndx]
		else:
			filename = 'Models_' + self.met[score_ndx]
		fig.savefig('./images/' + filename + '.png', dpi=500, bbox_inches='tight')
		plt.show()

	def plteofs(self, mode):
		"""A simple function for ploting EOFs computed by CPT

		PARAMETERS
		----------
			models: list of models to plot
			predictand: exactly that
			mode: EOF being visualized
			M: total number of EOFs computed by CPT (max defined in PyCPT is 10)
			loni: western longitude
			lone: eastern longitude
			lati: southern latitude
			late: northern latitude
			fprefix:


			---plot observations - check angel version
		"""
		try:
			shape_feature = ShapelyFeature(Reader(self.shp_file).geometries(), ccrs.PlateCarree(), facecolor='none')
		except:
			pass #print('Failed to load custom shape file')

		M = self.eof_modes
		#mol=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
		if self.mpref=='None':
			print('No EOFs are computed if MOS=None is used')
			return
		print('\n\n\n-------------EOF {}-------------{}'.format(mode+1, os.linesep))
		nmods=len(self.models) + 1 #nmods + obs
		nsea=len(self.mons)
		tari=self.tgts[0]
		model=self.models[0]
		monn=self.mons[0]

		with open('{}/output/'.format(self.cwd_work)+model+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_EOFX_'+tari+'_'+monn+'.ctl', "r") as fp:
			for line in self.lines_that_contain("XDEF", fp):
				W = int(line.split()[1])
				XD= float(line.split()[4])
		with open('{}/output/'.format(self.cwd_work)+model+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_EOFX_'+tari+'_'+monn+'.ctl', "r") as fp:
			for line in self.lines_that_contain("YDEF", fp):
				H = int(line.split()[1])
				YD= float(line.split()[4])

		if self.mpref=='CCA':
			with open('{}/output/'.format(self.cwd_work)+model+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_EOFY_'+tari+'_'+monn+'.ctl', "r") as fp:
				for line in self.lines_that_contain("XDEF", fp):
					Wy = int(line.split()[1])
					XDy= float(line.split()[4])
			with open('{}/output/'.format(self.cwd_work)+model+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_EOFY_'+tari+'_'+monn+'.ctl', "r") as fp:
				for line in self.lines_that_contain("YDEF", fp):
					Hy = int(line.split()[1])
					YDy= float(line.split()[4])
			eofy=np.empty([M,Hy,Wy])  #define array for later use

		eofx=np.empty([M,H,W])  #define array for later use

		#plt.figure(figsize=(20,10))
		#fig, ax = plt.subplots(figsize=(20,15),sharex=True,sharey=True)
		fig, ax = plt.subplots(nrows=nmods, ncols=nsea, sharex=False,sharey=False, figsize=(10*nsea,6*nmods), subplot_kw={'projection': ccrs.PlateCarree()})
		if nsea==1 and nmods == 1:
			ax = [[ax]]
		elif nsea == 1:
			ax = [[ax[q]] for q in range(nmods)]
		elif nmods == 1:
			ax = [ax]

		#current_cmap = plt.cm.get_cmap('RdYlBu', 14)
		current_cmap = self.make_cmap(14)
		#current_cmap.set_bad('white',1.0)
		#current_cmap.set_under('white', 1.0)

		for i in range(nmods):
			for j in range(nsea):

				tari=self.tgts[j]
				if i == 0:
					model = self.models[0]
				else:
					model=self.models[i-1]



				if i == 0 and self.obs == 'ENACTS-BD':
					ax[i][j].set_extent([87.5,93,20.5,27], crs=ccrs.PlateCarree())
				else:
					if self.mpref=='PCR':
						ax[i][j].set_extent([self.wlo1,self.wlo1+W*XD,self.sla1,self.sla1+H*YD], crs=ccrs.PlateCarree())  #EOF domains will look different between CCA and PCR if X and Y domains are different
					else:
						ax[i][j].set_extent([self.wlo1,self.wlo1+Wy*XDy,self.sla1,self.sla1+Hy*YDy], crs=ccrs.PlateCarree())

				#Create a feature for States/Admin 1 regions at 1:10m from Natural Earth
				states_provinces = feature.NaturalEarthFeature(
					category='cultural',
	#				name='admin_1_states_provinces_shp',
					name='admin_0_countries',
					scale='10m',
					facecolor='none')

				ax[i][j].add_feature(feature.LAND)
				#ax[i][j].add_feature(feature.COASTLINE)

				#tick_spacing=0.5
				#ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

				pl=ax[i][j].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
					  linewidth=1, color='gray', alpha=0.5, linestyle=(0,(2,4)))
				pl.xlabels_top = False
				pl.ylabels_left = True
				pl.ylabels_right = False
				pl.xlabels_bottom = True
				pl.xformatter = LONGITUDE_FORMATTER
				pl.yformatter = LATITUDE_FORMATTER
				try:
					ax[i][j].add_feature(self.shape_feature, edgecolor='black')
				except:
					pass
				if self.use_default == 'True':
					ax[i][j].add_feature(states_provinces, edgecolor='black')
				if self.obs == 'ENACTS-BD' and i ==0:
					ax[i][j].set_ybound(lower=20.5, upper=27)
					ax[i][j].set_xbound(lower=87.5, upper=93)
				else:
					ax[i][j].set_ybound(lower=self.sla1, upper=self.nla1)
					ax[i][j].set_xbound(lower=self.wlo1, upper=self.elo1)


				if j == 0:
					if i == 0:
						ax[i][j].text(-0.42, 0.5, 'Obs',rotation='vertical', verticalalignment='center', horizontalalignment='center', transform=ax[i][j].transAxes)
					else:
						ax[i][j].text(-0.42, 0.5, self.models[i-1],rotation='vertical', verticalalignment='center', horizontalalignment='center', transform=ax[i][j].transAxes)

				if i == 0: #kjch101620
					ax[i][j].set_title(self.tgts[j])
				#Since CPT writes grads files in sequential format, we need to excise the 4 bytes between records (recl)
				if i ==0:
					tari=self.tgts[j]
					model=self.models[0]
					monn=self.mons[0]
					nsea=len(self.mons)
					tar = self.tgts[j]
					mon=self.mons[j]
					if self.mpref == 'CCA':
						#f=open('{}/output/'.format(self.cwd_work)+models[0]+'_'+fprefix+predictand+'_'+mpref+'FCST_Obs_'+mons[j]+'_'+mon+str(fyr)+'.dat','rb')
						#f=open('{}/output/'.format(self.cwd_work)+models[i-1]+'_'+fprefix+predictand+'_'+mpref+'_EOFX_'+mons[j]+'_'+mon+'.dat','rb')
						f=open('{}/output/'.format(self.cwd_work)+model+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_EOFY_'+tar+'_'+mon+'.dat','rb')
						#cycle for all time steps  (same approach to read GrADS files as before, but now read T times)
						for mo in range(M):
							#Now we read the field
							if platform.system() == 'Windows':
								blah = struct.unpack('s', f.read(1))
								recl = struct.unpack('i', f.read(4))
								A0=np.fromfile(f, dtype='float32', count=Wy*Hy)
								recl = struct.unpack('i', f.read(4))
								blah = struct.unpack('s', f.read(1))
							else:
								recl=struct.unpack('i',f.read(4))[0]
								numval=int(recl/np.dtype('float32').itemsize) #this if for each time/EOF stamp
								A0=np.fromfile(f,dtype='float32',count=numval)
								endrec=struct.unpack('i',f.read(4))[0]  #needed as Fortran sequential repeats the header at the end of the record!!!
							eofy[mo,:,:]= np.transpose(A0.reshape((Wy, Hy), order='F'))
						eofy[eofy==-999.]=np.nan #nans

						if self.obs == 'ENACTS-BD':
							CS=ax[i][j].pcolormesh(np.linspace(87.6, 93.0,num=Wy), np.linspace(27.1, 20.4, num=Hy), eofy[mode,:,:],
							vmin=-.1,vmax=.1,
							cmap=current_cmap,
							transform=ccrs.PlateCarree())
						else:
						#CS=ax[i][j].pcolormesh(np.linspace(loni, loni+Wy*XDy,num=Wy), np.linspace(lati+Hy*YDy, lati, num=Hy), eofy[mode,:,:],
							CS=ax[i][j].pcolormesh(np.linspace(self.wlo1, self.elo1,num=Wy), np.linspace(self.nla1, self.sla1, num=Hy), eofy[mode,:,:],
							vmin=-.1,vmax=.1,
							cmap=current_cmap,
							transform=ccrs.PlateCarree())

						label = 'EOF charges'
					else:
						mon=self.mons[j]
						f=open('{}/output/'.format(self.cwd_work)+model+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_EOFX_'+self.tgts[j]+'_'+mon+'.dat','rb')
						#cycle for all time steps  (same approach to read GrADS files as before, but now read T times)
						for mo in range(M):
							#Now we read the field
							if platform.system() == "Windows":
								blah = struct.unpack('s', f.read(1))
								recl = struct.unpack('i', f.read(4))
								A0=np.fromfile(f, dtype='float32', count=W*H)
								recl = struct.unpack('i', f.read(4))
								blah = struct.unpack('s', f.read(1))
							else:
								recl=struct.unpack('i',f.read(4))[0]
								numval=int(recl/np.dtype('float32').itemsize) #this if for each time/EOF stamp
								A0=np.fromfile(f,dtype='float32',count=numval)
								endrec=struct.unpack('i',f.read(4))[0]  #needed as Fortran sequential repeats the header at the end of the record!!!
							eofx[mo,:,:]= np.transpose(A0.reshape((W, H), order='F'))

						eofx[eofx==-999.]=np.nan #nans
						CS=ax[i][j].pcolormesh(np.linspace(self.wlo1, self.wlo1+W*XD,num=W), np.linspace(self.sla1+H*YD, self.sla1, num=H), eofx[mode,:,:],
						vmin=-.105, vmax=.105,
						cmap=current_cmap,
						transform=ccrs.PlateCarree())
						label = 'EOF charges'

				else:
					mon=self.mons[j]
					f=open('{}/output/'.format(self.cwd_work)+self.models[i-1]+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'_EOFX_'+self.tgts[j]+'_'+mon+'.dat','rb')
					#cycle for all time steps  (same approach to read GrADS files as before, but now read T times)
					for mo in range(M):
						#Now we read the field
						if platform.system() == 'Windows':
							if self.mpref == 'CCA':
								blah = struct.unpack('s', f.read(1))
								recl = struct.unpack('i', f.read(4))[0]
								A0=np.fromfile(f, dtype='float32', count=int(recl / 4) )
								recl = struct.unpack('i', f.read(4))
								blah = struct.unpack('s', f.read(1))
							else:
								blah = struct.unpack('s', f.read(1))
								recl = struct.unpack('i', f.read(4))[0]
								A0=np.fromfile(f, dtype='float32', count=int(recl / 4))
								recl = struct.unpack('i', f.read(4))
								blah = struct.unpack('s', f.read(1))
						else:
							recl=struct.unpack('i',f.read(4))[0]
							numval=int(recl/np.dtype('float32').itemsize) #this if for each time/EOF stamp
							A0=np.fromfile(f,dtype='float32',count=numval)
							endrec=struct.unpack('i',f.read(4))[0]  #needed as Fortran sequential repeats the header at the end of the record!!!
						if self.mpref == 'CCA':
							eofx[mo,:,:]= np.transpose(A0.reshape((W, H), order='F'))
						else:
							eofx[mo,:,:]= np.transpose(A0.reshape((W, H), order='F'))


					eofx[eofx==-999.]=np.nan #nans
					CS=ax[i][j].pcolormesh(np.linspace(self.wlo1, self.wlo1+W*XD,num=W), np.linspace(self.sla1+H*YD, self.sla1, num=H), eofx[mode,:,:],
					vmin=-.105, vmax=.105,
					cmap=current_cmap,
					transform=ccrs.PlateCarree())
					label = 'EOF charges'

				is_left = False
				if is_left:
					axins = inset_axes(ax[i][j],
		                   width="5%",  # width = 5% of parent_bbox width
		                   height="100%",  # height : 50%
		                   loc='center left',
		                   bbox_to_anchor=(-0.22, 0., 1, 1),
		                   bbox_transform=ax[i][j].transAxes,
		                   borderpad=0.1,
		                   )
				else:
					axins = inset_axes(ax[i][j],
		                   width="5%",  # width = 5% of parent_bbox width
		                   height="100%",  # height : 50%
		                   loc='center right',
		                   bbox_to_anchor=(0., 0., 1.15, 1),
		                   bbox_transform=ax[i][j].transAxes,
		                   borderpad=0.1,
		                   )
				cbar = plt.colorbar(CS,ax=ax[i][j], cax=axins, orientation='vertical', pad=0.01, ticks= [-0.09, -0.075, -0.06, -0.045, -0.03, -0.015, 0, 0.015, 0.03, 0.045, 0.06, 0.075, 0.09])
				#cbar.set_label(label) #, rotation=270)
				#axins.yaxis.tick_left()
				f.close()
		model_names = ['obs']
		model_names.extend(self.models)
		if self.models[0] == 'NextGen':
			fig.savefig('./images/EOF{}_NextGen.png'.format(mode+1, dpi=500, bbox_inches='tight'))
		else:
			fig.savefig('./images/EOF{}_Models.png'.format(mode+1, dpi=500, bbox_inches='tight'))
				#plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
				#cbar_ax = plt.add_axes([0.85, 0.15, 0.05, 0.7])
				#plt.tight_layout()

				#plt.autoscale(enable=True)
		#plt.subplots_adjust(bottom=0.15, top=0.9)
		#cax = plt.axes([0.2, 0.08, 0.6, 0.04])
		plt.show()

	def setNextGenModels(self, models):
		self.original_models = self.models
		self.models = models

	def make_cmap_blue(self,x):
		colors = [(244, 255,255),
		(187, 252, 255),
		(160, 235, 255),
		(123, 210, 255),
		(89, 179, 238),
		(63, 136, 254),
		(52, 86, 254)]
		colors = [ (colors[i][0] / 255.0, colors[i][1] / 255.0, colors[i][2] / 255.0) for i in range(len(colors))]
		#colors.reverse()
		return LinearSegmentedColormap.from_list( "matlab_clone", colors, N=x)

	def make_cmap(self, x):
		colors = [(238, 43, 51),
		(255, 57, 67),
		(253, 123, 91),
		(248, 175, 123),
		(254, 214, 158),
		(252, 239, 188),
		(255, 254, 241),
		(244, 255,255),
		(187, 252, 255),
		(160, 235, 255),
		(123, 210, 255),
		(89, 179, 238),
		(63, 136, 254),
		(52, 86, 254)
		]
		colors = [ (colors[i][0] / 255.0, colors[i][1] / 255.0, colors[i][2] / 255.0) for i in range(len(colors))]
		colors.reverse()
		return LinearSegmentedColormap.from_list( "matlab_clone", colors, N=x)

	def make_cmap_gray(self, x):
		colors = [(55,55,55),(235,235,235)]
		colors = [ (colors[i][0] / 255.0, colors[i][1] / 255.0, colors[i][2] / 255.0) for i in range(len(colors))]
		colors.reverse()
		return LinearSegmentedColormap.from_list( "matlab_clone", colors, N=x)

	def readGrADSctl(self, models,fprefix,predictand,mpref,id,tar,monf,fyr):
		#Read grads binary file size H, W, T
		with open('{}/output/'.format(self.cwd_work)+str(models[0])+'_'+fprefix+predictand+'_'+mpref+id+'_'+tar+'_'+monf+str(fyr)+'.ctl', "r") as fp:
			for line in self.lines_that_contain("XDEF", fp):
				W = int(line.split()[1])
				Wi= float(line.split()[3])
				XD= float(line.split()[4])
		with open('{}/output/'.format(self.cwd_work)+str(models[0])+'_'+fprefix+predictand+'_'+mpref+id+'_'+tar+'_'+monf+str(fyr)+'.ctl', "r") as fp:
			for line in self.lines_that_contain("YDEF", fp):
				H = int(line.split()[1])
				Hi= float(line.split()[3])
				YD= float(line.split()[4])
		with open('{}/output/'.format(self.cwd_work)+str(models[0])+'_'+fprefix+predictand+'_'+mpref+id+'_'+tar+'_'+monf+str(fyr)+'.ctl', "r") as fp:
			for line in self.lines_that_contain("TDEF", fp):
				T = int(line.split()[1])
				Ti= int((line.split()[3])[-4:])
				TD= 1  #not used
		return (W, Wi, XD, H, Hi, YD, T, Ti, TD)

	def lines_that_contain(self, string, fp):
		return [line for line in fp if string in line]

	def writeCPT(self, tar_ndx, var, outfile):
		"""Function to write seasonal output in CPT format,
		using information contained in a GrADS ctl file.

		PARAMETERS
		----------
			var: a Dataframe with dimensions T,Y,X
		"""
		vari = 'prec'
		varname = vari
		units = 'mm'
		var[np.isnan(var)]=-999. #use CPT missing value
		tar = self.tgts[tar_ndx]
		L=0.5*(float(self.tgtf[tar_ndx])+float(self.tgti[tar_ndx]))
		monthdic = {'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06','Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}
		S=monthdic[self.mons[tar_ndx]]
		if '-' in tar:
			mi=monthdic[tar.split("-")[0]]
			mf=monthdic[tar.split("-")[1]]

			#Read grads file to get needed coordinate arrays
			W, Wi, XD, H, Hi, YD, T, Ti, TD = self.readGrADSctl(self.models,self.fprefix,self.PREDICTAND,self.mpref,self.file,tar,self.monf[tar_ndx],self.fyr)
			if tar=='Dec-Feb' or tar=='Nov-Jan':  #double check years are sync
				xyear=True  #flag a cross-year season
			else:
				#Ti=Ti+1
				xyear=False
		else:
			mi=monthdic[tar]
			mf=monthdic[tar]

			#Read grads file to get needed coordinate arrays
			W, Wi, XD, H, Hi, YD, T, Ti, TD = self.readGrADSctl(self.models,self.fprefix,self.PREDICTAND,self.mpref,self.file,tar,self.monf[tar_ndx],self.fyr)
			if tar=='Dec-Feb' or tar=='Nov-Jan':  #double check years are sync
				xyear=True  #flag a cross-year season
			else:
				#Ti=Ti+1
				xyear=False

		Tarr = np.arange(Ti, Ti+T)
		Xarr = np.linspace(Wi, Wi+W*XD,num=W+1)
		Yarr = np.linspace(Hi+H*YD, Hi,num=H+1)

		#Now write the CPT file
		f = open(outfile, 'w')
		f.write("xmlns:cpt=http://iri.columbia.edu/CPT/v10/{}".format(os.linesep))
		#f.write("xmlns:cf=http://cf-pcmdi.llnl.gov/documents/cf-conventions/1.4/\n")   #not really needed
		f.write("cpt:nfields=1{}".format(os.linesep))
		#f.write("cpt:T	" + str(Tarr)+"\n")  #not really needed
		for it in range(T):
			if xyear==True:
				f.write("cpt:field="+vari+", cpt:L="+str(L)+" months, cpt:S="+str(Ti)+"-"+S+"-01T00:00, cpt:T="+str(Tarr[it])+"-"+mi+"/"+str(Tarr[it]+1)+"-"+mf+", cpt:nrow="+str(H)+", cpt:ncol="+str(W)+", cpt:row=Y, cpt:col=X, cpt:units="+units+", cpt:missing=-999.{}".format(os.linesep))
			else:
				f.write("cpt:field="+vari+", cpt:L="+str(L)+" months, cpt:S="+str(Ti)+"-"+S+"-01T00:00, cpt:T="+str(Tarr[it])+"-"+mi+"/"+mf+", cpt:nrow="+str(H)+", cpt:ncol="+str(W)+", cpt:row=Y, cpt:col=X, cpt:units="+units+", cpt:missing=-999.{}".format(os.linesep))
			#f.write("\t")
			np.savetxt(f, Xarr[0:-1], fmt="%.3f",newline='\t') #f.write(str(Xarr)[1:-1])
			f.write("{}".format(os.linesep)) #next line
			for iy in range(H):
				#f.write(str(Yarr[iy]) + "\t" + str(var[it,iy,0:-1])[1:-1]) + "\n")
				np.savetxt(f,np.r_[Yarr[iy+1],var[it,iy,0:]],fmt="%.3f", newline='\t')  #excise extra line
				f.write("{}".format(os.linesep)) #next line
		f.close()

	def NGensemble(self, tar_ndx):
		"""A simple function for computing the NextGen ensemble

		PARAMETERS
		----------
			models: array with selected models
		"""

		nmods=len(self.models)

		W, Wi, XD, H, Hi, YD, T, Ti, TD = self.readGrADSctl(self.models,self.fprefix,self.PREDICTAND,self.mpref,self.file,self.tgts[tar_ndx],self.monf[tar_ndx],self.fyr)

		ens  =np.empty([nmods,T,H,W])  #define array for later use

		k=-1
		for model in self.models:
			print('Preparing CPT files for '+model+' and initialization '+self.mons[tar_ndx]+'...')

			k=k+1 #model
			memb0=np.empty([T,H,W])  #define array for later use

			#Since CPT writes grads files in sequential format, we need to excise the 4 bytes between records (recl)
			f=open('{}/output/'.format(self.cwd_work)+model+'_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+self.file+'_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'.dat','rb')
			#cycle for all time steps  (same approach to read GrADS files as before, but now read T times)
			for it in range(T):
				#Now we read the field
				if platform.system() == 'Windows':
					garbage = struct.unpack('s', f.read(1))[0]
				recl=struct.unpack('i',f.read(4))[0]
				numval=int(recl/np.dtype('float32').itemsize) #this if for each time stamp
				A0=np.fromfile(f,dtype='float32',count=numval)
				endrec=struct.unpack('i',f.read(4))[0]
				if platform.system() == 'Windows':
					garbage = struct.unpack('s', f.read(1))[0]  #needed as Fortran sequential repeats the header at the end of the record!!!
				memb0[it,:,:]= np.transpose(A0.reshape((W, H), order='F'))

			memb0[memb0==-999.]=np.nan #identify NaNs

			ens[k,:,:,:]=memb0

		# NextGen ensemble mean (perhaps try median too?)
		NG=np.nanmean(ens, axis=0)  #axis 0 is ensemble member

		#Now write output:
		#writeCPT(NG,'./output/NextGen_'+fprefix+'_'+tar+'_ini'+mon+'.tsv',models,fprefix,predictand,mpref,id,tar,mon,tgti,tgtf,monf,fyr)
		if self.file=='FCST_xvPr':
			self.writeCPT(tar_ndx, NG,'./input/NextGen_'+self.fprefix+'_'+self.tgts[tar_ndx]+'_ini'+self.mons[tar_ndx]+'.tsv')
			print('Cross-validated prediction files successfully produced')
		if self.file=='FCST_mu':
			self.writeCPT(tar_ndx, NG,'./output/NextGen_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'FCST_mu_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(self.fyr)+'.tsv')
			print('Forecast files successfully produced')
		if self.file=='FCST_var':
			self.writeCPT(tar_ndx, NG,'./output/NextGen_'+self.fprefix+self.PREDICTAND+'_'+self.mpref+'FCST_var_'+self.tgts[tar_ndx]+'_'+self.monf[tar_ndx]+str(fyr)+'.tsv')
			print('Forecast error files successfully produced')

	def plt_ng_probabilistic(self):
		"""A simple function for ploting the statistical scores

		PARAMETERS
		----------
			fcst_type: either 'deterministic' or 'probabilistic'
			loni: western longitude
			lone: eastern longitude
			lati: southern latitude
			late: northern latitude
		"""
		try:
			shape_feature = ShapelyFeature(Reader(self.shp_file).geometries(), ccrs.PlateCarree(), facecolor='none')
		except:
			pass #print('Failed to load custom shape file')

		self._tempmods = copy.deepcopy(self.models)
		self.models = ['NextGen']
		cbar_loc, fancy = 'bottom', True
		nmods=len(self.models)
		nsea=len(self.tgts)
		xdim=1
		#
		list_probabilistic_by_season = [[[], [], []] for i in range(nsea)]
		list_det_by_season = [[] for i in range(nsea)]
		if platform.system() == "Windows":
			for i in range(nmods):
				for j in range(nsea):
					plats, plongs, av = self.read_forecast_bin('probabilistic', self.models[i], self.PREDICTAND, self.mpref, self.tgts[j], self.mons[j], self.fyr )
					for kl in range(av.shape[0]):
						list_probabilistic_by_season[j][kl].append(av[kl])
					dlats, dlongs, av = self.read_forecast_bin('deterministic', self.models[i], self.PREDICTAND, self.mpref, self.tgts[j], self.mons[j], self.fyr )
					list_det_by_season[j].append(av[0])
		else:
			for i in range(nmods):
				for j in range(nsea):
					plats, plongs, av = self.read_forecast('probabilistic', self.models[i], self.PREDICTAND, self.mpref, self.tgts[j], self.mons[j], self.fyr )
					for kl in range(av.shape[0]):
						list_probabilistic_by_season[j][kl].append(av[kl])
					dlats, dlongs, av = self.read_forecast('deterministic', self.models[i], self.PREDICTAND, self.mpref, self.tgts[j], self.mons[j], self.fyr )
					list_det_by_season[j].append(av[0])

		ng_probfcst_by_season = []
		ng_detfcst_by_season = []
		pbn, pn, pan = [],[],[]
		for j in range(nsea):
			p_bn_array = np.asarray(list_probabilistic_by_season[j][0])
			p_n_array = np.asarray(list_probabilistic_by_season[j][1])
			p_an_array = np.asarray(list_probabilistic_by_season[j][2])

			p_bn = np.nanmean(p_bn_array, axis=0) #average over the models
			p_n = np.nanmean(p_n_array, axis=0)   #some areas are NaN
			p_an = np.nanmean(p_an_array, axis=0) #if they are Nan for All, mark

			all_nan = np.zeros(p_bn.shape)
			for ii in range(p_bn.shape[0]):
				for jj in range(p_bn.shape[1]):
					if np.isnan(p_bn[ii,jj]) and np.isnan(p_n[ii,jj]) and np.isnan(p_an[ii,jj]):
						all_nan[ii,jj] = 1
			missing = np.where(all_nan > 0)

			max_ndxs = np.argmax(np.asarray([p_bn, p_n, p_an]), axis=0)
			p_bn[np.where(max_ndxs!= 0)] = np.nan
			p_n[np.where(max_ndxs!= 1)] = np.nan
			p_an[np.where(max_ndxs!= 2)] = np.nan
			pbn.append(p_bn)
			pn.append(p_n)
			pan.append(p_an)

		fig, ax = plt.subplots(nrows=xdim, ncols=nsea, figsize=(nsea*13, xdim*10), sharex=False,sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})

		if nsea == 1:
			ax = [ax]
		ax = [ax]

		for i in range(xdim):
			for j in range(nsea):
				current_cmap = plt.get_cmap('BrBG')
				current_cmap.set_under('white', 0.0)

				current_cmap_copper = plt.get_cmap('YlOrRd', 9)
				current_cmap_binary = plt.get_cmap('Greens', 4)
				current_cmap_ylgn = self.make_cmap_blue(9)

				lats, longs = plats, plongs

				ax[i][j].set_extent([longs[0],longs[-1],lats[0],lats[-1]], ccrs.PlateCarree())

				#Create a feature for States/Admin 1 regions at 1:10m from Natural Earth
				states_provinces = feature.NaturalEarthFeature(
					category='cultural',
		#				name='admin_1_states_provinces_shp',
					name='admin_0_countries',
					scale='10m',
					facecolor='none')

				ax[i][j].add_feature(feature.LAND)
				#ax[i][j].add_feature(feature.COASTLINE)
				pl=ax[i][j].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
					  linewidth=1, color='gray', alpha=0.5, linestyle=(0,(2,4)))
				pl.xlabels_top = False
				pl.ylabels_left = True
				pl.ylabels_right = False
				#pl.xlabels_bottom = False
				#if i == nmods - 1: change so long vals in every plot
				pl.xlabels_bottom = True
				pl.xformatter = LONGITUDE_FORMATTER
				pl.yformatter = LATITUDE_FORMATTER
				pl.xlabel_style = {'size': 8}#'rotation': 'vertical'}
				try:
					ax[i][j].add_feature(self.shape_feature, edgecolor='black')
				except:
					pass
				if self.use_default == 'True':
					ax[i][j].add_feature(states_provinces, edgecolor='black')

				ax[i][j].set_ybound(lower=self.sla2, upper=self.nla2)
				titles = ["Deterministic Forecast", "Probabilistic Forecast (Dominant Tercile)"]


				if j == 0:
					ax[i][j].text(-0.25, 0.5, "Probabilistic Forecast (Dominant Tercile)",rotation='vertical', verticalalignment='center', horizontalalignment='center', transform=ax[i][j].transAxes)

				labels = ['Rainfall (mm)', 'Probability (%)']
				ax[i][j].set_title(self.tgts[j])

				if platform.system() == "Windows":
					#fancy probabilistic
					CS1 = ax[i][j].pcolormesh(longs, lats, pbn[j],
						vmin=35, vmax=80,
						#norm=MidpointNormalize(midpoint=0.),
						cmap=current_cmap_copper)
					CS2 = ax[i][j].pcolormesh(longs, lats, pn[j],
						vmin=35, vmax=55,
						#norm=MidpointNormalize(midpoint=0.),
						cmap=current_cmap_binary)
					CS3 = ax[i][j].pcolormesh(longs, lats, pan[j],
						vmin=35, vmax=80,
						#norm=MidpointNormalize(midpoint=0.),
						cmap=current_cmap_ylgn)
				else:
					#fancy probabilistic
					CS1 = ax[i][j].pcolormesh(np.linspace(longs[0], longs[-1],num=len(longs)), np.linspace(lats[0], lats[-1], num=len(lats)), pbn[j],
						vmin=35, vmax=80,
						#norm=MidpointNormalize(midpoint=0.),
						cmap=current_cmap_copper)
					CS2 = ax[i][j].pcolormesh(np.linspace(longs[0], longs[-1],num=len(longs)), np.linspace(lats[0], lats[-1], num=len(lats)), pn[j],
						vmin=35, vmax=55,
						#norm=MidpointNormalize(midpoint=0.),
						cmap=current_cmap_binary)
					CS3 = ax[i][j].pcolormesh(np.linspace(longs[0], longs[-1],num=len(longs)), np.linspace(lats[0], lats[-1], num=len(lats)), pan[j],
						vmin=35, vmax=80,
						#norm=MidpointNormalize(midpoint=0.),
						cmap=current_cmap_ylgn)

				bounds = [40,45,50,55,60,65,70,75]
				nbounds = [40,45,50]

				#fancy probabilistic cb bottom
				axins_f_bottom = inset_axes(ax[i][j],
	            	width="40%",  # width = 5% of parent_bbox width
	               	height="5%",  # height : 50%
	               	loc='lower left',
	               	bbox_to_anchor=(-0.2, -0.15, 1.2, 1),
	               	bbox_transform=ax[i][j].transAxes,
	               	borderpad=0.1 )
				axins2_bottom = inset_axes(ax[i][j],
	            	width="20%",  # width = 5% of parent_bbox width
	               	height="5%",  # height : 50%
	               	loc='lower center',
	               	bbox_to_anchor=(-0.0, -0.15, 1, 1),
	               	bbox_transform=ax[i][j].transAxes,
	               	borderpad=0.1 )
				axins3_bottom = inset_axes(ax[i][j],
	            	width="40%",  # width = 5% of parent_bbox width
	               	height="5%",  # height : 50%
	               	loc='lower right',
	               	bbox_to_anchor=(0, -0.15, 1.2, 1),
	               	bbox_transform=ax[i][j].transAxes,
	               	borderpad=0.1 )
				cbar_fbl = fig.colorbar(CS1, ax=ax[i][j], cax=axins_f_bottom, orientation='horizontal', ticks=bounds)
				cbar_fbl.set_label('BN Probability (%)') #, rotation=270)\

				cbar_fbc = fig.colorbar(CS2, ax=ax[i][j],  cax=axins2_bottom, orientation='horizontal', ticks=nbounds)
				cbar_fbc.set_label('N Probability (%)') #, rotation=270)\

				cbar_fbr = fig.colorbar(CS3, ax=ax[i][j],  cax=axins3_bottom, orientation='horizontal', ticks=bounds)
				cbar_fbr.set_label('AN Probability (%)') #, rotation=270)\

		fig.savefig('./images/NG_Probabilistic_RealtimeForecasts.png', dpi=500, bbox_inches='tight')
		self.models = self._tempmods

	def read_forecast_bin(self, fcst_type, model, predictand, mpref, mons, mon, fyr):
		if fcst_type == 'deterministic':
			f = open("{}/output/".format(self.cwd_work) + model + '_' + predictand + predictand + '_' + mpref + 'FCST_mu_' +mons + '_' +mon+str(fyr)+'.dat', 'rb')
			garb = struct.unpack('s',f.read(1))[0]
			recl = struct.unpack('i', f.read(4))[0]
			numval = int(recl / np.dtype('float32').itemsize)
			data= np.fromfile(f, dtype='float32', count=numval)
			recl = struct.unpack('i', f.read(4))[0]
			garb = struct.unpack('s', f.read(1))[0]

			with open("{}/output/".format(self.cwd_work) + model + '_' + predictand +predictand + '_' + mpref + 'FCST_mu_' +mons + '_' +mon+str(fyr)+'.ctl', "r") as fp:
				for line in self.lines_that_contain("XDEF", fp):
					W = int(line.split()[1])
					Wi = float(line.split()[3])
					XD= float(line.split()[4])

			with open("{}/output/".format(self.cwd_work) + model + '_' + predictand + predictand + '_' + mpref + 'FCST_mu_' +mons + '_' +mon+str(fyr)+'.ctl', "r") as fp:
				for line in self.lines_that_contain("YDEF", fp):
					H = int(line.split()[1])
					Hi = float(line.split()[3])

					YD= float(line.split()[4])

			data = np.transpose(data.reshape((W, H), order='F'))
			data[np.where(data == -999.0)] = np.nan
			#lons, lats = np.linspace(self.wlo2, self.elo2,num=W), np.linspace(self.sla2, self.nla2, num=H)
			lons = np.linspace(Wi, Wi+W*XD,num=W)
			lats = np.linspace(Hi+H*YD, Hi, num=H)
			return lats, lons, np.asarray([data]) # data.shape = (target, lats, lons)

		else:
			f = open("{}/output/".format(self.cwd_work) + model + '_' + predictand + predictand+ '_' + mpref + 'FCST_P_' +mons + '_' +mon+str(fyr)+'.dat', 'rb')
			garb = struct.unpack('s',f.read(1))[0]
			recl = struct.unpack('i', f.read(4))[0]
			numval = int(recl / np.dtype('float32').itemsize)
			bn_data= np.fromfile(f, dtype='float32', count=numval)
			recl = struct.unpack('i', f.read(4))[0]
			garb = struct.unpack('s', f.read(1))[0]

			garb = struct.unpack('s',f.read(1))[0]
			recl = struct.unpack('i', f.read(4))[0]
			numval = int(recl / np.dtype('float32').itemsize)
			n_data= np.fromfile(f, dtype='float32', count=numval)
			recl = struct.unpack('i', f.read(4))[0]
			garb = struct.unpack('s', f.read(1))[0]

			garb = struct.unpack('s',f.read(1))[0]
			recl = struct.unpack('i', f.read(4))[0]
			numval = int(recl / np.dtype('float32').itemsize)
			an_data= np.fromfile(f, dtype='float32', count=numval)
			recl = struct.unpack('i', f.read(4))[0]
			garb = struct.unpack('s', f.read(1))[0]

			with open("./output/".format(self.cwd_work) + model + '_' + predictand + predictand + '_' + mpref + 'FCST_mu_' +mons + '_' +mon+str(fyr)+'.ctl', "r") as fp:
				for line in self.lines_that_contain("XDEF", fp):
					W = int(line.split()[1])
					Wi = float(line.split()[3])
					XD= float(line.split()[4])

			with open("./output/".format(self.cwd_work) + model + '_' + predictand +predictand + '_' + mpref + 'FCST_mu_' +mons + '_' +mon+str(fyr)+'.ctl', "r") as fp:
				for line in self.lines_that_contain("YDEF", fp):
					H = int(line.split()[1])
					Hi = float(line.split()[3])
					YD= float(line.split()[4])

			bn_data = np.transpose(bn_data.reshape((W, H), order='F'))
			bn_data[np.where(bn_data == -1.0)] = np.nan
			n_data = np.transpose(n_data.reshape((W, H), order='F'))
			n_data[np.where(n_data == -1.0)] = np.nan
			an_data = np.transpose(an_data.reshape((W, H), order='F'))
			an_data[np.where(an_data == -1.0)] = np.nan
			lons = np.linspace(Wi, Wi+W*XD,num=W)
			lats = np.linspace(Hi+H*YD, Hi, num=H)

			#ons, lats = np.linspace(self.wlo2, self.elo2,num=W), np.linspace(self.sla2, self.nla2, num=H)
			return lats, lons, np.asarray([bn_data, n_data, an_data])


	def read_forecast(self, fcst_type, model, predictand, mpref, mons, mon, fyr):
		if fcst_type == 'deterministic':
			f = open("{}/output/".format(self.cwd_work)  + model + '_' + predictand + '_' + mpref + 'FCST_mu_' +mons + '_' +mon+str(fyr)+'.txt', 'r')
		elif fcst_type == 'probabilistic':
			f = open("{}/output/".format(self.cwd_work)  + model + '_' + predictand + '_' + mpref + 'FCST_P_' +mons + '_' +mon+str(fyr)+'.txt', 'r')
		else:
			print('invalid fcst_type')
			return
		lats, all_vals, vals = [], [], []
		flag = 0
		for line in f:
			if line[0:4] == 'cpt:':
				if flag == 2:
					vals = np.asarray(vals, dtype=float)
					if fcst_type == 'deterministic':
						vals[vals == -999.0] = np.nan
					if fcst_type == 'probabilistic':
						vals[vals == -1.0] = np.nan
					all_vals.append(vals)
					lats = []
					vals = []
				flag = 1
			elif flag == 1 and line[0:4] != 'cpt:':
				longs = line.strip().split('\t')
				longs = [float(i) for i in longs]
				flag = 2
			elif flag == 2:
				latvals = line.strip().split('\t')
				lats.append(float(latvals.pop(0)))
				vals.append(latvals)
		vals = np.asarray(vals, dtype=float)
		if fcst_type == 'deterministic':
			vals[vals == -999.0] = np.nan
		if fcst_type == 'probabilistic':
			vals[vals == -1.0] = np.nan
		all_vals.append(vals)
		all_vals = np.asarray(all_vals)
		return lats, longs, all_vals

	def ensemblefiles(self,models,work):
		"""A simple function for preparing the NextGen ensemble files for the DL
		lion brand yarn
		PARAMETERS
		----------
			models: array with selected models
		"""
		if platform.system() == 'Windows':
			get_ipython().system("mkdir " + os.path.normpath("{}/output/NextGen".format(self.cwd_work)) )
			print('made NextGen')
	#		get_ipython().system("mkdir ./output/NextGen/")
		else:
			get_ipython().system("mkdir ./output/NextGen/") #this is fine
		#Go to folder and delate old TXT and TGZ files in folder
		if platform.system() == 'Windows':
			get_ipython().system("del /s /q " +  os.path.normpath("{}/output/NextGen/*_NextGen.tgz".format(self.cwd_work)) )
			print('del /s /q ./output/*_NextGen.tgz')
			get_ipython().system("del /s /q " + os.path.normpath( "{}/output/NextGen/*.txt".format(self.cwd_work)) )
			print('del /s /q *.txt')
		else:
			get_ipython().system("cd ./output/NextGen/; rm -Rf *_NextGen.tgz *.txt")

		for i in range(len(self.models)):
			if platform.system() == 'Windows':
				get_ipython().system("copy " + os.path.normpath("{}/output/*".format(self.cwd_work) +self.models[i]+"*") +" " + os.path.normpath("{}/output/NextGen".format(self.cwd_work)))
				print('copy ./output/*' + self.models[i] + "*.txt ./NextGen/")
			else:
				get_ipython().system("cp ./output/*"+self.models[i]+"*.txt .")
		if platform.system() == "Windows":
			get_ipython().system("tar cvzf " + os.path.normpath("./output/NextGen/" +work+"_NextGen.tgz") + " " + os.path.normpath("./output/Nextgen/*")) #this ~should~ be fine ? unless they have a computer older than last march 2019
		else:
			get_ipython().system("tar cvzf ./output/NextGen/".format(self.cwd_work) +work+"_NextGen.tgz *.txt") #this ~should~ be fine ? unless they have a computer older than last march 2019

		print("tar cvzf ./output/NextGen/"+work+"_NextGen.tgz *.txt")
		if platform.system() == 'Windows':
			get_ipython().system("echo %cd%")
		else:
			get_ipython().system('pwd')

		print("Compressed file "+work+"_NextGen.tgz created in output/NextGen/")
		print("Now send that file to your contact at the IRI")

	def plt_ng_deterministic(self):
		"""A simple function for ploting the statistical scores

		PARAMETERS
		----------
			fcst_type: either 'deterministic' or 'probabilistic'
			loni: western longitude
			lone: eastern longitude
			lati: southern latitude
			late: northern latitude
		"""
		try:
			shape_feature = ShapelyFeature(Reader(self.shp_file).geometries(), ccrs.PlateCarree(), facecolor='none')
		except:
			pass
		self._tempmods = copy.deepcopy(self.models)
		self.models=['NextGen']
		cbar_loc, fancy = 'bottom', True
		nmods=len(self.models)
		nsea=len(self.tgts)
		xdim = 1
		list_probabilistic_by_season = [[[], [], []] for i in range(nsea)]
		list_det_by_season = [[] for i in range(nsea)]
		if platform.system() == "Windows":
			for i in range(nmods):
				for j in range(nsea):
					plats, plongs, av = self.read_forecast_bin('probabilistic', self.models[i], self.PREDICTAND, self.mpref, self.tgts[j], self.mons[j], self.fyr )
					for kl in range(av.shape[0]):
						list_probabilistic_by_season[j][kl].append(av[kl])
					dlats, dlongs, av = self.read_forecast_bin('deterministic', self.models[i], self.PREDICTAND, self.mpref, self.tgts[j], self.mons[j], self.fyr )
					list_det_by_season[j].append(av[0])
		else:
			for i in range(nmods):
				for j in range(nsea):
					plats, plongs, av = self.read_forecast('probabilistic', self.models[i], self.PREDICTAND, self.mpref, self.tgts[j], self.mons[j], self.fyr )
					for kl in range(av.shape[0]):
						list_probabilistic_by_season[j][kl].append(av[kl])
					dlats, dlongs, av = self.read_forecast('deterministic', self.models[i], self.PREDICTAND, self.mpref, self.tgts[j], self.mons[j], self.fyr )
					list_det_by_season[j].append(av[0])


		ng_probfcst_by_season = []
		ng_detfcst_by_season = []
		for j in range(nsea):
			d_array = np.asarray(list_det_by_season[j])
			d_nanmean = np.nanmean(d_array, axis=0)
			ng_detfcst_by_season.append(d_nanmean)

		fig, ax = plt.subplots(nrows=xdim, ncols=nsea, figsize=(nsea*13, xdim*10), sharex=True,sharey=True, subplot_kw={'projection': ccrs.PlateCarree()})
		if nsea == 1:
			ax = [ax]
		ax = [ax]




		for i in range(xdim):
			for j in range(nsea):
				current_cmap = plt.get_cmap('BrBG')
				current_cmap.set_bad('white',0.0)
				current_cmap.set_under('white', 0.0)

				lats, longs = dlats, dlongs
				ax[i][j].set_extent([longs[0],longs[-1],lats[0],lats[-1]], ccrs.PlateCarree())

				#Create a feature for States/Admin 1 regions at 1:10m from Natural Earth
				states_provinces = feature.NaturalEarthFeature(
					category='cultural',
		#				name='admin_1_states_provinces_shp',
					name='admin_0_countries',
					scale='10m',
					facecolor='none')

				ax[i][j].add_feature(feature.LAND)
				#ax[i][j].add_feature(feature.COASTLINE)
				try:
					ax[i][j].add_feature(self.shape_feature, edgecolor='black')
				except:
					pass
				if self.use_default == 'True':
					ax[i][j].add_feature(states_provinces, edgecolor='black')

				pl=ax[i][j].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
					  linewidth=1, color='gray', alpha=0.5, linestyle=(0,(2,4)))
				pl.xlabels_top = False
				pl.ylabels_left = True
				pl.ylabels_right = False
				#pl.xlabels_bottom = False
				#if i == nmods - 1: change so long vals in every plot
				pl.xlabels_bottom = True
				pl.xformatter = LONGITUDE_FORMATTER
				pl.yformatter = LATITUDE_FORMATTER
				ax[i][j].add_feature(states_provinces, edgecolor='black')
				ax[i][j].set_ybound(lower=self.sla2, upper=self.nla2)
				pl.xlabel_style = {'size': 8}#'rotation': 'vertical'}

				titles = ["Deterministic Forecast", "Probabilistic Forecast (Dominant Tercile)"]


				if j == 0:
					ax[i][j].text(-0.25, 0.5, "Deterministic Forecast",rotation='vertical', verticalalignment='center', horizontalalignment='center', transform=ax[i][j].transAxes)

				labels = ['Rainfall (mm)', 'Probability (%)']
				ax[i][j].set_title(self.tgts[j])

				#fancy deterministic
				var = ng_detfcst_by_season[j]
				if platform.system() == "Windows":
					CS_det = ax[i][j].pcolormesh(longs, lats, var,
						norm=MidpointNormalize(midpoint=0.),
						cmap=current_cmap)
				else:
					CS_det = ax[i][j].pcolormesh(np.linspace(longs[0], longs[-1],num=len(longs)), np.linspace(lats[0], lats[-1], num=len(lats)), var,
						norm=MidpointNormalize(midpoint=0.),
						cmap=current_cmap)

				if cbar_loc == 'left':
					#fancy deterministic cb left
					axins_det = inset_axes(ax[i][j],
		            	width="5%",  # width = 5% of parent_bbox width
		               	height="100%",  # height : 50%
		               	loc='center left',
		               	bbox_to_anchor=(-0.25, 0., 1, 1),
		               	bbox_transform=ax[i][j].transAxes,
		               	borderpad=0.1 )
					cbar_ldet = fig.colorbar(CS_det, ax=ax[i][j], cax=axins_det,  orientation='vertical', pad=0.02)
					cbar_ldet.set_label(labels[i]) #, rotation=270)\
					axins_det.yaxis.tick_left()
				else:
					#fancy deterministic cb bottom
					axins_det = inset_axes(ax[i][j],
		            	width="100%",  # width = 5% of parent_bbox width
		               	height="5%",  # height : 50%
		               	loc='lower center',
		               	bbox_to_anchor=(-0.1, -0.15, 1.1, 1),
		               	bbox_transform=ax[i][j].transAxes,
		               	borderpad=0.1 )
					cbar_bdet = fig.colorbar(CS_det, ax=ax[i][j],  cax=axins_det, orientation='horizontal', pad = 0.02)
					cbar_bdet.set_label(labels[i])
		fig.savefig('./images/NG_Deterministic_RealtimeForecasts.png', dpi=500, bbox_inches='tight')
		self.models = self._tempmods

#set up the required directory structure lol
def setup_directories(workdir,working_directory, force_download, cptdir):
	os.chdir(working_directory)

	if force_download and os.path.isdir(workdir):
		if platform.system() == 'Windows':
			print('Windows deleting folders')
			get_ipython().system('del /S /Q {} > op.txt'.format(workdir))
			os.chdir(workdir)
			get_ipython().system('rmdir /S /Q {} > op.txt'.format('scripts'))
			get_ipython().system('rmdir /S /Q {} > op.txt'.format( 'input'))
			get_ipython().system('rmdir /S /Q {} > op.txt'.format('output'))
			get_ipython().system('rmdir /S /Q {} > op.txt'.format( 'images'))
			os.chdir(working_directory)
			get_ipython().system('rmdir /S /Q {} > op.txt'.format(workdir))

		else:
			print('Mac deleting folders')
			get_ipython().system('rm -rf {}'.format(workdir))

	if not os.path.isdir(workdir):
		get_ipython().system('mkdir {}'.format(workdir))

	os.chdir(workdir)

	if not os.path.isdir(workdir + '/scripts'):
		get_ipython().system('mkdir scripts'.format(workdir))

	if not os.path.isdir(workdir + '/images'):
		get_ipython().system('mkdir images')

	if not os.path.isdir(workdir + '/input'):
		get_ipython().system('mkdir input')

	if not os.path.isdir(workdir + '/output'):
		get_ipython().system('mkdir output')

	os.environ["CPT_BIN_DIR"] = cptdir
















def exceedprob(x,dof,lo,sc):
	return t.sf(x, dof, loc=lo, scale=sc)*100

def skilltab(score,wknam,lon1,lat1,lat2,lon2,loni,lone,lati,late,fprefix,mpref,training_season,mon,fday,nwk):
	"""A simple function for ploting probabilities of exceedance and PDFs (for a given threshold)

	PARAMETERS
	----------
		thrs: the threshold, in the units of the predictand
		lon: longitude
		lat: latitude
	"""

	#Read grads binary file size H, W  --it assumes all files have the same size, and that 2AFC exists
	with open('{}/output/'.format(self.cwd_work)+model+'_'+fprefix+'_'+mpref+'_2AFC_'+training_season+'_wk1.ctl', "r") as fp:
		for line in lines_that_contain("XDEF", fp):
			W = int(line.split()[1])
			XD= float(line.split()[4])
	with open('{}/output/'.format(self.cwd_work)+model+'_'+fprefix+'_'+mpref+'_2AFC_'+training_season+'_wk1.ctl', "r") as fp:
		for line in lines_that_contain("YDEF", fp):
			H = int(line.split()[1])
			YD= float(line.split()[4])

	#Find the gridbox:
	lonrange = np.linspace(loni, loni+W*XD,num=W)
	latrange = np.linspace(lati+H*YD, lati, num=H)  #need to reverse the latitudes because of CPT (GrADS YREV option)
	lon_grid, lat_grid = np.meshgrid(lonrange, latrange)
	#first point
	a = abs(lat_grid-lat1)+abs(lon_grid-lon1)
	i1,j1 = np.unravel_index(a.argmin(),a.shape)   #i:latitude   j:longitude
	#second point
	a = abs(lat_grid-lat2)+abs(lon_grid-lon2)
	i2,j2 = np.unravel_index(a.argmin(),a.shape)   #i:latitude   j:longitude

	df = pd.DataFrame(index=wknam[0:nwk])
	for L in range(nwk):
		wk=L+1
		for S in score:
			#Since CPT writes grads files in sequential format, we need to excise the 4 bytes between records (recl)
			f=open('{}/output/'.format(self.cwd_work)+model+'_'+fprefix+'_'+mpref+'_'+str(S)+'_'+training_season+'_wk'+str(wk)+'.dat','rb')
			recl=struct.unpack('i',f.read(4))[0]
			numval=int(recl/np.dtype('float32').itemsize)
			#Now we read the field
			A=np.fromfile(f,dtype='float32',count=numval)
			var = np.transpose(A.reshape((W, H), order='F'))
			var[var==-999.]=np.nan #only sensible values
			df.at[wknam[L], str(S)] = round(np.nanmean(np.nanmean(var[i1:i2,j1:j2], axis=1), axis=0),2)
			df.at[wknam[L], 'max('+str(S)+')']  = round(np.nanmax(var[i1:i2,j1:j2]),2)
			df.at[wknam[L], 'min('+str(S)+')']  = round(np.nanmin(var[i1:i2,j1:j2]),2)
	return df
	f.close()

def pltmapProb(loni,lone,lati,late,fprefix,mpref,training_season, mon, fday, nwk):
	"""A simple function for ploting probabilistic forecasts

	PARAMETERS
	----------
		score: the score
		loni: western longitude
		lone: eastern longitude
		lati: southern latitude
		late: northern latitude
		title: title
	"""
	#Need this score to be defined by the calibration method!!!
	score = 'CCAFCST_P'

	plt.figure(figsize=(15,20))

	for L in range(nwk):
		wk=L+1
		#Read grads binary file size H, W  --it assumes that 2AFC file exists (template for final domain size)
		with open('{}/output/'.format(self.cwd_work)+model+'_'+fprefix+'_'+mpref+'_2AFC_'+training_season+'_wk'+str(wk)+'.ctl', "r") as fp:
			for line in lines_that_contain("XDEF", fp):
				W = int(line.split()[1])
				XD= float(line.split()[4])
		with open('{}/output/'.format(self.cwd_work)+model+'_'+fprefix+'_'+mpref+'_2AFC_'+training_season+'_wk'+str(wk)+'.ctl', "r") as fp:
			for line in lines_that_contain("YDEF", fp):
				H = int(line.split()[1])
				YD= float(line.split()[4])

		#Prepare to read grads binary file  [float32 for Fortran sequential binary files]
		Record = np.dtype(('float32', H*W))

		#Create a feature for States/Admin 1 regions at 1:10m from Natural Earth
		states_provinces = feature.NaturalEarthFeature(
			category='cultural',
#			name='admin_1_states_provinces_shp',
			name='admin_0_countries',
			scale='10m',
			facecolor='none')


		f=open('{}/output/'.format(self.cwd_work)+model+'_'+fprefix+'_'+score+'_'+training_season+'_'+mon+str(fday)+'_wk'+str(wk)+'.dat','rb')

		tit=['Below Normal','Normal','Above Normal']
		for i in range(3):
				#Since CPT writes grads files in sequential format, we need to excise the 4 bytes between records (recl)
				recl=struct.unpack('i',f.read(4))[0]
				numval=int(recl/np.dtype('float32').itemsize)
				#We now read the field for that record (probabilistic files have 3 records: below, normal and above)
				B=np.fromfile(f,dtype='float32',count=numval) #astype('float')
				endrec=struct.unpack('i',f.read(4))[0]
				var = np.flip(np.transpose(B.reshape((W, H), order='F')),0)
				var[var<0]=np.nan #only positive values
				ax2=plt.subplot(nwk, 3, (L*3)+(i+1),projection=ccrs.PlateCarree())
				ax2.set_title("Week "+str(wk)+ ": "+tit[i])
				ax2.add_feature(feature.LAND)
				ax2.add_feature(feature.COASTLINE)
				#ax2.set_ybound(lower=lati, upper=late)
				pl2=ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
					linewidth=1, color='gray', alpha=0.5, linestyle=(0,(2,4)))
				pl2.xlabels_top = False
				pl2.ylabels_left = True
				pl2.ylabels_right = False
				pl2.xformatter = LONGITUDE_FORMATTER
				pl2.yformatter = LATITUDE_FORMATTER
				ax2.add_feature(states_provinces, edgecolor='gray')
				ax2.set_extent([loni,loni+W*XD,lati,lati+H*YD], ccrs.PlateCarree())

				#ax2.set_ybound(lower=lati, upper=late)
				#ax2.set_xbound(lower=loni, upper=lone)
				#ax2.set_adjustable('box')
				#ax2.set_aspect('auto',adjustable='datalim',anchor='C')
				CS=ax2.pcolormesh(np.linspace(loni, loni+W*XD,num=W), np.linspace(lati,lati+H*YD, num=H), var,
				vmin=0,vmax=100,
				cmap=plt.cm.RdYlBu,
				transform=ccrs.PlateCarree())
				#plt.show(block=False)

	plt.subplots_adjust(hspace=0)
	plt.subplots_adjust(bottom=0.15, top=0.9)
	cax = plt.axes([0.2, 0.08, 0.6, 0.04])
	cbar = plt.colorbar(CS,cax=cax, orientation='horizontal', pad=0.01)
	cbar.set_label('Probability (%)') #, rotation=270)
	f.close()

def pltmapff(models,predictand,thrs,ntrain,loni,lone,lati,late,fprefix,mpref,monf,fyr,mons,tgts, obs):
	"""A simple function for ploting probabilistic forecasts in flexible format (for a given threshold)

	PARAMETERS
	----------
		thrs: the threshold, in the units of the predictand
		loni: western longitude
		lone: eastern longitude
		lati: southern latitude
		late: northern latitude
	"""
	#Implement: read degrees of freedom from CPT file
	#Formally, for CCA, dof=ntrain - #CCAmodes -1 ; since ntrain is huge after concat, dof~=ntrain for now
	if obs == 'ENACTS-BD':
		x_offset = 0.6
		y_offset = 0.4
	else:
		x_offset = 0
		y_offset = 0
	dof=ntrain
	nmods=len(models)
	tar=tgts[mons.index(monf)]
	#Read grads binary file size H, W  --it assumes all files have the same size, and that 2AFC exists
	with open('{}/output/'.format(self.cwd_work)+models[0]+'_'+fprefix+predictand+'_'+mpref+'FCST_mu_'+tar+'_'+monf+str(fyr)+'.ctl', "r") as fp:
		for line in lines_that_contain("XDEF", fp):
			W = int(line.split()[1])
			XD= float(line.split()[4])
	with open('{}/output/'.format(self.cwd_work)+models[0]+'_'+fprefix+predictand+'_'+mpref+'FCST_mu_'+tar+'_'+monf+str(fyr)+'.ctl', "r") as fp:
		for line in lines_that_contain("YDEF", fp):
			H = int(line.split()[1])
			YD= float(line.split()[4])

	#plt.figure(figsize=(15,20))
	fig, ax = plt.subplots(figsize=(6,6),sharex=True,sharey=True)
	k=0
	for model in models:
		k=k+1
		#Read grads binary file size H, W  --it assumes all files have the same size, and that 2AFC exists
		f=open('{}/output/'.format(self.cwd_work)+model+'_'+fprefix+predictand+'_'+mpref+'FCST_mu_'+tar+'_'+monf+str(fyr)+'.dat','rb')
		recl=struct.unpack('i',f.read(4))[0]
		numval=int(recl/np.dtype('float32').itemsize)
		#Now we read the field
		A=np.fromfile(f,dtype='float32',count=numval)
		muf = np.transpose(A.reshape((W, H), order='F'))
		muf[muf==-999.]=np.nan #only sensible values

		#Read variance
		f=open('{}/output/'.format(self.cwd_work)+model+'_'+fprefix+predictand+'_'+mpref+'FCST_var_'+tar+'_'+monf+str(fyr)+'.dat','rb')
		recl=struct.unpack('i',f.read(4))[0]
		numval=int(recl/np.dtype('float32').itemsize)
		#Now we read the field
		A=np.fromfile(f,dtype='float32',count=numval)
		vari = np.transpose(A.reshape((W, H), order='F'))
		vari[vari<0.]=np.nan #only positive values

		#Compute scale parameter for the t-Student distribution
		scalef=np.sqrt((dof-2)/dof*vari)

		fprob = exceedprob(thrs,dof,muf,scalef)

		ax = plt.subplot(nmods, 1, k, projection=ccrs.PlateCarree())
		ax.set_extent([loni+x_offset,loni+W*XD+x_offset,lati+y_offset,lati+H*YD+y_offset], ccrs.PlateCarree())

		#Create a feature for States/Admin 1 regions at 1:10m from Natural Earth
		states_provinces = feature.NaturalEarthFeature(
			category='cultural',
#			name='admin_1_states_provinces_shp',
			name='admin_0_countries',
			scale='10m',
			facecolor='none')

		ax.add_feature(feature.LAND)
		ax.add_feature(feature.COASTLINE)

		if k==1:
			ax.set_title('Probability (%) of Exceeding '+str(thrs)+" mm/day")
		#current_cmap = plt.cm.get_cmap('RdYlBu', 10)
		current_cmap = self.make_cmap(10)
		#current_cmap.set_bad('white',1.0)
		current_cmap.set_under('white', 1.0)
		pl=ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
			linewidth=1, color='gray', alpha=0.5, linestyle=(0,(2,4)))
		pl.xlabels_top = False
		pl.ylabels_left = True
		pl.ylabels_right = False
		pl.xformatter = LONGITUDE_FORMATTER
		pl.yformatter = LATITUDE_FORMATTER
		ax.add_feature(states_provinces, edgecolor='gray')
		ax.set_ybound(lower=lati+y_offset, upper=late+y_offset)
		CS=plt.pcolormesh(np.linspace(loni+x_offset, loni+W*XD+x_offset,num=W), np.linspace(lati+H*YD+y_offset, lati+y_offset, num=H), fprob,
			vmin=0,vmax=100,
			cmap=current_cmap,
			transform=ccrs.PlateCarree())
		label = 'Probability (%) of Exceedance'

		#plt.autoscale(enable=True)

		#plt.tight_layout()
		plt.subplots_adjust(hspace=0)
		plt.subplots_adjust(bottom=0.15, top=0.9)
		cbar = fig.colorbar(CS,ax=ax, orientation='vertical', pad=0.05)

		# for i, row in enumerate(ax):
		# 	for j, cell in enumerate(row):
		# 		if i == len(ax) - 1:
		# 			cell.set_xlabel("noise column: {0:d}".format(j + 1))
		# 		if j == 0:
		# 			cell.set_ylabel("noise row: {0:d}".format(i + 1))

		ax.set_ylabel(model, rotation=90)
		cbar.set_label(label) #, rotation=270)
		f.close()

def pltprobff(models,predictand,thrs,ntrain,lon,lat,loni,lone,lati,late,fprefix,mpref,monf,fyr,mons,tgts):
	"""A simple function for ploting probabilities of exceedance and PDFs (for a given threshold)

	PARAMETERS
	----------
		thrs: the threshold, in the units of the predictand
		lon: longitude
		lat: latitude
	"""
	#Implement: read degrees of freedom from CPT file
	#Formally, for CCA, dof=ntrain - #CCAmodes -1 ; since ntrain is huge after concat, dof~=ntrain for now
	dof=ntrain
	tar=tgts[mons.index(monf)]
	#Read grads binary file size H, W  --it assumes all files have the same size, and that 2AFC exists
	with open('{}/output/'.format(self.cwd_work)+models[0]+'_'+fprefix+predictand+'_'+mpref+'FCST_mu_'+tar+'_'+monf+str(fyr)+'.ctl', "r") as fp:
		for line in lines_that_contain("XDEF", fp):
			W = int(line.split()[1])
			XD= float(line.split()[4])
	with open('{}/output/'.format(self.cwd_work)+models[0]+'_'+fprefix+predictand+'_'+mpref+'FCST_mu_'+tar+'_'+monf+str(fyr)+'.ctl', "r") as fp:
		for line in lines_that_contain("YDEF", fp):
			H = int(line.split()[1])
			YD= float(line.split()[4])
	with open('{}/output/'.format(self.cwd_work)+models[0]+'_'+fprefix+predictand+'_'+mpref+'FCST_Obs_'+tar+'_'+monf+str(fyr)+'.ctl', "r") as fp:
		for line in lines_that_contain("TDEF", fp):
			T = int(line.split()[1])
			TD= 1  #not used

	#Find the gridbox:
	lonrange = np.linspace(loni, loni+W*XD,num=W)
	latrange = np.linspace(lati+H*YD, lati, num=H)  #need to reverse the latitudes because of CPT (GrADS YREV option)
	lon_grid, lat_grid = np.meshgrid(lonrange, latrange)
	a = abs(lat_grid-lat)+abs(lon_grid-lon)
	i,j = np.unravel_index(a.argmin(),a.shape)   #i:latitude   j:longitude

	#Now compute stuff and plot
	#plt.figure(figsize=(15,15))

	k=0
	for model in models:
		k=k+1
		#Forecast files--------
		#Read mean
		#Since CPT writes grads files in sequential format, we need to excise the 4 bytes between records (recl)
		f=open('{}/output/'.format(self.cwd_work)+model+'_'+fprefix+predictand+'_'+mpref+'FCST_mu_'+tar+'_'+monf+str(fyr)+'.dat','rb')
		recl=struct.unpack('i',f.read(4))[0]
		numval=int(recl/np.dtype('float32').itemsize)
		#Now we read the field
		A=np.fromfile(f,dtype='float32',count=numval)
		muf = np.transpose(A.reshape((W, H), order='F'))
		muf[muf==-999.]=np.nan #only sensible values
		muf=muf[i,j]

		#Read variance
		f=open('{}/output/'.format(self.cwd_work)+model+'_'+fprefix+predictand+'_'+mpref+'FCST_var_'+tar+'_'+monf+str(fyr)+'.dat','rb')
		recl=struct.unpack('i',f.read(4))[0]
		numval=int(recl/np.dtype('float32').itemsize)
		#Now we read the field
		A=np.fromfile(f,dtype='float32',count=numval)
		varf = np.transpose(A.reshape((W, H), order='F'))
		varf[varf==-999.]=np.nan #only sensible values
		varf=varf[i,j]

		#Obs file--------
		#Compute obs mean and variance.
		#
		muc0=np.empty([T,H,W])  #define array for later use
		#Since CPT writes grads files in sequential format, we need to excise the 4 bytes between records (recl)
		f=open('{}/output/'.format(self.cwd_work)+models[0]+'_'+fprefix+predictand+'_'+mpref+'FCST_Obs_'+tar+'_'+monf+str(fyr)+'.dat','rb')
		#cycle for all time steps  (same approach to read GrADS files as before, but now read T times)
		for it in range(T):
			#Now we read the field
			recl=struct.unpack('i',f.read(4))[0]
			numval=int(recl/np.dtype('float32').itemsize) #this if for each time stamp
			A0=np.fromfile(f,dtype='float32',count=numval)
			endrec=struct.unpack('i',f.read(4))[0]  #needed as Fortran sequential repeats the header at the end of the record!!!
			muc0[it,:,:]= np.transpose(A0.reshape((W, H), order='F'))

		muc0[muc0==-999.]=np.nan #identify NaNs
		muc=np.nanmean(muc0, axis=0)  #axis 0 is T
		#Compute obs variance
		varc=np.nanvar(muc0, axis=0)  #axis 0 is T
		#Select gridbox values
		muc=muc[i,j]
		#print(muc)   #Test it's actually zero
		varc=varc[i,j]

		#Compute scale parameter for the t-Student distribution
		scalef=np.sqrt(dof*varf)   #due to transformation from Gamma
		scalec=np.sqrt((dof-2)/dof*varc)

		x = np.linspace(min(t.ppf(0.00001, dof, loc=muf, scale=scalef),t.ppf(0.00001, dof, loc=muc, scale=scalec)),max(t.ppf(0.9999, dof, loc=muf, scale=scalef),t.ppf(0.9999, dof, loc=muc, scale=scalec)), 100)

		style = dict(size=10, color='black')

		#cprob = special.erfc((x-muc)/scalec)
		cprob = exceedprob(thrs,dof,muc,scalec)
		fprob = exceedprob(thrs,dof,muf,scalef)
		cprobth = round(t.sf(thrs, dof, loc=muc, scale=scalec)*100,2)
		fprobth = round(t.sf(thrs, dof, loc=muf, scale=scalef)*100,2)
		cpdf=t.pdf(x, dof, loc=muc, scale=scalec)*100
		fpdf=t.pdf(x, dof, loc=muf, scale=scalef)*100
		oddsrc =(fprobth/cprobth)

		fig, ax = plt.subplots(1, 2,figsize=(12,4))
		#font = {'family' : 'Palatino',
		#        'size'   : 16}
		#plt.rc('font', **font)
		#plt.rc('text', usetex=True)
		#plt.rc('font', family='serif')

		#plt.subplot(1, 2, 1)
		ax[0].plot(x, t.sf(x, dof, loc=muc, scale=scalec)*100,'b-', lw=5, alpha=0.6, label='clim')
		ax[0].plot(x, t.sf(x, dof, loc=muf, scale=scalef)*100,'r-', lw=5, alpha=0.6, label='fcst')
		ax[0].axvline(x=thrs, color='k', linestyle=(0,(2,4)))
		ax[0].plot(thrs, fprobth,'ok')
		ax[0].plot(thrs, cprobth,'ok')
		ax[0].text(thrs+0.05, cprobth, str(cprobth)+'%', **style)
		ax[0].text(thrs+0.05, fprobth, str(fprobth)+'%', **style)
		#ax[0].text(0.1, 10, r'$\frac{P(fcst)}{P(clim)}=$'+str(round(oddsrc,1)), **style)
		ax[0].text(min(t.ppf(0.0001, dof, loc=muf, scale=scalef),t.ppf(0.0001, dof, loc=muc, scale=scalec)), -20, 'P(fcst)/P(clim)='+str(round(oddsrc,1)), **style)
		ax[0].legend(loc='best', frameon=False)
		# Add title and axis names
		ax[0].set_title('Probabilities of Exceedance')
		ax[0].set_xlabel('Rainfall')
		ax[0].set_ylabel('Probability (%)')
		# Limits for the Y axis
		#ax[0].set_xlim(left=min(t.ppf(0.001, dof, loc=muf, scale=scalef),t.ppf(0.001, dof, loc=muc, scale=scalec)),right=max(t.ppf(0.99, dof, loc=muf, scale=scalef),t.ppf(0.99, dof, loc=muc, scale=scalec)))
		#ax[0].set_ylim(left=0, right=1)
		#ax[1].subplot(1, 2, 2)
		ax[1].plot(x, cpdf,'b-', lw=5, alpha=0.6, label='clim')
		ax[1].plot(x, fpdf,'r-', lw=5, alpha=0.6, label='fcst')
		ax[1].axvline(x=thrs, color='k', linestyle=(0,(2,4)))
		ax[1].legend(loc='best', frameon=False)
		# Add title and axis names
		ax[1].set_title('Probability Density Functions')
		ax[1].set_xlabel('Rainfall')
		ax[1].set_ylabel('')
		#ax[1].set_ylim(left=0, right=1)

		# Limits for the Y axis
		#print()
		ax[1].set_xlim(left=min(t.ppf(0.001, dof, loc=muf, scale=scalef),t.ppf(0.001, dof, loc=muc, scale=scalec)),right=max(t.ppf(0.99, dof, loc=muf, scale=scalef),t.ppf(0.99, dof, loc=muc, scale=scalec)))


	plt.subplots_adjust(hspace=0)
	plt.subplots_adjust(bottom=0.15, top=0.9)
	#cax = plt.axes([0.2, 0.08, 0.6, 0.04])
	#cbar = plt.colorbar(CS,cax=cax, orientation='horizontal')
	#cbar.set_label(label) #, rotation=270)
	f.close()
