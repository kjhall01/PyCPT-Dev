import torch as t
from torch import nn
import torch.nn.functional as F
from pycpt_functions_seasonal import *
from scipy.interpolate import griddata
import sys

#plotting libraries
import cartopy.crs as ccrs
from cartopy import feature
import cartopy.mpl.ticker as cticker
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

class ClimateTensor():
	"""Holds model data & obs data for one year"""
	def __init__(self):
		self.t = [] #stores a list of years for which there is data for each model
		self.lat = [] #stores a list of latitudes for which there is data for each model
		self.lon = [] #stores a list of longitudes for which there is data for each model
		self.data = [] #stores an array of data of shape (nyears, W, H) for each model
		self.N = 0
		self.model_name = ''

	def add_observations(self, data_tuple, yr_ndx):
		""" Adds observation data to the ClimateTensor, where
			data_tuple = (	[years list],
							[lats list],
							[lons list],
							[array of shape ( W, H)] )"""
		self.obs_t = data_tuple[0]
		self.obs_lat = data_tuple[1]
		self.obs_lon = data_tuple[2]
		self.obs_data = data_tuple[3][yr_ndx]
		self.obs_nla = self.obs_lat[0]
		self.obs_sla = self.obs_lat[-1]
		self.obs_elo = self.obs_lon[-1]
		self.obs_wlo = self.obs_lon[0]
		self.model_names.append('Observations')
		self.x_offset, self.y_offset = 0,0

	def plot_obs(self):
		nsea, nmods = 1, self.N+1
		fig, ax = plt.subplots(nrows=self.N + 1, ncols=1, figsize=(5*nsea, 5*nmods),sharex=False,sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})

		if nsea==1 and nmods == 1:
			ax = [[ax]]
		elif nsea == 1:
			ax = [[ax[q]] for q in range(nmods)]
		elif nmods == 1:
			ax = [ax]
		for year in self.good_years:
			vmin,vmax = 0.0,np.nanmax(self.obs_data[self.obs_t.index(year)])
			norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
			for i in range(nmods):
				for j in range(nsea):
					states_provinces = feature.NaturalEarthFeature(
						category='cultural',
						name='admin_0_countries',
						scale='10m',
						facecolor='none')

					ax[i][j].set_extent([self.obs_wlo+self.x_offset,self.obs_elo+self.x_offset,self.obs_sla+self.y_offset,self.obs_nla+self.y_offset], ccrs.PlateCarree())

					ax[i][j].add_feature(feature.LAND)
					pl=ax[i][j].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
						  linewidth=1, color='gray', alpha=0.5, linestyle=(0,(2,4)))
					pl.xlabels_top = False
					pl.ylabels_left = True
					pl.ylabels_right = False
					pl.xlabels_bottom = True
					pl.xformatter = LONGITUDE_FORMATTER
					pl.yformatter = LATITUDE_FORMATTER

					ax[i][j].add_feature(states_provinces, edgecolor='black')
					ax[i][j].set_ybound(lower=self.obs_sla, upper=self.obs_nla)
					if j == 0:
						ax[i][j].text(-0.42, 0.5, self.model_names[i-1],rotation='vertical', verticalalignment='center', horizontalalignment='center', transform=ax[i][j].transAxes)
					if i == 0:
						ax[i][j].set_title("self.tgts[j]")
					if i == 0:
						var = self.obs_data[self.obs_t.index(year)]
					else:
						var = self.new_data[i-1][self.t[i-1].index(year)]


					if i == 0:
						CS=ax[i][j].pcolormesh(np.linspace(self.obs_lon[0], self.obs_lon[-1], num=len(self.obs_lon)+1), np.linspace(self.obs_lat[0],self.obs_lat[-1], num=len(self.obs_lat)+1 ), var,
						vmin=vmin, vmax=vmax, #vmax=np.max(var),
						norm=norm,
						cmap=plt.get_cmap('BrBG'),
						transform=ccrs.PlateCarree())
					else:
						CS=ax[i][j].pcolormesh(np.linspace(self.obs_lon[0], self.obs_lon[-1], num=int(self.newx_dims[i-1]+1)), np.linspace(self.obs_lat[0],self.obs_lat[-1], num=int(self.newy_dims[i-1]+1) ), var,
						vmin=vmin, vmax=vmax, #vmax=np.max(var),
						norm=norm,
						cmap=plt.get_cmap('BrBG'),
						transform=ccrs.PlateCarree())
					axins = inset_axes(ax[i][j],
						   width="5%",  # width = 5% of parent_bbox width
						   height="100%",  # height : 50%
						   loc='center right',
						   bbox_to_anchor=(0., 0., 1.15, 1),
						   bbox_transform=ax[i][j].transAxes,
						   borderpad=0.1,
						   )

					cbar = fig.colorbar(CS, ax=ax[i][j], norm=norm, cax=axins, orientation='vertical', pad=0.02)

			fig.savefig('testing{}.png'.format(year), dpi=500, bbox_inches='tight')
			plt.show()
			plt.close()

	def add_model(self, data_tuple, yr_ndx):
		""" Adds a model's data to the ClimateTensor, where
			data_tuple = (	[years list],
							[lats list],
							[lons list],
							[array of shape (nyears, W, H)]
							name of model )"""
		self.t = data_tuple[0]
		self.lat = data_tuple[1]
		self.lon = data_tuple[2]
		self.data = data_tuple[3][yr_ndx]

		self.N += 1
		self.model_name = data_tuple[4]

	def rectify(self):
		""" Cuts out all data across time & space where
			any of the models is missing a datapoint"""
		self.new_data = [[[] for j in range(len(self.t[i]))] for i in range(self.N)]
		self.nla, self.sla = 90, -90
		self.wlo, self.elo = -180, 180

		for i in range(self.N):
			if self.lat[i][0] < self.nla: #we must find the southernmost northernmost latitude,
				self.nla = self.lat[i][0]
			if self.lat[i][-1] > self.sla: #the northernmost southernmost latitude,
				self.sla = self.lat[i][-1]
			if self.lon[i][-1] < self.elo: #the westernmost easternmost longitude,
				self.elo = self.lon[i][-1]
			if self.lon[i][0] > self.wlo: #and the easternmost westernmost longitude
				self.wlo = self.lon[i][0] #in order to excise any data where there are not data for all Models and obs
		#forget this for now - we'll just assume Predictand is within Predictor and use that

		#afirst, find good years
		for year in self.obs_t:
			flag = 1
			for i in range(self.N):
				if year not in self.t[i]:
					flag = 0
			if flag == 1:
				self.good_years.append(year)

		#first, interpolate the model data before we cut off a bunch of it
		print(len(self.obs_lon), len(self.obs_lat) )
		print(self.obs_lon[0], self.obs_lat[0] )
		print(self.obs_lon[-1], self.obs_lat[-1] )



		print(len(self.lon[i]), len(self.lat[i]) )
		print(self.lon[i][0], self.lat[i][0] )
		print(self.lon[i][-1], self.lat[i][-1] )

		sys.exit()
		for i in range(self.N):
			#Increasing resolution of model by 16x
			xx, yy = np.mgrid[self.lat[i][-1]:self.lat[i][0]:complex(0,4*len(self.lat[i])), self.lon[i][0]:self.lat[i][-1]:complex(0,4*len(self.lon[i]))]

			for year in self.t[i]:
				points, vals = [], []
				for j in range(len(self.lat[i])):
					for k in range(len(self.lon[i])):
						points.append([self.lat[i][j], self.lon[i][k]] )
						vals.append(self.data[i][self.t[i].index(year)][j][k])

				self.new_data[i][self.t[i].index(year)] = griddata(points, vals, (xx, yy), method='nearest')
				newnansx, newnansy = [], []
				nansx, nansy = np.where(np.isnan(self.obs_data[self.obs_t.index(year)]))
				print(nansx, nansy)
				nansx, nansy = nansx / len(self.obs_lon), nansy / len(self.obs_lat)
				print(len(self.obs_lon), len(self.obs_lat) )
				print(nansx, nansy)

				nansx, nansy = nansx*self.newx_dims[i], nansy*self.newy_dims[i]
				print(nansx, nansy)

				newnans = np.asarray([tuple(np.round(nansx)), tuple(np.round(nansy))])
				print(newnans)
				self.new_data[i][self.t[i].index(year)][newnans.astype(int)] = np.nan


class DataLoader():
	def __init__(self, pycpt):
		self.pycpt = pycpt
		self.models = pycpt.models
		self.tgts = pycpt.tgts
		self.mons = pycpt.mons
		self.model_ndx = -1
		self.pycpt.setup_directories()

	def read_data(self, f):
		"""reads an input data file and returns its data_tuple"""
		self.model_ndx += 1
		lats, all_vals, vals = [], [], []
		flag = 0
		for line in f:
			if line[0:4] == 'cpt:':
				if line[4] == 'T':
					line = line.strip()
					line = line.split('-')[:-1]
					years = [int(i[len(i)-4:len(i)]) for i in line]
				if flag == 2:
					vals = np.asarray(vals, dtype=float)
					vals[vals == -999.0] = np.nan
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
		vals[vals == -999.0] = np.nan
		all_vals.append(vals)
		all_vals = all_vals
		f.close()
		if self.model_ndx < len(self.models):
			name = self.models[self.model_ndx]
		else:
			name = "Observations"
		return (years, lats, longs, all_vals, name)

	def read_all_data(self):
		"""Returns a 3d list of shape (nmodels, ntgts, years) ClimateTensor for each [model,tgt,year]"""
		climatetensors = [[[] for j in range(len(self.tgts))] for i in range(len(self.models))] #create climate tensors object

		for i in range(len(self.models)): #add a climate tensor for each [model,tgt,year]
			for j in range(len(self.tgts)):
				f = open("./input/{}_PRCP_{}_ini{}.tsv".format(self.models[i], self.tgts[j], self.mons[j]), 'r')
				data_tuple = self.read_data(f)

				for year in range(len(data_tuple[0])):
					climatetensors[i][j].append(ClimateTensor())
					climatetensors[i][j][year].add_model(data_tuple, year)

		for j in range(len(self.tgts)):
			f = open('./input/obs_PRCP_{}.tsv'.format(self.tgts[j]), 'r')
			data_tuple = self.read_data(f)
			for i in range(len(self.models)):
				for k in range(len(climatetensors[i][j])):
					# data_tuple[0].index(climatetensors[i][j].t[k]) represents the index of the model year within the observation years
					climatetensors[i][j][k].add_observations(data_tuple, data_tuple[0].index(climatetensors[i][j].t[k])) #assume that there is observation data for every year of each model

		return climatetensors

	def fetch_iri_data(self):
		for model in range(len(self.models)):
			for tgt in range(len(self.tgts)):
				self.pycpt.setupParams(tgt)
				self.pycpt.prepFiles(tgt, model)
				#we do not write CPT scripts or run CPT here, just download input data
