import os
import time
import requests 
import concurrent.futures
from io import BytesIO
import numpy as np
from astropy import units as u
from astroquery.esa.euclid import Euclid
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp 
from astropy.visualization import  PowerStretch, SqrtStretch, LogStretch
from astropy.visualization import AsinhStretch, LinearStretch, AsymmetricPercentileInterval
from astropy.coordinates import SkyCoord 

#from sparcl.client import SparclClient  for DESI stuff
#from dl import queryClient as qc

#from scipy.ndimage import zoom



class euclid_cutouts_class:
    
    def __init__(self, ra, dec, 
                 euclid_filters = ["VIS", "NIR_Y", "NIR_J", "NIR_H"],
                 save_dir = "data/cutouts"):
        self.coordinates = SkyCoord(ra, dec, unit = "degree", frame = "icrs")   
        self.euclid_filters = euclid_filters
        self.save_dir = save_dir
        self.has_stacked_cutout = False
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        return None
    
    def get_cone(self, initial_radius = 0.3*u.degree, async_job=True, verbose = True):
        tic = time.perf_counter()
        job = Euclid.cone_search(self.coordinates, initial_radius, table_name="sedm.mosaic_product", ra_column_name="ra",
                                      dec_column_name="dec", columns="*", async_job= async_job)
        self.cone_results = job.get_results()
        toc = time.perf_counter()
        if verbose:
            print(f"Cone search required {toc-tic} seconds")
        return None
    
    @staticmethod
    def get_info_cutout(cone_results, filter_name):
        line = cone_results[cone_results["filter_name"]==filter_name][0]
        file_path = os.path.join(line["file_path"], line["file_name"])
        instrument = line["instrument_name"]
        obs_id = line["tile_index"]
        return file_path, instrument, obs_id
    
    def get_band_cutout(self, band):
        file_path, instrument, obs_id = self.get_info_cutout(self.cone_results, band)
        output_file = os.path.join(self.save_dir, f"{obs_id}_{band}.fits")  #This is not unique, cutous might be overwritten
        return Euclid.get_cutout(file_path=file_path, instrument=instrument, id=obs_id, 
                                coordinate=self.coordinates, radius = self.cutout_radius, output_file=output_file)[0]


    def get_cutouts(self, radius, verbose = False):
        
        self.cutout_radius = radius*u.arcsec
        self.cutouts_paths = {}
        
        tic = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = { 
                 executor.submit(self.get_band_cutout, band) : band
                 for band in self.euclid_filters
            }
        for future in concurrent.futures.as_completed(futures):
            band = futures[future]
            save_path = future.result()
            self.cutouts_paths[band] = save_path  
        
        toc = time.perf_counter()
        if verbose:
                print(f"Retrieving all cutouts requiered {toc-tic} seconds")
        return None 
    

    def read_cutouts(self):
        self.data = {}
        self.wcs = {}
        self.arcsec_per_pix ={}
        for band in self.cutouts_paths.keys():
            with fits.open(self.cutouts_paths[band]) as hdul:
                self.data |= {band : hdul[0].data}         #probably doesn't work in 3.8--> use .update
                self.wcs  |= {band : WCS(hdul[0].header)}  #Required for stacking images}  
                self.arcsec_per_pix  |= {band : np.abs(hdul[0].header["CD1_1"]*3600)}  
        return None 
    
    
    def reproject_cutouts(self, reference = "VIS"):
        """Aligns and resizes VIS and NISP images so that can be stacked
        Reference can be either the name of the filter or its index"""

        if isinstance(reference, int):
            reference = self.euclid_filters[reference]
        
        ref_wcs = self.wcs[reference]  
        ref_shape = self.data[reference].shape
        self.arcsec_per_pix |= {"stacked" : self.arcsec_per_pix[reference]}
       
        
        self.reprojected_data = {}
        for band in self.euclid_filters:
            reprojected, _ = reproject_interp((self.data[band], self.wcs[band]), ref_wcs, shape_out=ref_shape)
            self.reprojected_data |= {band : reprojected}
        
        return None
    
    @staticmethod
    def transform_image(image, 
                        stretch = LinearStretch(slope =1), 
                        interval = AsymmetricPercentileInterval(lower_percentile = 1, upper_percentile=99)):
        transform = stretch + interval 
        return transform(image)
    

    def stack_cutouts(self, r_img = "NIR_H", g_img = "NIR_Y", b_img = "VIS",
                        stretch = LinearStretch(slope =1), 
                        interval = AsymmetricPercentileInterval(lower_percentile = 1, upper_percentile=99) ):
        
        stretch_map = {"Linear": lambda: LinearStretch(slope=1),
                      "Sqrt": lambda: SqrtStretch(),
                      "Log" : lambda: LogStretch(),
                      "Asinh": lambda: AsinhStretch(),
                      "PowerLaw": lambda: PowerStretch(a=2)}


        if isinstance(stretch, str):
            stretch = stretch_map.get(stretch)()

        norm_images = [self.transform_image(self.reprojected_data[band], stretch = stretch, interval = interval)
                   for band in [r_img, g_img, b_img]]
        self.reprojected_data |= {"stacked" : np.dstack(norm_images)}
        return None
        

    @staticmethod
    def zoom_image(image, scale, same_shape = True, zooming_order = 1):
        if scale >= 1:
            return image
        if scale <= 0:
            return None
        height, width = image.shape[:2]
        new_height, new_width = int(height * scale), int(width * scale)
        
        top = (height - new_height) // 2
        left = (width - new_width) // 2
        bottom = top + new_height
        right = left + new_width
        
        cropped_image = image[top:bottom, left:right]
        if not same_shape:
            return cropped_image
        zoom_factors = (height / new_height, width / new_width) if image.ndim == 2 else (height / new_height, width / new_width, 1)
        #return zoom(cropped_image, zoom_factors, order=zooming_order)
    
    def get_scaled_cutout(self, scale, same_shape = True, zooming_order = 1, verbose = False, filtro = "stacked"):
        """here scale refers to the current scaled image"""

        tic = time.perf_counter()
        if not hasattr(self,"scaled_cutouts"):
            self.scaled_cutouts = self.data.copy()
            self.scales = {band : 1 for band in self.scaled_cutouts}   #currently same scale for all bands
            self.scaled_arcsec_per_pix = self.arcsec_per_pix.copy()
        try:
            self.scales[filtro] = self.scales[filtro]*scale
            self.scaled_cutouts[filtro] = self.zoom_image(self.data[filtro], self.scales[filtro], same_shape = same_shape,
                                                         zooming_order=zooming_order)
            if same_shape:
                    self.scaled_arcsec_per_pix[filtro] = self.scaled_arcsec_per_pix[filtro] * scale
        except KeyError:    
            for band, image in self.data.items():
                self.scales[band] = self.scales[band]*scale
                self.scaled_cutouts[band] = self.zoom_image(image, self.scales[band], same_shape = same_shape,
                                                       zooming_order=zooming_order)
                if same_shape:
                    self.scaled_arcsec_per_pix[band] = self.scaled_arcsec_per_pix[band] * scale

        toc = time.perf_counter()
        
        if verbose:
            print(f"Scaling cutouts required {toc-tic} seconds")
        return None
    
    
##### Not working with AstronomicAL environment 
class DESI_spectra_class:
    #actually queries also BOSS and SDSS DR16
    def __init__(self, ra, dec, distance_radius = 1):
        self.ra = ra
        self.dec = dec
        self.radius = distance_radius/3600 #arcsec--> degreee
        self.client = SparclClient()
    
    def query_main_table(self):
        """Astro Data Lab does not accept Circle or Point functions,
          referred to https://datalab.noirlab.edu/help/index.php?qa=366&qa_1=dont-adql-functions-like-point-circle-work-query-interface
          for cone search.
          LIMIT = 100 just for avoiding strange results"""

        query = f"""SELECT sparcl_id, specid, ra, dec, redshift, spectype, 
                 data_release, redshift_err, specprimary, redshift_warning,
                 3600.0 * q3c_dist(ra, dec, {self.ra}, {self.dec}) AS separation_arcsec
                 FROM sparcl.main
                 WHERE Q3C_RADIAL_QUERY(ra, dec, {self.ra},{self.dec}, {self.radius})
                 ORDER BY separation_arcsec ASC
                 LIMIT 100"""
        self.results = qc.query(sql=query, fmt='pandas')
        self.available_spectra = len(self.results)
        return None
    
    def set_ra_dec(self, ra, dec):
        """To update class without recalling the SparcClient"""
        self.ra = ra
        self.dec = dec
        return None
    
    def query_spectra(self, verbose = False):
        include = ['sparcl_id', 'specid', 'data_release', 'redshift', 'flux',
                             'wavelength', 'model', 'spectype']
        if self.available_spectra >= 1:
            sparcl_id, dataset = self.get_info_spectra(self.results)
            tic = time.perf_counter()
            self.spectrum_query = self.client.retrieve(uuid_list = sparcl_id, dataset_list = dataset,
                        include = include)
            self.spectrum = self.spectrum_query.records[0]
            toc = time.perf_counter()
            if verbose:
                print(f"Retrieving spectrum required {toc-tic} seconds")
        else:
            print("No available spectra")
        return None

    @staticmethod
    def get_info_spectra(table, max_sep = 1, datasets = ["DESI-DR1", "DESI-EDR", "BOSS-DR16", "SDSS-DR16"]):
        if isinstance(datasets, str): 
            datasets = [datasets]
        for dataset in datasets:
            logic = (table["data_release"]==dataset) & (table["separation_arcsec"]<= max_sep) & (table["specprimary"]<= 1)
            selected = table[logic].reset_index(drop = True)
            if len(selected) >=1:
                break
        return [selected.loc[0, "sparcl_id"]], [selected.loc[0, "data_release"]]

     




































class radio_cutouts_class:

    def __init__(self, ra, dec):
        self.ra = float(ra)
        self.dec = float(dec)
        self.get_url()
        return None
    
    def get_url(self):
        h = np.floor(self.ra / 15.0)
        d = self.ra - h * 15
        m = np.floor(d / 0.25)
        d = d - m * 0.25
        s = d / (0.25 / 60.0)
        s = np.round(s)
        ra_conv = f"{h} {m} {s}"
        sign = 1
        if self.dec < 0:
            sign = -1
        g = np.abs(self.dec)
        d = np.floor(g) * sign
        g = g - np.floor(g)
        m = np.floor(g * 60.0)
        g = g - m / 60.0
        s = g * 3600.0

        s = np.round(s)
        dec_conv = f"{d} {m} {s}"

        url1 = "https://third.ucllnl.org/cgi-bin/firstimage?RA="
        url2 = "&Equinox=J2000&ImageSize=2.5&MaxInt=200&GIF=1"
        self.url = f"{url1}{ra_conv} {dec_conv}{url2}"
        return None

    def get_image(self):
        r = requests.get(self.url)
        self.image = BytesIO(r.content).seek(0)
        return None
    
    def reset(self, ra, dec):
        self.ra = float(ra)
        self.dec = float(dec)
        self.get_url()
        return None



class sdss_cutouts_class:
        
        def __init__(self, ra, dec, scale = 0.2):
            self.ra = ra
            self.dec = dec 
            self.scale = scale
            self.get_url()

        def get_url(self):
            url = "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra="

            self.url = f"{url}{self.ra}&dec={self.dec}&opt=G&scale={self.scale}"
            return None

        def update_scale(self, scale):
            url = "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra="
            self.scale = scale
            self.url = f"{url}{self.ra}&dec={self.dec}&opt=G&scale={self.scale}"
            return None
        
        def reset(self, ra, dec, scale = 0.2):
            self.ra = ra
            self.dec = dec 
            self.scale = scale
            self.get_url()
            return None
         
