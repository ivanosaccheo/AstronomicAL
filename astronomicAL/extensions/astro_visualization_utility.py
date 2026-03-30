from dataclasses import dataclass, field
import copy
import numpy as np 
from astropy.visualization import  PowerStretch, SqrtStretch, LogStretch
from astropy.visualization import AsinhStretch, LinearStretch, AsymmetricPercentileInterval, MinMaxInterval, PercentileInterval
from astropy.wcs import WCS
from reproject import reproject_interp
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs.utils import proj_plane_pixel_scales



@dataclass
class BandState:
    raw: np.ndarray
    wcs: WCS | None = None
    stretched: np.ndarray = None
    clipped: np.ndarray = None
    scaled: np.ndarray = None
    
    _resampled_raw: np.ndarray = None
    _resampled_wcs: WCS | None = None

    config: dict = field(default_factory=dict)
    
    def invalidate_from(self, level: str):

        if level == "band":
            self.stretched = None
            self.clipped = None
            self.scaled = None
            self._resampled_raw = None
            self._resampled_wcs = None

        elif level == "stretch":
            self.clipped = None
            self.scaled = None

        elif level == "clip":
            self.scaled = None

        elif level == "resample":
            self.stretched = None
            self.clipped = None
            self.scaled = None

        elif level == "scale":
            pass



class ImageVisaulizationClass:
    
    _STRETCH_MAP = {
        "Linear": lambda scale: LinearStretch(slope=scale if scale is not None else 1),
        "Sqrt": lambda scale: SqrtStretch(),
        "Log": lambda scale: LogStretch(a=scale if scale is not None else 1000),
        "Asinh": lambda scale: AsinhStretch(a=scale if scale is not None else 0.1),
        "PowerLaw": lambda scale: PowerStretch(a=scale if scale is not None else 2),
    }

    _INTERVAL_MAP = {
        "Asymmetric": lambda perc: AsymmetricPercentileInterval(
            lower_percentile=perc[0],
            upper_percentile=perc[1],),
        "MinMax": lambda _: MinMaxInterval(),
        "Percentile": lambda perc: PercentileInterval(perc),
    }

    _INTERVAL_DEFAULTS = {
        "Asymmetric": (0.1, 100),
        "MinMax": None,
        "Percentile": 99.5,
    }


    def __init__(self, 
                images, 
                wcs = None, 
                band_names = None,
                color_image = False,
                color_bands = None,
                color_name = None,
                target_wcs = None):
        """ images : 2D np.ndarray or list of 2D np.ndarrays. The image(s) which will be transformed and rendered
            wcs : WCS file 
            band_names : string or list of strings. Band identifiers
            color_image : bool. True if a color image must be created
            color_bands : list of strings. names of the images used as R,G and B 
        """
        if isinstance(images, np.ndarray):
            if images.ndim != 2:
                raise ValueError("Single image must be a 2D numpy array")
            images = [images]

        elif isinstance(images, list):
            if not all(isinstance(img, np.ndarray) and img.ndim == 2 for img in images):
                raise ValueError("All images must be 2D numpy arrays")
        else:
            raise TypeError("'images' must be a 2D numpy array or list of 2D numpy arrays")
        

        if wcs is not None:
            if isinstance(wcs, WCS):
                wcs = [wcs]
            elif isinstance(wcs, (list, tuple)):
                if not all(isinstance(x, WCS) for x in wcs):
                    raise TypeError("All elements in wcs list must be WCS objects")
            else:
                raise TypeError("wcs must be a WCS object or a list/tuple of WCS objects")
            if len(wcs) != len(images):
                raise ValueError("Length of wcs must match number of images")

        if band_names is None:
            band_names = [str(i) for i in range(len(images))]

        elif isinstance(band_names, str):
            band_names = [band_names]
            
        if len(band_names) != len(images):
            raise ValueError("'images' and 'image_names' must have the same length")
            
        self.data = dict(zip(band_names, images))
        self.target_wcs = target_wcs
        self._bands = {}

        for idx, (band_name, img) in enumerate(self.data.items()):
        #This dataclass stores the image properties  by band so that i do not need to apply all the transformation every time
            band_wcs = None
            if wcs is not None:
                band_wcs = wcs[idx]
            self._bands[band_name] = BandState(
                raw=img,
                wcs=band_wcs,
            )

        self.has_color_img = color_image
        if self.has_color_img:
            if len(self.data) < 3:
                raise ValueError(
                    "Color image can be created only if at least three images are given"
                )

            if color_bands is None:
                raise ValueError(
                    "'color_bands' must be provided when color_image=True"
                )

            if len(color_bands) != 3:
                raise ValueError(
                    "'color_bands' must contain exactly three bands (R, G, B)"
                )

            if not all(band in self.data for band in color_bands):
                raise KeyError(
                    "Bands used for the color image must be among band_names"
                )

            self.color_bands = color_bands
        else:
            self.color_bands = None
        self.color_name =  color_name if color_name is not None else "ColorImage"


    
    def _get_aligned_image(self, band_state: BandState):

        if self.target_wcs is None:
            return band_state.raw, band_state.wcs
        # Already computed 
        if (
            band_state._resampled_raw is not None
            and band_state._resampled_wcs == self.target_wcs
        ):
            return band_state._resampled_raw, band_state._resampled_wcs

        # Image has the same WCS
        if band_state.wcs == self.target_wcs:
            band_state._resampled_raw = band_state.raw
            band_state._resampled_wcs = band_state.wcs
            return band_state.raw, band_state.wcs

        #Compute reprojection
        resampled, _ = reproject_interp(
            (band_state.raw, band_state.wcs),
            self.target_wcs,
            shape_out=self.target_wcs.array_shape,
        )

        band_state._resampled_raw = resampled
        band_state._resampled_wcs = self.target_wcs

        return resampled, self.target_wcs
   
    def _get_stretch(self, stretch_type, stretch_scale=None):
        if stretch_type not in self._STRETCH_MAP:
            raise ValueError(f"Unknown stretch type: {stretch_type}")
        return self._STRETCH_MAP[stretch_type](stretch_scale)
    

    def _get_interval(self, interval_type, interval_param=None):
        if interval_type not in self._INTERVAL_MAP:
             raise ValueError(f"Unknown interval type: {interval_type}")

        if interval_param is None:
            interval_param = self._INTERVAL_DEFAULTS.get(interval_type)
        return self._INTERVAL_MAP[interval_type](interval_param)
 
    @staticmethod
    def _stretch_image(image, stretch, stretch_interval):
        if stretch is None:
            stretch = LinearStretch()
        if stretch_interval is None:
            transform = stretch
        else:
            transform = stretch + stretch_interval 
        return transform(image)
    
    @staticmethod
    def _clip_image(image, low, high, image_min = None, image_max = None):
        
        if (low == 0) and (high == 1):
            return image
        if image_min is None:
            image_min = np.nanmin(image)
        if image_max is None:
            image_max = np.nanmax(image)

        image_range = image_max - image_min
        absolute_low = image_min + low * image_range
        absolute_high = image_min + high * image_range

        return np.clip(image, absolute_low, absolute_high)
    

    @staticmethod
    def _scale_image(image, scale_method= "minmax", image_min = None, image_max = None):
        if scale_method.lower() =="minmax":
            if image_min is None:
                image_min = np.nanmin(image)
            if image_max is None:
                image_max = np.nanmax(image)
            scaled_image = (image-image_min)/(image_max-image_min)
            scaled_image = np.clip(scaled_image, 0,1)
    
        elif scale_method.lower() == "expand":
            mid_value = np.nanmedian(image)
            sigma = np.nanstd(image)
            scaled_image = np.where(image>mid_value+(1*sigma), image * 2, image / 2)   
        else:
            raise ValueError(f"Unknown scale_method: {scale_method}") 
        return scaled_image
    
    @staticmethod
    def _unzip(value):
        if np.iterable(value) and not isinstance(value, (str, bytes)):
            return value
        return [value, value, value]
    
    def get_current_plot_config(self):
        if getattr(self, "_plot_config", None) is None:
            return None  # clearer than {}
        return copy.deepcopy(self._plot_config)

    def get_plot_data(self, 
                      band, 
                      stretch = "Linear", 
                      stretch_scale = None,
                      stretch_interval = "Asymmetric",
                      low_clip = 0,
                      high_clip = 1,
                      gamma_color = (1,1,1),
                      scale_method = "MinMax",
                      _internal = False,
                      ):
        if not _internal:
            self._plot_config = {
                "band": band,
                "stretch": stretch,
                "stretch_scale": stretch_scale,
                "stretch_interval": stretch_interval,
                "low_clip": low_clip,
                "high_clip": high_clip,
                "gamma_color": gamma_color,
                "scale_method": scale_method}

        if band == self.color_name:
            if not self.has_color_img:
                raise ValueError("Color mode not enabled")

            low_clips = self._unzip(low_clip)
            high_clips = self._unzip(high_clip)
            gamma = self._unzip(gamma_color)
            
            channels = []
            for i, color_band in enumerate(self.color_bands):
                channel = self.get_plot_data(
                        color_band,
                        stretch=stretch,
                        stretch_scale=stretch_scale,
                        stretch_interval=stretch_interval,
                        low_clip=low_clips[i],
                        high_clip=high_clips[i],
                        scale_method=scale_method,
                        _internal = True
                    )
                channels.append(channel ** gamma[i])

            return np.stack(channels, axis=2)

        band_state = self._bands[band]
        
        if band_state.config is None:
            band_state.config = {} 
        band_config = band_state.config

        image, _ = self._get_aligned_image(band_state)
     
        if (
            band_state.stretched is None or
            band_config.get("stretch") != stretch or
            band_config.get("stretch_scale") != stretch_scale or
            band_config.get("stretch_interval") != stretch_interval
            ):

            stretch_func = self._get_stretch(stretch, stretch_scale)
            stretch_interval_func = self._get_interval(stretch_interval)

            band_state.stretched = self._stretch_image(
                                    image,
                                    stretch = stretch_func,
                                    stretch_interval = stretch_interval_func)

            band_state.invalidate_from("stretch")
            band_config.update({"stretch": stretch,
                                "stretch_scale": stretch_scale,
                                "stretch_interval": stretch_interval})

        if (
            band_state.clipped is None or
            band_config.get("low_clip") != low_clip or
            band_config.get("high_clip") != high_clip
            ):

            band_state.clipped = self._clip_image(
                                    band_state.stretched,
                                    low=low_clip,
                                    high=high_clip)
            
            band_state.invalidate_from("clip")
            band_config.update({"low_clip": low_clip,
                                "high_clip": high_clip})   
        
        if (band_state.scaled is None or 
            band_config.get("scale_method") != scale_method
            ):

            band_state.scaled = self._scale_image(band_state.clipped,
                                                  scale_method = scale_method)
            band_state.invalidate_from("scale") # in case something is added later
            band_config.update({"scale_method": scale_method})
        
        if band_state.scaled is None:
            raise RuntimeError(f"Scaling failed for band {band}")
        
        band_state.config = band_config

        return band_state.scaled
    

    def world2pixel(self, ra, dec, band):
        """Converts from Sky Coordinates to pixel coordinates
           if available returns the pixel coordinates in the resampled_frame """
        if band not in self._bands:
            raise ValueError(f"Band {band} not found")
        band_state = self._bands[band]
        wcs = band_state._resampled_wcs or band_state.wcs 
        if wcs is None:
            raise ValueError(f"Band {band} has no WCS")
        
        coords = SkyCoord(ra=np.atleast_1d(ra) * u.deg, dec=np.atleast_1d(dec) * u.deg, frame="icrs")
        xpix, ypix = wcs.world_to_pixel(coords)
        return xpix, ypix 
    
  


    def get_arcsec_per_pixel(self, band, scalar = True):
        """
        Return the pixel scale (arcsec/pixel) for a given band.
        If the band has been resampled, use the resampled WCS
        """
        if band not in self._bands:
            raise ValueError(f"Band {band} not found")
    
        band_state = self._bands[band]
        wcs = band_state._resampled_wcs or band_state.wcs
        if wcs is None:
            raise ValueError(f"Band {band} has no WCS defined")
    
    
        scales_deg = proj_plane_pixel_scales(wcs)
        scales_arcsec = scales_deg * 3600.0
        if scalar:
            return np.mean(scales_arcsec)
        
        return scales_arcsec[0], scales_arcsec[1]
    