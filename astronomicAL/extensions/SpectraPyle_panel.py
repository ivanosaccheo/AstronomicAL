import holoviews as hv
import astronomicAL.config as config
import numpy as np
import os
import pandas as pd
import panel as pn
import json
import param
import uuid
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from panel.io import save
from astronomicAL.utils.optimise import matches_type
from pydantic import ValidationError
import sys
sys.path.append("SpectraPyle/project_root/src")
from spectraPyle.runtime.runtime_adapter import build_config_from_dict
from spectraPyle.schema.schema import StackingConfigResolver
from  spectraPyle.stacking import stacking


class SpectraPylePanel(param.Parameterized):
    
    _NAMING_CONVECTION = {}


    #module_dir = os.path.dirname(spectraPyle.__file__)
    module_dir = "SpectraPyle/project_root/src/spectraPyle"
    rules_path = os.path.join(module_dir, "instruments", "instruments_rules.json")
    with open(rules_path) as f:
        _RULES = json.load(f)
    


    #Cosmology 
    cosmo_H0 = param.Number(default = 70, step = 0.1, bounds = (0,100), allow_None = False)
    cosmo_Om0 = param.Number(default = 0.3, step = 0.01, bounds = (0,1),allow_None = False)
    
    #Redshift
    z_type  = param.Selector(objects ={'Rest frame': "rest_frame",
                                      "Observed frame": "observed_frame",
                                      "Minimum z" : "minimum_z",
                                      "Maximum z":  "maximum_z",
                                      "Median z" : "median_z",
                                      "Custom value": "custom"}, 
                                      default = "rest_frame")
    z_value = param.Number(default = 0.0, bounds = (0,None), allow_None=False)

    #Normalization 
    spectra_normalization = param.Selector(objects = ["no_normalization", "custom", "median", 
                                            "interval", "integral", "template"], default = "median")
    conservation = param.Selector(objects = ["flux", "luminosity"], default = "flux")

    #Interval wavelength for interval normalization, and statistic to apply to the flux in the interval:
    lambda_min = param.Number(default = None, allow_None = True)
    lambda_max = param.Number(default = None, allow_None = True)
    interval_stat = param.Selector(objects=["median", "mean", "maximum", "minimum"], default = "median")
    
    #Resampling
    pixel_resampling_type = param.Selector(objects={"Linear λ sampling" : "lambda", 
                                        "Log λ sampling" : "log_lambda",
                                        "Shifted λ sampling" : "lambda_shifted",
                                        "No resampling (observed frame only)" :  None},
                                        default="lambda", 
                                        doc = "`None` allowed only in observed_frame and requires identical wavelength grids.")
    pixel_size_type = param.Selector(objects = {"Manual pixel size" : "manual",
                                        "Instrumental resolution" : "instrumental"}, default = "manual",
                                        doc = "`Instrumental resolution`: the pixel size will be calculated according to the instrumental resolution provided in the instrument file.")
    pixel_resampling = param.Number(default = 6, bounds = (0, None), allow_None = False)
    nyquist_sampling = param.Integer(default = 6, bounds = (0, None), allow_None = False)
    
    #Edges cropping
    lambda_enable = param.Boolean(default=False)
    left_edge = param.Number(default = 6150, step =1, bounds = (0, None), allow_None=False)
    right_edge = param.Number(default = 6750, step =1, bounds = (0, None), allow_None=False)

    edge_enable = param.Boolean(default = False)
    first_pixel = param.Integer(default = 10, step =1, allow_None=False)
    last_pixel = param.Integer(default = -10, step =1, allow_None=False)
    
    #Various
    bootstrapping_R = param.Integer(default=300, bounds = (0,1000), allow_None=False)
    sigma_clipping_conditions = param.Number(default=4, bounds = (0,5), step = 0.25, allow_None=False)
    parallel_enable = param.Boolean(default = True)
    max_cpu_fraction = param.Number(default = 0.9, allow_None = False, bounds=(0.1, 0.95), step = 0.05)
    
    #Mission specific
    instrument_name = param.Selector(objects = list(_RULES.keys()))
    survey_name = param.Selector(objects = [])
    grism_type = param.Selector(objects = [])
    data_release = param.Selector(objects = [])
    pixel_mask = param.ListSelector(default = [0,6],  objects=list(range(7)), doc= "Bad pixel bits")
    n_min_dithers = param.Integer(default = 2, bounds = (1,50), step =1, 
                           doc = "Minimum number of dithers in the coadded 1D spectrum. Higher is better. Note: Euclid Q1 max dithers = 4 (recommended ≥ 2).")
    
    #Input/Output
    spectra_dir = param.String(default = '', allow_None = True, doc = "Leave empty if single paths are complete")
    output_dir = param.String(default = "data", allow_None = False)
    filename_out = param.String(default = "AUTO", allow_None= False, doc = "'AUTO' will generate an automatic filename")
    spectra_path_column = param.Selector(default = [], objects =[],  allow_None = False)
    galactic_extinction = param.Boolean(default = False)
    gal_ext_column_name = param.Selector(default = [], objects =[],  allow_None = False)
    redshift_column_name = param.Selector(default = [], objects =[],  allow_None = False)
    ID_column_name = param.Selector(default = [], objects =[],  allow_None = False)

    spectra_mode = param.Selector(objects =["individual fits", "metadata path", "combined fits"],
                                   default =  "individual fits", doc="Spectra format")
    
    input_dir = param.String(default = "Boh", allow_None= False)
    filename_in = param.String(default = "Boh", allow_None= False)
    filename_in_extention = param.Selector(default = "csv", objects =["npz", "fits", "csv"],  allow_None = False)


    #My params for updating the UI
    _show_survey      = param.Boolean(default=False)
    _show_datarelease = param.Boolean(default=False)
    _show_grism       = param.Boolean(default=False)

    def __init__(self,
                data,
                close_button,
                panel_name = "SpectraPyle",
                **params
                ):
        super().__init__(**params) 
        self.df = data
        self.close_button = close_button
        self.panel_id = str(uuid.uuid4())
        self.panel_name = panel_name
        self._update_column_options()
        print(f"Creating a {self.panel_name} panel")
        self._initialise_widgets()
        self.param.watch(self._update_instrument_options, ["instrument_name"])
        self.param.watch(self._update_survey_options, ["instrument_name", "survey_name"])
        self._update_instrument_options()
        self._update_survey_options()
        self._export_config_file()

        self.layout =  self._update_layout()


    def _initialise_widgets(self):
       
        float_kwargs = {"width" : 120, "height" : 40}
        int_kwargs = {"width" : 120, "height" : 40}
        select_kwargs = {"width":200, "height":80}
   
        
 
        self.param_widgets = {
                        "cosmo_H0" : pn.widgets.FloatInput.from_param(
                               self.param.cosmo_H0, name = "H0", **float_kwargs),
                        "cosmo_Om0" : pn.widgets.FloatInput.from_param(
                               self.param.cosmo_Om0, name = "Omega0", **float_kwargs),
                        "z_value" : pn.widgets.FloatInput.from_param(
                               self.param.z_value, name = "Custom Redshift", **float_kwargs),
                        "lambda_min" : pn.widgets.FloatInput.from_param(
                               self.param.lambda_min, name = "λ min", **float_kwargs),
                        "lambda_max" : pn.widgets.FloatInput.from_param(
                               self.param.lambda_max, name = "λ max", **float_kwargs),
                        "max_cpu_fraction" : pn.widgets.FloatSlider.from_param(
                               self.param.max_cpu_fraction , name = "CPU Fraction"),
                        "pixel_resampling" : pn.widgets.FloatInput.from_param(
                               self.param.pixel_resampling, name = "Δλ [Å]", **float_kwargs),
                        "left_edge" : pn.widgets.FloatInput.from_param(
                               self.param.left_edge, name = "Left edge", **float_kwargs),
                        "right_edge" : pn.widgets.FloatInput.from_param(
                               self.param.right_edge, name = "Right edge",  **float_kwargs),  
                        "nyquist_sampling" : pn.widgets.IntInput.from_param(
                               self.param.nyquist_sampling, name = "Nyquist sampling N", **int_kwargs),
                        "first_pixel" : pn.widgets.IntInput.from_param(
                               self.param.first_pixel, name = "First pixel", **int_kwargs),
                        "last_pixel" : pn.widgets.IntInput.from_param(
                               self.param.last_pixel, name = "Last pixel", **int_kwargs),
                        "bootstrapping_R" : pn.widgets.IntInput.from_param(
                               self.param.bootstrapping_R, name = "Bootstrap R", **int_kwargs),
                        "n_min_dithers" : pn.widgets.IntInput.from_param(
                               self.param.n_min_dithers, name = "N dither min", **int_kwargs),
                        "lambda_enable" : pn.widgets.Checkbox.from_param(
                               self.param.lambda_enable, name = "Limit wavelength range"),
                        "edge_enable" : pn.widgets.Checkbox.from_param(
                               self.param.edge_enable, name = "Crop spectrum range"),
                        "multiprocessing" : pn.widgets.Checkbox.from_param(
                               self.param.parallel_enable, name = "Enable Multiprocessing"),
                        "z_type": pn.widgets.Select.from_param(
                               self.param.z_type, name = "Redshift Type", **select_kwargs),
                        "spectra_normalization": pn.widgets.Select.from_param(
                               self.param.spectra_normalization, name = "Normalization", **select_kwargs),
                        "conservation": pn.widgets.Select.from_param(
                               self.param.conservation, name = "Conservation", **select_kwargs),
                        "interval_stat": pn.widgets.Select.from_param(
                               self.param.interval_stat, **select_kwargs),
                        "pixel_resampling_type": pn.widgets.Select.from_param(
                               self.param.pixel_resampling_type, name = "Resampling type",  **select_kwargs),
                        "pixel_size_type": pn.widgets.RadioBoxGroup.from_param(
                               self.param.pixel_size_type, inline = True),
                        "instrument_name": pn.widgets.Select.from_param(
                               self.param.instrument_name, name = "Instrument", **select_kwargs),
                        "survey_name": pn.widgets.Select.from_param(
                               self.param.survey_name,  name = "Survey", **select_kwargs),
                        "grism_type": pn.widgets.Select.from_param(
                               self.param.grism_type, name = "Grism", **select_kwargs),
                        "data_release": pn.widgets.Select.from_param(
                               self.param.data_release, name = "Data Release", **select_kwargs),
                        "sigma_clipping_conditions" : pn.widgets.FloatSlider.from_param(
                               self.param.sigma_clipping_conditions, name = "Sigma Clip", width = 130),
                        "pixel_mask": pn.widgets.CheckBoxGroup.from_param(
                               self.param.pixel_mask, name = "Bad pixel bits", inline = True),
                        "spectra_path_column": pn.widgets.Select.from_param(
                               self.param.spectra_path_column, name = "Column with spectra path",
                               **select_kwargs),
                        "gal_ext_column_name": pn.widgets.Select.from_param(
                               self.param.gal_ext_column_name, name = "Column with E[B-V]",
                               **select_kwargs),
                        "galactic_extinction" : pn.widgets.Checkbox.from_param(
                               self.param.galactic_extinction, name = "Galactic E[B-V]"),
                        "spectra_dir" : pn.widgets.TextInput.from_param(
                               self.param.spectra_dir, name = "Directory with Spectra"),
                        "output_dir" :  pn.widgets.TextInput.from_param(
                                self.param.output_dir, name = "Output Directory"),
                        "filename_out" :  pn.widgets.TextInput.from_param(
                                self.param.filename_out, name = "Output filename"),
                        "redshift_column_name": pn.widgets.Select.from_param(
                               self.param.redshift_column_name, name = "Column with Redshift",
                               **select_kwargs),
                        "spectra_mode": pn.widgets.RadioButtonGroup.from_param(
                               self.param.spectra_mode, name = "Spectra input mode",
                               disabled = True, button_type =  "primary"),
                        "ID_column_name": pn.widgets.Select.from_param(
                               self.param.ID_column_name, name = "Column with ObjectID",
                               **select_kwargs),
                        "input_dir" :  pn.widgets.TextInput.from_param(
                                self.param.input_dir, name = "Input table Directory", disabled = False),
                        "filename_in" :  pn.widgets.TextInput.from_param(
                                self.param.filename_in, name = "Input table filename", disabled = False),
                        "filename_in_extention" :  pn.widgets.Select.from_param(
                                self.param.filename_in_extention, name = "Input table format", 
                                disabled = False, **select_kwargs),
                        
       }
        
        self.verify_config_button = pn.widgets.Button(name = "Verify config", button_type = "warning")
        self.run_code_button = pn.widgets.Button(name = "Run", button_type = "danger", disabled = True)
        self.verify_config_button.on_click(self._verify_config_cb)
        self.run_code_button.on_click(self._run_code_cb)

    
    def _update_instrument_options(self, *events):
        instrument_rules = self._RULES.get(self.instrument_name, {})
        survey_options = list(instrument_rules.get("surveys", {}).keys())
        current = self.survey_name
        self.param.survey_name.objects = survey_options
        self.survey_name = current if current in survey_options else (survey_options[0] if survey_options else None)
        self._show_survey = bool(survey_options)

    def _update_survey_options(self, *events):
        survey_rules = self._RULES.get(self.instrument_name, {}).get("surveys", {}).get(self.survey_name, {})

        dr_options = list(survey_rules.get("data_release", []))
        current = self.data_release
        self.param.data_release.objects = dr_options
        self.data_release = current if current in dr_options else (dr_options[0] if dr_options else None)
        self._show_datarelease = bool(dr_options)

        grism_options = list(survey_rules.get("grisms", []))
        current = self.grism_type
        self.param.grism_type.objects = grism_options
        self.grism_type = current if current in grism_options else (grism_options[0] if grism_options else None)
        self._show_grism = bool(grism_options)


    def _update_column_options(self):
        objectid_options = [col for col in self.df.columns if matches_type(self.df[col].dtype, ["int", "float", "string", "object"])]
        path_options = [col for col in self.df.columns if matches_type(self.df[col].dtype, ["string", "object"])]
        float_options = [col for col in self.df.columns if matches_type(self.df[col].dtype, ["float"])]
        
        self.param.spectra_path_column.objects = path_options
        self.param.gal_ext_column_name.objects = float_options
        self.param.redshift_column_name.objects = float_options
        self.param.ID_column_name.objects = objectid_options



    def _get_input_output_layout(self, **markdown_kwargs):
         layout = pn.Column(pn.pane.Markdown("## Input and Output", **markdown_kwargs),
                            self.param_widgets["spectra_mode"],
                            self.param_widgets["spectra_path_column"],
                            self.param_widgets["ID_column_name"],
                            self.param_widgets["redshift_column_name"],
                            self.param_widgets["input_dir"],
                            self.param_widgets["filename_in"],
                            self.param_widgets["filename_in_extention"],
                            self.param_widgets["spectra_dir"],
                            self.param_widgets["filename_out"],
                            self.param_widgets["output_dir"],)
         return layout
         
    def _get_instrument_layout(self, **markdown_kwargs):
        layout = pn.Column(pn.pane.Markdown("## Instrument", **markdown_kwargs),
                            self.param_widgets["instrument_name"])
        if self._show_survey:
            layout.append(self.param_widgets["survey_name"])
        if self._show_datarelease:
            layout.append(self.param_widgets["data_release"])
        if self._show_grism:
            layout.append(self.param_widgets["grism_type"])
        if self.instrument_name == "euclid":
            layout.append(self.param_widgets["n_min_dithers"])
            layout.append(self.param_widgets["pixel_mask"])
        return layout

    def _get_cosmology_layout(self, **markdown_kwargs):
        layout = pn.Column(pn.pane.Markdown("## Cosmology", **markdown_kwargs),
                 pn.Row(self.param_widgets["cosmo_H0"],self.param_widgets["cosmo_Om0"]))
        return layout
    
    def _get_redshift_layout(self, **markdown_kwargs):
        layout = pn.Column(pn.pane.Markdown("## Redshift", **markdown_kwargs),
                            self.param_widgets["z_type"])
        if self.z_type =="custom":
            layout.append(self.param_widgets["z_value"])
        return layout
        
    def _get_normalization_layout(self, **markdown_kwargs):
        layout = pn.Column(pn.pane.Markdown("## Normalization", **markdown_kwargs),
                           self.param_widgets["spectra_normalization"])
        if self.spectra_normalization  == "no_normalization":
            layout.append(pn.pane.Markdown("### Flux conservation mode", **markdown_kwargs))
            layout.append(self.param_widgets["conservation"])
        elif self.spectra_normalization == "interval":
            layout.append(pn.pane.Markdown("### Normalization Wavelength interval (rest-frame)",  **markdown_kwargs ))
            layout.append(pn.Row(self.param_widgets["lambda_min"], self.param_widgets["lambda_min"]))
            layout.append(self.param_widgets["interval_stat"])
        return layout
    
    def _get_resampling_layout(self, **markdown_kwargs):
        layout = pn.Column(pn.pane.Markdown("## Resampling", **markdown_kwargs),
                            self.param_widgets["pixel_resampling_type"])
        if self.pixel_resampling_type is not None:
            layout.append(self.param_widgets["pixel_size_type"])
            if self.pixel_size_type == "manual":
                layout.append(self.param_widgets["pixel_resampling"])
            elif self.pixel_size_type == "instrumental":
                layout.append(self.param_widgets["nyquist_sampling"])  
        return layout

    def _get_wavelength_layout(self, **markdown_kwargs):
        layout = pn.Column(pn.pane.Markdown("## Refining wavelength extent (optional)",  **markdown_kwargs),
                                           self.param_widgets["lambda_enable"])
        if self.lambda_enable:
            layout.append(self._get_HTML_pane("""
                        <div style="border-left:6px solid #2c7fb8; background:#eef6fb; padding:10px;">
                        Optional wavelength restriction for the stacked spectrum.<br><br>
                        
                        If enabled, the stacked spectrum will only include wavelengths between:<br>
                        <b>left_edge × (1+z_stacking)</b> and <b>right_edge × (1+z_stacking)</b><br><br>
                        
                        Useful to focus on emission lines or specific spectral regions.
                        </div>
                        """))
            layout.append(pn.Row(self.param_widgets["left_edge"], self.param_widgets["right_edge"]))
        layout.append( self.param_widgets["edge_enable"])
        if self.edge_enable:
            layout.append(self._get_HTML_pane("""
                        <div style="border-left:6px solid #f28e2b; background:#fff4e6; padding:10px;">
                        Optional cropping of spectrum edges.<br><br>
                        
                        Removes the first N and last M pixels before stacking.<br><br>
                        
                        Uses Python slicing rules.<br>
                        Example: first=10, last=-10 → keeps spectrum[10:-10]
                        </div>
                        """))
            layout.append(pn.Row(self.param_widgets["first_pixel"], self.param_widgets["last_pixel"]))
        return layout

    def _get_miscellaneous_layout(self, **markdown_kwargs):
        layout = pn.Column(pn.pane.Markdown("## Sigma Clipping",  **markdown_kwargs),
                                           self.param_widgets["sigma_clipping_conditions"])
        layout.append(pn.pane.Markdown("## Bootstrap",  **markdown_kwargs))
        layout.append(self.param_widgets["bootstrapping_R"])
        layout.append(pn.pane.Markdown("## E[B-V]",  **markdown_kwargs))
        layout.append(self.param_widgets["galactic_extinction"])
        if self.galactic_extinction:
            layout.append(self.param_widgets["gal_ext_column_name"])
        layout.append(pn.pane.Markdown("## Parallel",  **markdown_kwargs))
        layout.append(self.param_widgets["multiprocessing"])
        if self.parallel_enable:
            layout.append(self.param_widgets["max_cpu_fraction"])
        return layout
    
    def _get_HTML_pane(self, html_text,  **kwargs):
        return pn.pane.HTML(html_text, **kwargs)
    
    
    @param.depends("cosmo_H0", "cosmo_Om0", "z_value", "lambda_min","lambda_max","max_cpu_fraction","pixel_resampling",
    "left_edge","right_edge","nyquist_sampling","first_pixel","last_pixel","bootstrapping_R",
    "n_min_dithers","lambda_enable", "edge_enable","parallel_enable","z_type",
    "spectra_normalization","conservation","interval_stat","pixel_resampling_type","pixel_size_type",
    "instrument_name","survey_name","grism_type","data_release", "sigma_clipping_conditions", "pixel_mask", "spectra_path_column",
    "gal_ext_column_name", "galactic_extinction", "spectra_dir", "output_dir", "spectra_mode", "redshift_column_name",
    "ID_column_name", "filename_in", "filename_in_extention", "input_dir",
    watch = True)
    def _export_config_file(self):
        """Creates the config dictionary based on the widgets values"""
        self.config = {name: getattr(widget, "value", None)
                        for name, widget in self.param_widgets.items()}
       
        self.config["lambda_edges_rest"], self.config["spectrum_edges"] = self._build_wavelength_config()
        self._sanitaze_config_file()
        self.run_code_button.disabled = True
        

    def _build_wavelength_config(self):
        if self.lambda_enable:
            lambda_edges_rest = [float(self.left_edge),  float(self.right_edge)]
        else:
            lambda_edges_rest = None

        if self.edge_enable:
            spectrum_edges = [int(self.first_pixel),int(self.last_pixel)]
       
        else:
            spectrum_edges = None

        return lambda_edges_rest, spectrum_edges
    
    def _sanitaze_config_file(self):
        """This function applies some cleaning to the congig file to be accepted 
        from Salvatore config validation code"""
        if self.spectra_normalization != "no_normalization":
           self.config["conservation"] = None
        if not self.galactic_extinction:
            self.config["gal_ext_column_name"] = None
       
        self.config["plot_results"] = False

    def _run_code_cb(self, event):
        if hasattr(self, "validated_config"):
            print("Here I should run the code")
            stacking.main(self.validated_config)
            

    def _verify_config_cb(self, event):
        try:
           self.validated_config = build_config_from_dict(self.config)
           self.validated_config = StackingConfigResolver.resolve(self.validated_config)
           self.run_code_button.disabled = False
           self._update_layout()
        except ValidationError as e:
            print(e)
            self.run_code_button.disabled = True
            self._update_layout()

    def get_toolbar(self):
        return pn.Row(
                    pn.Spacer(width=25,),
                    self.close_button,
                    max_width=400, max_height=50
                    )
    
    def get_body(self, markdown_kwargs = {}):
        in_out_layout = self._get_input_output_layout(**markdown_kwargs)
        instrument_layout = self._get_instrument_layout(**markdown_kwargs)
        cosmology_layout = self._get_cosmology_layout(**markdown_kwargs)
        redshift_layout = self._get_redshift_layout(**markdown_kwargs)
        normalization_layout = self._get_normalization_layout(**markdown_kwargs)
        resampling_layout = self._get_resampling_layout(**markdown_kwargs)                                  
        wavelength_layout  = self._get_wavelength_layout(**markdown_kwargs)
        miscellaneous_layout = self._get_miscellaneous_layout(**markdown_kwargs)
       
        return pn.Column(
                        in_out_layout,
                        instrument_layout, 
                        cosmology_layout,
                        redshift_layout,
                        normalization_layout,
                        resampling_layout,
                        wavelength_layout,
                        miscellaneous_layout,
                        pn.Row(self.verify_config_button, self.run_code_button),
                        scroll = True, width = 600)
        
    @param.depends("z_type", "parallel_enable",
                   "edge_enable", "lambda_enable",
                   "_show_survey", "_show_datarelease", "_show_grism",
                   "pixel_size_type", "spectra_normalization", "galactic_extinction", "instrument_name")
    def _update_layout(self):  
        body = self.get_body()
        toolbar = self.get_toolbar()
        return pn.Column(toolbar, body,
                sizing_mode="stretch_both")
       
            
    def panel(self):
        return self._update_layout
    

