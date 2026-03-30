import holoviews as hv
import astronomicAL.config as config
import numpy as np
import os
import html
import pandas as pd
import panel as pn
import json
import param
import uuid
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from panel.io import save
from astronomicAL.utils.optimise import matches_type
from astronomicAL.extensions.shared_data import shared_data
from astronomicAL.extensions.astro_visualization_utility import ImageVisaulizationClass



class CustomPanel(param.Parameterized):
    
    def __init__(self,
                data,
                src,
                close_button,
                panel_name = "custom_plot",
                ):
        super().__init__()
        self.df = data
        self.src = src
        self.close_button = close_button
        self.panel_id = str(uuid.uuid4())
        self.panel_name = panel_name
        print(f"Creating a {self.panel_name} panel")
        self._src_callback = self._change_source_cb
        self.src.on_change("data", self._src_callback)
        self.figure = pn.pane.HoloViews(sizing_mode="stretch_both")
        self.message_pane = pn.pane.Markdown("## Loading...", sizing_mode="stretch_width", height = 80)
        self.settings_button = pn.widgets.Button(name="Open Settings", button_type="primary", max_height = 40, max_width=100, sizing_mode="stretch_both" )
        self.settings_button.on_click(self._toggle_settings_panel)
        self.settings_panel = pn.Column(visible = False, scroll = True)

    def _change_source_cb(self, attr, old, new):
        self._get_data()
    
    def _get_data(self):
        raise NotImplementedError
    
    def _toggle_settings_panel(self, event):
        self.settings_panel.visible = not self.settings_panel.visible
        self.settings_button.name = "Close Settings" if self.settings_panel.visible else "Open Settings"

    def _get_selected_source(self):
        if self.src is None:
            return None
        cols = list(self.df.columns)
        if len(self.src.data[cols[0]]) == 1:
            return pd.DataFrame(self.src.data, columns=cols, index=[0])
        return None
    
    def _get_value_from_df(self, column):
        selected_source = self._get_selected_source()
        if (selected_source is not None) and self._check_required_column(column):
            return selected_source[column][0]
        return None
            
    def _get_ra_dec(self, err_message = "No ra and dec available for this source"):
        ra_dec = self._get_value_from_df("ra_dec")
        if ra_dec is not None:
            ra = float(ra_dec[: ra_dec.index(",")])
            dec = float(ra_dec[ra_dec.index(",") + 1 :])
        else:
            print(err_message)
            ra, dec = None, None
        return ra, dec
    
    def _get_selected_id(self):
        return self._get_value_from_df(config.settings["id_col"])

    def _check_required_column(self, column):
        return column in self.df.columns
    
    def _save_panel(self, directory_path = "data/saved_sources", 
                    save_fits_files = True, 
                    prefix = None,): 
        paths = {}
        try:
            paths["figure"] = self._save_figure(directory_path = directory_path, prefix = prefix)
        except AttributeError:
            print(f"{self.panel_name} has no _save_panel_method")
            
        if save_fits_files:
            try:
                paths["fits_file"] = self._save_data_to_fits(directory_path = directory_path)
            #paths["fits_file"] is currently always None
            except AttributeError:
                pass
        return paths
            
    @staticmethod
    def _get_empty_image():
        """Returns a completely white image to update the previous one if the query fails"""
        return hv.Image(np.ones((10,10))).opts(active_tools =[], 
                                            clim = (0,1), toolbar=None,
                                            padding = 0,border = 0,framewise = True, xaxis=None, 
                                            yaxis=None, cmap = "grey")
    
    def _initialise_param_objects(self, **extra_params):
        """
        This method initilizes the param objects of the class to the values in the config file
        """
        self.param.update(**extra_params)
    
    def get_error_panel(self, message_1, message_2):
        message = f"# {message_1}:\n"  
        message += f"## {message_2}"
        self.message_pane.object = message
        self.message_pane.visible = True 
        self.figure.objects = [self._get_empty_image()]

    def _subscribe_to_shared(self, key, function):
        if not shared_data.is_subscribed(self.panel_id, key):
               shared_data.subscribe(self.panel_id, key, function)

    def _remove_shared_data(self):
        """Removes subscriptions and published data from the shared data"""
        shared_data.cleanup_extension_panel(self.panel_id)
        print(f"[{self.panel_id}] removed from shared data")

    
    def _remove_src_listener(self):
        """Removes the callback to a change in the selected source"""
        if self.src is not None and hasattr(self, "_src_callback"):
            try:
                self.src.remove_on_change("data", self._src_callback)
                print(f"[{self.panel_id}] Listener removed")
            except Exception as e:
                print(f"[{self.panel_id}] Error removing src listener: {e}")
    
    def _manage_subscriptions(self):
        raise NotImplementedError
    
    def cleanup_panel_plot(self):
        self._remove_shared_data()
        self._remove_src_listener()

    def _update_plot(self):
        raise NotImplementedError("Implement the _update_plot function in your child class")

    def get_toolbar(self):
        return pn.Row(
                    pn.Spacer(width=25,),
                    self.close_button,
                    self.settings_button,
                    max_width=400, max_height=50
                    )
    
    def get_body(self):
        return  pn.Column(
                    self.message_pane,
                    pn.Row(self.figure, sizing_mode="scale_both"),
                    self.settings_panel, scroll = True)

    def panel(self):
        self._update_plot()
        toolbar = self.get_toolbar()
        body = self.get_body()
        return pn.Column(
                toolbar, body,
                sizing_mode="stretch_both")






