from functools import partial
import astronomicAL.config as config
import numpy as np
import pandas as pd
import panel as pn
import time



import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from astronomicAL.dashboard.astro_cutouts import euclid_cutouts_class
from astropy import units as u
import time
#pn.extension()

#import io
#from PIL import Image


class SelectedSourceDashboard:
    """A Dashboard used for showing detailed information about a source.

    Optical and Radio images of the source are provided free when the user has
    Right Ascension (RA) and Declination (Dec) information. The user can specify
    what extra information they want to display when a source is selected.

    Parameters
    ----------
    src : ColumnDataSource
        The shared data source which holds the current selected source.

    Attributes
    ----------
    df : DataFrame
        The shared dataframe which holds all the data.
    row : Panel Row
        The panel is housed in a row which can then be rendered by the
        parent Dashboard.
    selected_history : List of str
        List of source ids that have been selected.
    optical_image : Panel Pane JPG
        Widget for holding the JPG image of the selected source based its RADEC.
        The image is pulled from the SDSS SkyServer DR16 site.
    radio_image : Panel Pane GIF
        Widget for holding the GIF image of the selected source based its RADEC.
        The image is pulled from the FIRST cutout server.
    _image_zoom : float
        A float containing the current zoom level of the `optical_image`. The
        zoom is controlled by the zoom in and out buttons on the dashboard
    src : ColumnDataSource
        The shared data source which holds the current selected source.

    """


    def __init__(self, src, close_button):

        self.df = config.main_df

        self.src = src
        self.src.on_change("data", self._panel_cb)

        self.close_button = close_button

        print("reloading information")

        self.row = pn.Row(pn.pane.Str("loading"))


        self.selected_history = []

        self._search_status = ""

        self.euclid_pane = pn.pane.Matplotlib(
                           alt_text="Image Unavailable",
                           min_width=300,
                           min_height=300,
                           sizing_mode="scale_both")
        
        #self.loading_pane = pn.pane.GIF('https://upload.wikimedia.org/wikipedia/commons/d/de/Ajax-loader.gif',
        #                                fixed_aspect = True )

        self.loading_pane = pn.pane.Markdown("# Loading Image...")
        
        self.image_pane = pn.Column(self.euclid_pane)
        
        self._get_ra_dec()
        self.radius = 5.0
        self.get_euclid_object()
        self.get_euclid_cutout()
        self.get_euclid_figure()
        self._initialise_radius_scaling_widgets()

        self._add_selected_info()
        
        self.panel()

    
    def _add_loading_screen(self):
        self.image_pane.clear()
        self.image_pane.append(self.loading_pane)
    
    def _remove_loading_screen(self):
        self.image_pane.clear()
        self.image_pane.append(self.euclid_pane)
    


    def _add_selected_info(self):

        self.contents = "Selected Source"
        self.search_id = pn.widgets.TextInput(
            name="Select ID",
            placeholder="Select a source by ID",
            max_height=50,
        )

        self.search_id.param.watch(self._change_selected, "value")

        self.panel()

    def _change_selected(self, event):

        if event.new == "":
            self._search_status = ""
            self.panel()
            return

        self._search_status = "Searching..."

        if event.new not in list(self.df[config.settings["id_col"]].values):
            self._search_status = "ID not found in dataset"
            self.panel()
            return

        selected_source = self.df[self.df[config.settings["id_col"]] == event.new]

        selected_dict = selected_source.set_index(config.settings["id_col"]).to_dict(
            "list"
        )
        selected_dict[config.settings["id_col"]] = [event.new]
        self.src.data = selected_dict

        self.panel()

    def _panel_cb(self, attr, old, new):
        self._get_ra_dec()
        self.radius = 5
        self._add_loading_screen()
        self.get_euclid_object()
        if self.radius_input.value == self.radius:
                self.get_euclid_cutout()
                self.get_euclid_figure()
                self.contrast_scaler.value = (0,1)
        else:
            self.radius_input.value = self.radius  ###this already calls self.get_euclid_cutout and self.get_euclid_figure()
                                                    ####and self.contrast_scaler.value = (0,255)
        self._remove_loading_screen()
        self.panel()

    def _check_valid_selected(self):
        selected = False

        if config.settings["id_col"] in list(self.src.data.keys()):
            if len(self.src.data[config.settings["id_col"]]) > 0:
                if self.src.data[config.settings["id_col"]][0] in list(
                    self.df[config.settings["id_col"]].values
                ):
                    selected = True

        return selected

    def _add_selected_to_history(self):

        add_source_to_list = True

        if len(self.selected_history) > 0:
            selected_id = self.src.data[config.settings["id_col"]][0]
            top_of_history = self.selected_history[0]
            if selected_id == top_of_history:
                add_source_to_list = False
            elif selected_id == "":
                add_source_to_list = False

        if add_source_to_list:
            self.selected_history = [
                self.src.data[config.settings["id_col"]][0]
            ] + self.selected_history

    def _deselect_source_cb(self, event):
        self.deselect_button.disabled = True
        self.deselect_button.name = "Deselecting..."
        print("deselecting...")
        self.empty_selected()
        print("deselected...")
        self.search_id.value = ""
        self._search_status = ""
        print("blank...")
        self.deselect_button.disabled = False
        self.deselect_button.name = "Deselect"

    def empty_selected(self):
        """Deselect sources by emptying `src.data`.

        Returns
        -------
        None

        """
        empty = {}
        for key in list(self.src.data.keys()):
            empty[key] = []

        self.src.data = empty


    def _update_image(self): 
        try:
             self.euclid_pane.object = self.euclid_fig
        except Exception as e:         #too generic
            print("Euclid image unavailable")
            print(e)
        return None
    

    def check_required_column(self, column):
        """Check `df` has the required column.

        Parameters
        ----------
        column : str
            Check whether this column is in `df`

        Returns
        -------
        has_required : bool
            Whether the column is in `df`.

        """
        has_required = False
        if column in list(self.df.columns):
            has_required = True

        return has_required
    
    def _initialise_radius_scaling_widgets(self):
        self.radius_input = pn.widgets.FloatInput(name = "Radius [arcsec]", value = self.radius,
                                                  step = 0.5, start = 1, end = 100, height=50, width =100,
                                                  sizing_mode="fixed")
        self.radius_input.param.watch(self.update_radius, "value")

        
        self.stretching_input = pn.widgets.MenuButton(name = "Stretching function", 
                                                     items =  [('Linear', 'Linear'), ('Sqrt', 'Sqrt'), ('Log', 'Log'), ('Asinh', 'Asinh'), ('PowerLaw', 'PowerLaw')],
                                                     button_type = "primary", width = 250,
                                                     sizing_mode="fixed")
        
        self.stretching_input.on_click(self.update_stretching)

        
        self.contrast_scaler = pn.widgets.RangeSlider(name = "Image scaling", 
                                                               start = 0, end = 1,value = (0,1), 
                                                               step = 0.004, height=70, width =400,
                                                               sizing_mode="fixed")
        
        self.contrast_scaler.param.watch(self.update_intensity_scaling, "value")


    
    def update_radius(self, event):
        self.radius = event.new
        self._add_loading_screen()
        self.get_euclid_cutout()
        self.contrast_scaler.value = (0,1)
        self.get_euclid_figure()
        self._remove_loading_screen()
        self._update_image()

    @staticmethod
    def change_intensity_range(image, low, high):
        image = np.clip(image, low, high)
        image = (image-low)/(high-low)
        return np.clip(image, 0,1)

    def update_intensity_scaling(self, event):
        low, high = event.new
        scaled_image = self.change_intensity_range(self.euclid_cutout.reprojected_data["stacked"], 
                                                   low, high)
        self.image.set_data(scaled_image)
        self.euclid_fig.canvas.draw()
        self._update_image()
    
    def update_stretching(self, event):
        stretch = event.new
        self.euclid_cutout.stack_cutouts(stretch = stretch)
        self.image.set_data(self.euclid_cutout.reprojected_data["stacked"])
        self.euclid_fig.canvas.draw()
        self._update_image()
    
    
    def get_euclid_object(self):
        self.euclid_cutout = euclid_cutouts_class(self.ra, self.dec, 
                            euclid_filters= ["VIS", "NIR_Y", "NIR_H"])
        
        self.euclid_cutout.get_cone(verbose = True)
        return None
    
    def get_euclid_cutout(self, stretch = "Linear", verbose = True):
        if not hasattr(self, "euclid_cutout"):
            self.get_euclid_object()
        if len(self.euclid_cutout.cone_results) > 2:
            self.euclid_cutout.get_cutouts(radius = self.radius, verbose = verbose)
            self.euclid_cutout.read_cutouts()
            self.euclid_cutout.reproject_cutouts(reference = "VIS")
            self.euclid_cutout.stack_cutouts(stretch = stretch)
        else:
            print("No sources in Euclid dataset with those coordinates")
            return None 
        
    def get_plot_scale(self):
        bar_length_arcsecond = self.bar_length_pixels * self.euclid_cutout.arcsec_per_pix["stacked"]
        return bar_length_arcsecond

    
    def get_euclid_figure(self, show_scale = True):
        
        if hasattr(self, "image"):
            self.image.remove()
            self.image = self.ax.imshow(self.euclid_cutout.reprojected_data["stacked"], origin = "lower")


        self.euclid_fig = plt.Figure()
        self.ax = self.euclid_fig.add_subplot()
        self.ax.set_axis_off()
        self.image = self.ax.imshow(self.euclid_cutout.reprojected_data["stacked"], origin = "lower")
        
        if show_scale:
            image_height, image_width = self.image.get_size()
            self.bar_length_pixels = image_width * 0.2  #always show a bar 1/5 of the plot 
            x0, y0 = 0.1*image_width, 0.1*image_height
            x1 = x0 + self.bar_length_pixels
            self.ax.plot([x0, x1], [y0, y0], color='red', lw=3)
            self.scale_text =  self.ax.text((x0 + x1) / 2, y0 + y0/2, f'{self.get_plot_scale():.1f}"',
                                        color='red', ha='center', va='bottom', fontsize=14, fontweight='bold')


    def get_info_pane(self):
        extra_data_list = [
        ["Source ID", self.src.data[config.settings["id_col"]][0]]]
        for i, col in enumerate(config.settings["extra_info_cols"]):
            extra_data_list.append([col, self.src.data[f"{col}"][0]])
            extra_data_df = pd.DataFrame(extra_data_list, columns=["Column", "Value"])
            extra_data_pn = pn.widgets.DataFrame(
                            extra_data_df, show_index=False, autosize_mode="fit_viewport")
            return extra_data_pn
           
    def panel(self):
        """Render the current view.

        Returns
        -------
        row : Panel Row
            The panel is housed in a row which can then be rendered by the
            parent Dashboard.

        """
        # CHANGED :: Remove need to rerender with increases + decreases

        selected = self._check_valid_selected()

        if selected:

            self._add_selected_to_history()

            self.deselect_button = pn.widgets.Button(name="Deselect")
            self.deselect_button.on_click(self._deselect_source_cb)

            self.row[0] = pn.Card(
                pn.Column(
                    pn.Row(
                            pn.Column(self.image_pane,
                                      pn.Row(self.radius_input, self.stretching_input, align = "center"),
                                      self.contrast_scaler),
                            pn.Row(self.get_info_pane(), max_height=250, max_width=300),
                          ),
                        ),
                collapsible=False,
                header=pn.Row(self.close_button, self.deselect_button, max_width=300),
                )
            self._update_image()

        else:
            self.row[0] = pn.Card(
                pn.Column(
                    self.search_id,
                    self._search_status,
                    pn.Row(
                        pn.widgets.DataFrame(
                            pd.DataFrame(
                                self.selected_history, columns=["Selected IDs"]
                            ),
                            show_index=False,
                        ),
                        max_width=300,
                    ),
                ),
                header=pn.Row(self.close_button, max_width=300),
            )

        return self.row


    def _get_ra_dec(self):
        if self.check_required_column("ra_dec"):
            ra_dec = self.src.data["ra_dec"][0]
            self.ra = float(ra_dec[: ra_dec.index(",")])
            self.dec = float(ra_dec[ra_dec.index(",") + 1 :])
        else:
            print("No ra and dec available for this source")
            self.ra, self.dec = None, None





