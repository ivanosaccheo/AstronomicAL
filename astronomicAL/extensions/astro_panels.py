import asyncio
import holoviews as hv
import astronomicAL.config as config
import numpy as np
import panel as pn
import param
import matplotlib.pyplot as plt

from astronomicAL.extensions.custom_panels_layouts import CustomPanel
from astronomicAL.extensions.astro_visualization_utility import ImageVisaulizationClass
from astronomicAL.extensions.shared_data import shared_data
from astronomicAL.extensions.astro_data_utility import EuclidCutoutsClass


class AstroImagePanel(CustomPanel):
    """Panel for rendering and interacting with astrophysics image cutouts.

    Inherits general panel machinery (source callbacks, shared data subscriptions,
    toolbar/body layout) from CustomPanel. This class owns the image display
    widgets and coordinates with ImageVisaulizationClass for all pixel
    transformations. Data retrieval is delegated to _get_images_function, which
    must be implemented by concrete subclasses.
    """

    filter = param.Selector(objects=[""], default="", doc="Band to be shown")
    radius = param.Number(default=5, bounds=(1, 100), step=0.1, allow_None=False,
                          doc="Radius of the cutout in arcsec")
    clipping = param.Range(default=(0., 1.), bounds=(0, 1), step=0.01,
                           doc="Clipping range of the image")
    stretching = param.Selector(objects=["Linear", "Sqrt", "Log", "Asinh", "PowerLaw"],
                                default="Linear", doc="Stretch function of the image")
    scaling = param.Selector(objects=["MinMax", "Expand"], default="MinMax",
                             doc="Scaling method for the image")
    gamma_r = param.Number(default=1, bounds=(0, 1), step=0.01, allow_None=False,
                           doc="R channel gamma correction")
    gamma_g = param.Number(default=1, bounds=(0, 1), step=0.01, allow_None=False,
                           doc="G channel gamma correction")
    gamma_b = param.Number(default=1, bounds=(0, 1), step=0.01, allow_None=False,
                           doc="B channel gamma correction")
    clipping_r = param.Range(default=(0., 1.), bounds=(0, 1), step=0.01,
                             doc="Clipping range of R channel")
    clipping_g = param.Range(default=(0., 1.), bounds=(0, 1), step=0.01,
                             doc="Clipping range of G channel")
    clipping_b = param.Range(default=(0., 1.), bounds=(0, 1), step=0.01,
                             doc="Clipping range of B channel")
    source_coordinates = param.Boolean(default=False, doc="Overplot source coordinates")
    external_coordinates = param.Boolean(default=False, doc="Overplot external coordinates")

    def __init__(self,
                 data,
                 src,
                 close_button,
                 panel_name="Generic_Cutout",
                 color_image=False,
                 color_bands=None,
                 reference_wcs=None,
                 panels_to_subscribe=None):
        super().__init__(data, src, close_button, panel_name=panel_name)

        self.color_image = color_image
        self.color_bands = color_bands
        self.reference_wcs = reference_wcs
        self.panels_to_subscribe = panels_to_subscribe

        self.image_container = None
        self.image_stream = None
        self._tap_watcher = None

        self._initialise_settings_panel()
        self._initialise_settings_dictionary()
        self.param.watch(self._update_settings_dictionary_cb, list(self.param))
        self._manage_subscriptions()

    # Bridge: CustomPanel._change_source_cb calls self._get_data() synchronously;
    # override it here to schedule the async version instead.
    def _change_source_cb(self, attr, old, new):
        asyncio.ensure_future(self._get_data())

    # ------------------------------------------------------------------
    # Settings panel
    # ------------------------------------------------------------------

    def _initialise_settings_panel(self):
        self.param_widgets = {
            "radius": pn.widgets.FloatInput.from_param(
                self.param.radius, width=120, height=80),
            "stretching": pn.widgets.Select.from_param(
                self.param.stretching, width=200, height=80),
            "scaling": pn.widgets.Select.from_param(
                self.param.scaling, width=200, height=80),
            "clipping": pn.widgets.RangeSlider.from_param(
                self.param.clipping, max_width=200, sizing_mode="stretch_both"),
            "filter": pn.widgets.Select.from_param(
                self.param.filter, width=150, height=80),
            "source_coordinates": pn.widgets.Checkbox.from_param(
                self.param.source_coordinates),
            "external_coordinates": pn.widgets.Checkbox.from_param(
                self.param.external_coordinates),
        }

        self.image_settings = pn.Column(
            self.param_widgets["clipping"],
            self.param_widgets["radius"],
            pn.Row(self.param_widgets["stretching"], self.param_widgets["scaling"]),
            self.param_widgets["filter"],
            pn.Row(self.param_widgets["source_coordinates"],
                   self.param_widgets["external_coordinates"]),
        )

        if self.color_image:
            self.param_widgets.update({
                "clipping_r": pn.widgets.RangeSlider.from_param(
                    self.param.clipping_r, max_width=200, sizing_mode="stretch_both",
                    bar_color="red"),
                "clipping_g": pn.widgets.RangeSlider.from_param(
                    self.param.clipping_g, max_width=200, sizing_mode="stretch_both",
                    bar_color="green"),
                "clipping_b": pn.widgets.RangeSlider.from_param(
                    self.param.clipping_b, max_width=200, sizing_mode="stretch_both",
                    bar_color="blue"),
                "gamma_r": pn.widgets.FloatInput.from_param(
                    self.param.gamma_r, width=80, height=80),
                "gamma_g": pn.widgets.FloatInput.from_param(
                    self.param.gamma_g, width=80, height=80),
                "gamma_b": pn.widgets.FloatInput.from_param(
                    self.param.gamma_b, width=80, height=80),
            })
            self.color_image_settings = pn.Column(
                self.param_widgets["clipping_r"],
                self.param_widgets["clipping_g"],
                self.param_widgets["clipping_b"],
                pn.Row(
                    self.param_widgets["gamma_r"],
                    self.param_widgets["gamma_g"],
                    self.param_widgets["gamma_b"],
                ),
            )

    def _get_settings_sections(self):
        """Return (title, content) pairs for the settings accordion.

        Subclasses should call super() and append their own sections.
        """
        sections = [("Image Settings", self.image_settings)]
        if self.color_image:
            sections.append(("Color Image Settings", self.color_image_settings))
        return sections

    def get_toolbar(self):
        return pn.Row(
            pn.Spacer(width=25),
            self.close_button,
            max_width=400, max_height=50,
        )

    def get_body(self):
        settings = pn.Accordion(*self._get_settings_sections(), active=[0], margin=0)
        return pn.Column(
            self.message_pane,
            pn.Row(self.figure, sizing_mode="scale_both"),
            settings,
            scroll=True,
        )

    def _get_images_function(self, ra, dec, radius, **kwargs):
        """Retrieve images for the given sky position and cutout radius.

        Must be implemented by subclasses.

        Parameters
        ----------
        ra, dec : float
            Sky coordinates in degrees.
        radius : float
            Cutout radius in arcsec.

        Returns
        -------
        images : list of 2-D np.ndarray
        wcs : list of astropy.wcs.WCS
        band_names : list of str
        """
        raise NotImplementedError

    @param.depends("radius", watch=True)
    def _on_radius_change(self):
        """Sync param watcher: schedules the async fetch when radius changes."""
        asyncio.ensure_future(self._get_data())

    async def _get_data(self):
        """Fetch images in a background thread without blocking the UI.

        Uses asyncio.to_thread so the Tornado event loop stays responsive.
        All UI mutations happen after the await, still on the event loop, so
        no curdoc / add_next_tick_callback machinery is needed.
        """
        self.ra, self.dec = self._get_ra_dec()
        if self.ra is None or self.dec is None:
            self.get_error_panel("Image unavailable", "Missing RA or DEC for this source")
            return

        self.message_pane.object = "## Loading..."
        self.message_pane.visible = True

        try:
            result = await asyncio.to_thread(
                self._get_images_function, self.ra, self.dec, self.radius
            )
        except Exception as e:
            self.get_error_panel("Image retrieval failed", str(e))
            return

        if result is None:
            self.get_error_panel("Image retrieval failed", "No data returned")
            return

        images, wcs, band_names = result
        color_bands = (list(self.color_bands.values())
                       if isinstance(self.color_bands, dict)
                       else self.color_bands)
        try:
            self.image_container = ImageVisaulizationClass(
                images, wcs, band_names,
                color_image=self.color_image,
                color_bands=color_bands,
                target_wcs=self.reference_wcs,
            )
        except Exception as e:
            self.get_error_panel("Image processing failed", str(e))
            return

        available_bands = list(band_names) if not isinstance(band_names, list) else band_names[:]
        if self.color_image:
            available_bands.append(self.image_container.color_name)

        self.param["filter"].objects = available_bands
        if self.filter not in available_bands:
            self.filter = available_bands[0] if available_bands else ""

        self.message_pane.visible = False
        self._update_plot()

    def panel(self):
        """Schedule the initial data fetch and return the layout.

        panel() is called by Panel's server per-connection, inside the running
        Tornado event loop, so ensure_future works immediately.
        """
        asyncio.ensure_future(self._get_data())
        return super().panel()

    # ------------------------------------------------------------------
    # Plot rendering
    # ------------------------------------------------------------------

    @param.depends("filter", "clipping", "stretching", "scaling",
                   "source_coordinates", "external_coordinates",
                   "gamma_r", "gamma_g", "gamma_b",
                   "clipping_r", "clipping_g", "clipping_b",
                   watch=True)
    def _update_plot(self):
        """Rebuild the HoloViews overlay from current parameters."""
        if self.image_container is None:
            return

        image_to_plot = self._get_image_to_plot()
        image = self.plot_image(image_to_plot)
        self._attach_stream(image, self._light_profile_callback)
        contours = self.plot_contours(image_to_plot)
        overplotted_coordinates = self.plot_coordinates()
        self.figure.object = hv.Overlay([image] + contours + overplotted_coordinates).opts(
            active_tools=["tap"]
        )

    def plot_image(self, image, cmap="grey"):
        """Wrap a numpy array in an hv.Image or hv.RGB element."""
        self.image_height, self.image_width = image.shape[:2]
        bounds = (0, 0, self.image_width, self.image_height)
        opts = dict(active_tools=["tap"], toolbar=None, padding=0, border=0,
                    framewise=True, xaxis=None, yaxis=None)
        if len(image.shape) == 3:
            return hv.RGB(image[::-1, ...], bounds=bounds).opts(**opts)
        return hv.Image(image[::-1, ...], bounds=bounds).opts(cmap=cmap, **opts)

    def _get_image_to_plot(self):
        """Return the scaled/stretched numpy array for the currently selected band."""
        color_name = self.image_container.color_name
        if self.filter != color_name:
            low_clip, high_clip = self.clipping
        else:
            low_clip = (self.clipping_r[0], self.clipping_g[0], self.clipping_b[0])
            high_clip = (self.clipping_r[1], self.clipping_g[1], self.clipping_b[1])

        return self.image_container.get_plot_data(
            band=self.filter,
            stretch=self.stretching,
            stretch_interval="Asymmetric",
            low_clip=low_clip,
            high_clip=high_clip,
            gamma_color=(self.gamma_r, self.gamma_g, self.gamma_b),
            scale_method=self.scaling,
        )

    def plot_contours(self, image):
        """Return a list of hv contour elements to overlay. Override to add contours."""
        return []

    def plot_coordinates(self):
        """Return a list of hv.Points elements to overlay on the image."""
        overplotted_coordinates = []

        if self.external_coordinates and hasattr(self, "stored_external_coordinates"):
            for dataset, coords in self.stored_external_coordinates.items():
                N = len(coords["ra"])
                colors = plt.get_cmap("gist_rainbow", max(N, 2))
                marker = "+" if dataset == "DESI" else "*"
                label = "Euclid Spectra" if dataset == "EuclidSpec" else f"{dataset} Spectra"
                xpix, ypix = self.image_container.world2pixel(
                    ra=coords["ra"], dec=coords["dec"], band=self.filter)
                for i, (x, y) in enumerate(zip(xpix, ypix)):
                    if (0 <= x < self.image_width) and (0 <= y < self.image_height):
                        points = hv.Points([(x, y)], label=label if i == 0 else "")
                        overplotted_coordinates.append(
                            points.opts(color=colors(i), marker=marker, size=20))

        if self.source_coordinates and hasattr(self, "ra") and self.ra is not None:
            x, y = self.image_container.world2pixel(
                ra=self.ra, dec=self.dec, band=self.filter)
            if (0 <= x < self.image_width) and (0 <= y < self.image_height):
                overplotted_coordinates.append(
                    hv.Points([(x, y)], label="Source").opts(
                        color="blue", marker="+", size=30))

        return overplotted_coordinates

    # ------------------------------------------------------------------
    # Tap stream and light profile
    # ------------------------------------------------------------------

    def _attach_stream(self, image, stream_function):
        """Attach a Tap stream to an hv element, safely replacing any prior one."""
        if self.image_stream is not None:
            try:
                if self._tap_watcher is not None:
                    self.image_stream.param.unwatch(self._tap_watcher)
                    self._tap_watcher = None
                self.image_stream.source = None
            except Exception:
                pass
            self.image_stream = None

        self.image_stream = hv.streams.Tap(source=image, x=np.nan, y=np.nan)
        self._tap_watcher = self.image_stream.param.watch(stream_function, ["x"])

    def _light_profile_callback(self, event):
        """Handle a tap event: compute pixel position and render the light profile."""
        if self.image_stream.x is None or self.image_stream.y is None:
            return
        col = int(round(self.image_stream.x))
        row = (self.image_height - 1) - int(round(self.image_stream.y))
        row = max(0, min(row, self.image_height - 1))
        col = max(0, min(col, self.image_width - 1))
        if self.filter != self.image_container.color_name:
            self._plot_light_profile(row, col)

    def _plot_light_profile(self, row, col):
        """Render horizontal and vertical flux profiles through a tapped pixel."""
        scaled_image = self._get_image_to_plot()

        def get_curve(values, idx, xlabel):
            curve = hv.Curve(values, kdims="index", vdims="value").opts(
                toolbar=None, padding=0.0, border=1, framewise=True,
                active_tools=[], xlabel=xlabel, yaxis=None,
                ylim=(min(0, np.nanmin(values)), np.nanmax(values) * 1.1),
                color="black",
            )
            line = hv.VLine(idx).opts(color="red", line_width=1, line_dash="dotted")
            return hv.Overlay([curve, line]).opts(responsive=True, toolbar=None)

        plot_x = get_curve(scaled_image[row, :], col, "X coordinate")
        plot_y = get_curve(scaled_image[:, col], row, "Y coordinate")
        self._attach_stream(plot_x, lambda event: self._update_plot())
        self.figure.object = (plot_x + plot_y).cols(1).opts(sizing_mode="stretch_both")

    # ------------------------------------------------------------------
    # External coordinate subscriptions
    # ------------------------------------------------------------------

    def _add_external_coordinates(self, coordinates, dataset):
        """Store coordinates published by another panel and optionally refresh the plot.

        Parameters
        ----------
        coordinates : dict with keys "ra" and "dec" (lists of floats)
        dataset : str
            Key identifying the source catalogue (e.g. "DESI", "SDSS").
        """
        if not coordinates or "ra" not in coordinates or "dec" not in coordinates:
            print(f"[{self.__class__.__name__}] Invalid coordinates for dataset '{dataset}'")
            return
        if not hasattr(self, "stored_external_coordinates"):
            self.stored_external_coordinates = {}
        self.stored_external_coordinates[dataset] = {
            "ra": coordinates["ra"],
            "dec": coordinates["dec"],
        }
        if self.external_coordinates:
            self._update_plot()

    def _manage_subscriptions(self):
        """Subscribe to coordinate keys published by panels listed in panels_to_subscribe."""
        if self.panels_to_subscribe is None:
            return
        for panel in self.panels_to_subscribe:
            key = f"{panel}_coordinates"
            self._subscribe_to_shared(
                key,
                lambda coords, p=panel: self._add_external_coordinates(coords, p),
            )
            existing = shared_data.get_data(key)
            if existing:
                self._add_external_coordinates(existing, panel)

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def _get_default_settings(self):
        return {
            "filter": "",
            "radius": 5.0,
            "stretching": "Linear",
            "clipping": (0, 1),
            "scaling": "MinMax",
            "gamma_r": 1,
            "gamma_g": 1,
            "gamma_b": 1,
            "clipping_r": (0, 1),
            "clipping_g": (0, 1),
            "clipping_b": (0, 1),
            "source_coordinates": False,
            "external_coordinates": False,
        }

    def _initialise_settings_dictionary(self):
        settings = config.settings.setdefault(f"{self.panel_name}", {})
        for key, value in self._get_default_settings().items():
            if key not in settings:
                self._update_settings_dictionary(key, value)

    def _update_settings_dictionary(self, key, value):
        config.settings[f"{self.panel_name}"][key] = value

    def _update_settings_dictionary_cb(self, event):
        self._update_settings_dictionary(event.name, event.new)


class EuclidImagePanel(AstroImagePanel):
    """AstroImagePanel subclass that retrieves Euclid image cutouts via EuclidCutoutsClass.

    Manages a single EuclidCutoutsClass instance across source/radius changes:
    - First call to _get_images_function initialises the instance.
    - Subsequent calls reuse it via reset_data() to avoid re-initialising the ESA client.

    An extra settings section exposes environment switching and, for non-PDR
    environments, credential inputs.
    """

    euclid_environment = param.Selector(
        objects=["PDR", "IDR", "OTF"],
        default="PDR",
        doc="Euclid Science Archive environment",
    )

    _EUCLID_BANDS = ["VIS", "NIR_Y", "NIR_J", "NIR_H"]
    _DEFAULT_COLOR_BANDS = {"r": "VIS", "g": "NIR_J", "b": "NIR_Y"}

    def __init__(self,
                 data,
                 src,
                 close_button,
                 panel_name="Euclid_Cutout",
                 euclid_bands=None,
                 color_bands=None,
                 check_moc_coverage=False,
                 panels_to_subscribe=None):

        # Must be set before super().__init__ because _get_data is called there.
        self.euclid_bands = euclid_bands or self._EUCLID_BANDS
        self.check_moc_coverage = check_moc_coverage
        self.euclid_object = None

        super().__init__(
            data, src, close_button,
            panel_name=panel_name,
            color_image=True,
            color_bands=color_bands or self._DEFAULT_COLOR_BANDS,
            panels_to_subscribe=panels_to_subscribe,
        )

    # ------------------------------------------------------------------
    # Settings panel — extend parent with Euclid-specific controls
    # ------------------------------------------------------------------

    def _initialise_settings_panel(self):
        super()._initialise_settings_panel()

        self.environment_selector = pn.widgets.Select.from_param(
            self.param.euclid_environment,
            name="ESA Environment",
            width=150,
        )

        self.user_input = pn.widgets.TextInput(
            name="Username",
            placeholder="Euclid Science Archive username",
            sizing_mode="stretch_width",
        )
        self.password_input = pn.widgets.PasswordInput(
            name="Password",
            placeholder="Euclid Science Archive password",
            sizing_mode="stretch_width",
        )
        self.connect_button = pn.widgets.Button(
            name="Connect",
            button_type="primary",
            max_height=35,
            max_width=100,
        )

        self.login_column = pn.Column(
            self.user_input,
            self.password_input,
            self.connect_button,
            visible=False,
        )

        self.environment_selector.param.watch(
            self._on_environment_changed, "value")
        self.connect_button.on_click(self._on_connect_clicked)

    def _get_settings_sections(self):
        sections = super()._get_settings_sections()
        sections.append(
            ("Euclid Archive", pn.Column(self.environment_selector, self.login_column))
        )
        return sections

    # ------------------------------------------------------------------
    # Environment / login callbacks
    # ------------------------------------------------------------------

    def _on_environment_changed(self, event):
        self.login_column.visible = event.new != "PDR"

    def _on_connect_clicked(self, event):
        if self.euclid_object is None:
            return
        env = self.euclid_environment
        user = self.user_input.value or None
        password = self.password_input.value or None
        try:
            self.euclid_object.change_environment(env, user=user, password=password)
            print(f"[{self.__class__.__name__}] Switched to environment: {env}")
        except Exception as e:
            self.get_error_panel("Environment change failed", str(e))

    # ------------------------------------------------------------------
    # Data retrieval
    # ------------------------------------------------------------------

    def _get_images_function(self, ra, dec, radius):
        """Manage the EuclidCutoutsClass lifecycle and return image arrays.

        On the first call the instance is created, which also initialises the
        ESA client. On subsequent calls reset_data() updates the coordinates
        without rebuilding the client.

        Returns
        -------
        images : list of 2-D np.ndarray, one per successfully downloaded band
        wcs_list : list of astropy.wcs.WCS
        available_bands : list of str
        """
        if self.euclid_object is None:
            self.euclid_object = EuclidCutoutsClass(
                bands_to_retrieve=self.euclid_bands,
                client=shared_data.get_data("Euclid_client"),
                check_moc_coverage=self.check_moc_coverage,
            )

        self.euclid_object.get_cutouts(ra, dec, radius)

        if self.euclid_object.error_tracker.has_error:
            raise RuntimeError("EuclidCutoutsClass reported an error during get_cutouts")

        available_bands = [b for b in self.euclid_bands
                           if b in self.euclid_object.data and
                           self.euclid_object.data[b] is not None]

        if not available_bands:
            raise RuntimeError("No Euclid bands were retrieved successfully")

        images = [self.euclid_object.data[b] for b in available_bands]
        wcs_list = [self.euclid_object.wcs[b] for b in available_bands]
        return images, wcs_list, available_bands


    def _save_data_to_fits(self, directory_path="data/saved_sources"):
        if self.euclid_object is not None:
            try:
                self.euclid_object.export_cutouts_to_fits(
                    bands_to_export=self.euclid_bands,
                    directory_path=directory_path,
                )
            except Exception as e:
                print(f"[{self.__class__.__name__}] Could not export FITS: {e}")
