from __future__ import annotations

import contextlib
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Literal, Optional

import numpy as np
import torch
import viser
import viser.theme
import viser.transforms as vtf
from typing_extensions import assert_never

from nerfstudio.viewer.viewer import Viewer
from nerfstudio.configs import base_config as cfg
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.viewer.control_panel import ControlPanel
from nerfstudio.viewer.export_panel import populate_export_tab
from nerfstudio.viewer.render_panel import populate_render_tab
from nerfstudio.viewer.render_state_machine import RenderAction, RenderStateMachine
from nerfstudio.viewer.utils import CameraState, parse_object
from nerfstudio.viewer.viewer_elements import ViewerControl, ViewerElement
from nerfstudio.viewer_legacy.server import viewer_utils

if TYPE_CHECKING:
    from nerfstudio.engine.trainer import Trainer

VISER_NERFSTUDIO_SCALE_RATIO: float = 10.0


class SatelliteSplatViewer(Viewer):

    def __init__(
            self,
            config: cfg.ViewerConfig,
            log_filename: Path,
            datapath: Path,
            pipeline: Pipeline,
            trainer: Optional[Trainer] = None,
            train_lock: Optional[threading.Lock] = None,
            share: bool = False,
    ):
        self.ready = False  # Set to True at end of constructor.
        self.config = config
        self.trainer = trainer
        self.last_step = 0
        self.train_lock = train_lock
        self.pipeline = pipeline
        self.log_filename = log_filename
        self.datapath = datapath.parent if datapath.is_file() else datapath
        self.include_time = self.pipeline.datamanager.includes_time

        if self.config.websocket_port is None:
            websocket_port = viewer_utils.get_free_port(default_port=self.config.websocket_port_default)
        else:
            websocket_port = self.config.websocket_port
        self.log_filename.parent.mkdir(exist_ok=True)

        # viewer specific variables
        self.output_type_changed = True
        self.output_split_type_changed = True
        self.step = 0
        self.train_btn_state: Literal["training", "paused", "completed"] = (
            "training" if self.trainer is None else self.trainer.training_state
        )
        self._prev_train_state: Literal["training", "paused", "completed"] = self.train_btn_state
        self.last_move_time = 0
        # track the camera index that last being clicked
        self.current_camera_idx = 0

        self.viser_server = viser.ViserServer(host=config.websocket_host, port=websocket_port)
        # Set the name of the URL either to the share link if available, or the localhost
        share_url = None
        if share:
            share_url = self.viser_server.request_share_url()
            if share_url is None:
                print("Couldn't make share URL!")

        if share_url is not None:
            self.viewer_info = [f"Viewer at: http://localhost:{websocket_port} or {share_url}"]
        elif config.websocket_host == "0.0.0.0":
            # 0.0.0.0 is not a real IP address and was confusing people, so
            # we'll just print localhost instead. There are some security
            # (and IPv6 compatibility) implications here though, so we should
            # note that the server is bound to 0.0.0.0!
            self.viewer_info = [f"Viewer running locally at: http://localhost:{websocket_port} (listening on 0.0.0.0)"]
        else:
            self.viewer_info = [f"Viewer running locally at: http://{config.websocket_host}:{websocket_port}"]

        buttons = (
            viser.theme.TitlebarButton(
                text="Getting Started",
                icon=None,
                href="https://nerf.studio",
            ),
            viser.theme.TitlebarButton(
                text="Github",
                icon="GitHub",
                href="https://github.com/rocketzhong/satellite_splat",
            ),
            viser.theme.TitlebarButton(
                text="Documentation",
                icon="Description",
                href="https://docs.nerf.studio",
            ),
        )
        image = viser.theme.TitlebarImage(
            image_url_light="https://www.gdut.edu.cn/images2020/logo.png",
            image_url_dark="https://www.gdut.edu.cn/images2020/logo.png",
            image_alt="GDUT Logo",
            href="https://www.gdut.edu.cn/",
        )
        titlebar_theme = viser.theme.TitlebarConfig(buttons=buttons, image=image)
        self.viser_server.gui.configure_theme(
            titlebar_content=titlebar_theme,
            control_layout="collapsible",
            dark_mode=True,
            brand_color=(255, 211, 105),
        )

        self.render_statemachines: Dict[int, RenderStateMachine] = {}
        self.viser_server.on_client_disconnect(self.handle_disconnect)
        self.viser_server.on_client_connect(self.handle_new_client)

        # Populate the header, which includes the pause button, train cam button, and stats
        self.pause_train = self.viser_server.gui.add_button(
            label="Pause Training", disabled=False, icon=viser.Icon.PLAYER_PAUSE_FILLED
        )
        self.pause_train.on_click(lambda _: self.toggle_pause_button())
        self.pause_train.on_click(lambda han: self._toggle_training_state(han))
        self.resume_train = self.viser_server.gui.add_button(
            label="Resume Training", disabled=False, icon=viser.Icon.PLAYER_PLAY_FILLED
        )
        self.resume_train.on_click(lambda _: self.toggle_pause_button())
        self.resume_train.on_click(lambda han: self._toggle_training_state(han))
        if self.train_btn_state == "training":
            self.resume_train.visible = False
        else:
            self.pause_train.visible = False

        # Add buttons to toggle training image visibility
        self.hide_images = self.viser_server.gui.add_button(
            label="Hide Train Cams", disabled=False, icon=viser.Icon.EYE_OFF, color=None
        )
        self.hide_images.on_click(lambda _: self.set_camera_visibility(False))
        self.hide_images.on_click(lambda _: self.toggle_cameravis_button())
        self.show_images = self.viser_server.gui.add_button(
            label="Show Train Cams", disabled=False, icon=viser.Icon.EYE, color=None
        )
        self.show_images.on_click(lambda _: self.set_camera_visibility(True))
        self.show_images.on_click(lambda _: self.toggle_cameravis_button())
        self.show_images.visible = False
        mkdown = self.make_stats_markdown(0, "0x0px")
        self.stats_markdown = self.viser_server.gui.add_markdown(mkdown)
        tabs = self.viser_server.gui.add_tab_group()
        control_tab = tabs.add_tab("Control", viser.Icon.SETTINGS)
        with control_tab:
            self.control_panel = ControlPanel(
                self.viser_server,
                self.include_time,
                VISER_NERFSTUDIO_SCALE_RATIO,
                self._trigger_rerender,
                self._output_type_change,
                self._output_split_type_change,
                default_composite_depth=self.config.default_composite_depth,
            )
        config_path = self.log_filename.parents[0] / "config.yml"
        with tabs.add_tab("Render", viser.Icon.CAMERA):
            self.render_tab_state = populate_render_tab(
                self.viser_server, config_path, self.datapath, self.control_panel
            )

        with tabs.add_tab("Export", viser.Icon.PACKAGE_EXPORT):
            populate_export_tab(self.viser_server, self.control_panel, config_path, self.pipeline.model)

        # Keep track of the pointers to generated GUI folders, because each generated folder holds a unique ID.
        viewer_gui_folders = dict()

        def prev_cb_wrapper(prev_cb):
            # We wrap the callbacks in the train_lock so that the callbacks are thread-safe with the
            # concurrently executing render thread. This may block rendering, however this can be necessary
            # if the callback uses get_outputs internally.
            def cb_lock(element):
                with self.train_lock if self.train_lock is not None else contextlib.nullcontext():
                    prev_cb(element)

            return cb_lock

        def nested_folder_install(folder_labels: List[str], prev_labels: List[str], element: ViewerElement):
            if len(folder_labels) == 0:
                element.install(self.viser_server)
                # also rewire the hook to rerender
                prev_cb = element.cb_hook
                element.cb_hook = lambda element: [prev_cb_wrapper(prev_cb)(element), self._trigger_rerender()]
            else:
                # recursively create folders
                # If the folder name is "Custom Elements/a/b", then:
                #   in the beginning: folder_path will be
                #       "/".join([] + ["Custom Elements"]) --> "Custom Elements"
                #   later, folder_path will be
                #       "/".join(["Custom Elements"] + ["a"]) --> "Custom Elements/a"
                #       "/".join(["Custom Elements", "a"] + ["b"]) --> "Custom Elements/a/b"
                #  --> the element will be installed in the folder "Custom Elements/a/b"
                #
                # Note that the gui_folder is created only when the folder is not in viewer_gui_folders,
                # and we use the folder_path as the key to check if the folder is already created.
                # Otherwise, use the existing folder as context manager.
                folder_path = "/".join(prev_labels + [folder_labels[0]])
                if folder_path not in viewer_gui_folders:
                    viewer_gui_folders[folder_path] = self.viser_server.gui.add_folder(folder_labels[0])
                with viewer_gui_folders[folder_path]:
                    nested_folder_install(folder_labels[1:], prev_labels + [folder_labels[0]], element)

        with control_tab:
            from nerfstudio.viewer_legacy.server.viewer_elements import ViewerElement as LegacyViewerElement

            if len(parse_object(pipeline, LegacyViewerElement, "Custom Elements")) > 0:
                from nerfstudio.utils.rich_utils import CONSOLE

                CONSOLE.print(
                    "Legacy ViewerElements detected in model, please import nerfstudio.viewer.viewer_elements instead",
                    style="bold yellow",
                )
            self.viewer_elements = []
            self.viewer_elements.extend(parse_object(pipeline, ViewerElement, "Custom Elements"))
            for param_path, element in self.viewer_elements:
                folder_labels = param_path.split("/")[:-1]
                nested_folder_install(folder_labels, [], element)

            # scrape the trainer/pipeline for any ViewerControl objects to initialize them
            self.viewer_controls: List[ViewerControl] = [
                e for (_, e) in parse_object(pipeline, ViewerControl, "Custom Elements")
            ]
        for c in self.viewer_controls:
            c._setup(self)

        # Diagnostics for Gaussian Splatting: where the points are at the start of training.
        # This is hidden by default, it can be shown from the Viser UI's scene tree table.
        if isinstance(pipeline.model, SplatfactoModel):
            self.viser_server.scene.add_point_cloud(
                "/gaussian_splatting_initial_points",
                points=pipeline.model.means.numpy(force=True) * VISER_NERFSTUDIO_SCALE_RATIO,
                colors=(255, 0, 0),
                point_size=0.01,
                point_shape="circle",
                visible=False,  # Hidden by default.
            )
        self.ready = True
