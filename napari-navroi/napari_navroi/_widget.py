import os
from pathlib import Path
import yaml

import numpy as np
from OpenGL import GL as gl
from skimage import io

from typing import TYPE_CHECKING

from magicgui import widgets
from qtpy.QtWidgets import (
    QFileDialog,
    QLineEdit,
    QPushButton,
    QVBoxLayout, 
    QWidget,
)

from ._utils import (
    affinder_matrix, 
    apply_affine_image, 
    create_box,
    get_affine_params,
    is_3ch_image, 
    is_gray_image,
    is_zstack_image,
    read_image_pyramid,
    Rx, Ry, Rz,
    transform_point_to_moving_frame,
    )

def set_view_params(dict):
    dict["tz_segreg"] = 0
    dict["ty_segreg"] = 0
    dict["tx_segreg"] = 0
    dict["sx_segreg"] = 1
    dict["sy_segreg"] = 1
    dict["dx"] = 0
    dict["dy"] = 0
    return

if TYPE_CHECKING:
    import napari

class NavROI(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        """Initialize the widget."""
        super().__init__()
        self.viewer = viewer

        # Create main layout
        self.main_layout = QVBoxLayout(self)

        # Create button to select file to be read
        self.search_button = QPushButton("Select config file")
        self.search_button.move(0, 0)
        self.search_button.clicked.connect(self._open_file_dialog)
        self.main_layout.addWidget(self.search_button)

        # Save path display
        self.save_path = QLineEdit()
        self.save_path.setReadOnly(True)
        self.main_layout.addWidget(self.save_path)

        # Create mosaic arrangement widgets
        self.zmag = widgets.FloatSlider(
            value=1, min=1, max=3, label='Z x: ')
        self.box = widgets.CheckBox(
            value=True, description='Show boxes of image boundaries.')
        self.reg_method = widgets.ComboBox(
            choices=['affinder','segreg'],
            value='segreg', 
            label='Registration:',
            tooltip='affinder or segreg.')
        self.button_show = widgets.Button(text='Show')

        # organize widgets        
        self.mosaic_widgets = [
                            self.zmag,
                            self.reg_method,
                            self.box,
                            self.button_show,
                            ]

        self.container = widgets.VBox(widgets=\
                            self.mosaic_widgets
                            )

        self.container.native.setMinimumWidth = 50

        self.main_layout.addWidget(self.container.native)

        # connect callbacks
        self.button_show.clicked.connect(self._visualize)


    def _open_file_dialog(self):
        """
        Opens a `QFileDialog` to select a directory or file with specific extensions.
        """
        options = QFileDialog.Options()
        dialog = QFileDialog(self, "Select config file", options=options)
        dialog.setFileMode(QFileDialog.AnyFile)
        # Filter files by extension
        dialog.setNameFilter(
            "All files (*.yaml)"
        )
        
        if dialog.exec_():
            selected_file = dialog.selectedFiles()[0]
            self.save_path.setText(selected_file)


    def _visualize(self):
        """
        Arrange tiles in the viewer according to the selected mosaic arrangement.
        """

        # Load the configuration file
        zscale_display = self.mosaic_widgets[0].value

        config_file = self.save_path.text()
        with open(config_file) as f:
            configs = yaml.safe_load(f)

        if configs["images"][0]["xyres_um"] > 0:
            scaling_of_napari_ref = configs["images"][0]["xyres_um"] / configs["images"][-1]["xyres_um"]
        else:
            scaling_of_napari_ref = 1

        for dict in configs["images"]:
            filepath = Path(os.path.join(os.path.dirname(config_file), dict["filepath"]))

            if filepath.suffix == '.tif' or filepath.suffix == '.png':
                img = io.imread(filepath)
                if is_gray_image(img) == True or is_3ch_image(img) == True:
                    img = img[np.newaxis, ...]
                img_shape = img.shape
                if img.shape[-1] > gl.GL_MAX_TEXTURE_SIZE:
                    print(f"It is recommended to change the following file format to Zarr: ", dict["filepath"])
            elif filepath.suffix == '.zarr':
                img = read_image_pyramid(filepath)
                img_shape = img[0].shape
            else:
                raise ValueError("Unsupported file format")

            multiscale = filepath.suffix == '.zarr'

            if img_shape[-1] > gl.GL_MAX_TEXTURE_SIZE and not multiscale:
                print(f"It is recommended to change the following file format to Zarr: {filepath}")

            if dict["napari_ref"] == 'Y':
                set_view_params(dict)
                ul_x_mov, ul_y_mov, ul_z_mov = 0, 0, 0
                sx_layer = dict["sx_segreg"] * scaling_of_napari_ref
                sy_layer = dict["sy_segreg"] * scaling_of_napari_ref
                sz_layer = zscale_display * scaling_of_napari_ref

            elif dict["napari_ref"] == 'N':
                if self.mosaic_widgets[1].value == 'segreg':
                    c_x_mov, c_y_mov, _ = transform_point_to_moving_frame(0, 
                                                    0, 
                                                    dict["x_offmov"], 
                                                    dict["y_offmov"], 
                                                    dict["size"], 
                                                    dict["tz_segreg"], 
                                                    dict["x_segreg"], 
                                                    dict["y_segreg"], 
                                                    dict["sx_segreg"], 
                                                    dict["sy_segreg"])
                    c_z_mov = dict["z_segreg"]

                    registration_center = np.asarray(((np.abs(dict["x_offref"]) + dict["size"] // 2) * dict["sx_segreg"], 
                                                    (np.abs(dict["y_offref"]) + dict["size"] // 2) * dict["sy_segreg"], 0))
                    r = Rx(float(dict["tx_segreg"])) @ Ry(float(dict["ty_segreg"])) @ Rz(float(dict["tz_segreg"])) @ registration_center
                    ul_x_mov, ul_y_mov, ul_z_mov = -r[0] + c_x_mov, -r[1] + c_y_mov, -r[2] + c_z_mov

                elif self.mosaic_widgets[1].value == 'affinder':
                    affine = affinder_matrix(Path(os.path.join(os.path.dirname(config_file), dict["affinder_output"])))
                    params = get_affine_params(affine)
                    img = apply_affine_image(img[0], affine, 3, [configs["images"][0]["size"], configs["images"][0]["size"]])
                    img = img[np.newaxis, ...]
                    img_shape = img.shape
                    set_view_params(dict)

                sx_layer = dict["sx_segreg"]
                sy_layer = dict["sy_segreg"]
                sz_layer = zscale_display * scaling_of_napari_ref

            ul_x_layer = (ul_x_mov * scaling_of_napari_ref) + dict["dx"]
            ul_y_layer = (ul_y_mov * scaling_of_napari_ref) + dict["dy"]
            ul_z_layer = (ul_z_mov * scaling_of_napari_ref) * zscale_display

            layer = self.viewer.add_image(img, 
                                    name = dict["name"], 
                                    colormap = dict["color"], 
                                    opacity = 0.8, 
                                    multiscale = multiscale, 
                                    rotate = [dict["tx_segreg"], dict["ty_segreg"], dict["tz_segreg"]], 
                                    scale = [sz_layer, sy_layer, sx_layer], 
                                    shear = None, 
                                    translate = [ul_z_layer, ul_y_layer, ul_x_layer]) 
    
            if self.mosaic_widgets[2].value == True:
                box = create_box(img_shape, box_width=9)
                layer = self.viewer.add_image(box, 
                                        name = dict["name"] + "_box", 
                                        colormap = dict["color"], 
                                        opacity = 0.3, 
                                        multiscale = False,
                                        rotate = [dict["tx_segreg"], dict["ty_segreg"], dict["tz_segreg"]], 
                                        scale = [sz_layer, sy_layer, sx_layer], 
                                        shear = None, 
                                        translate = [ul_z_layer, ul_y_layer, ul_x_layer]) 

        filepath = Path(os.path.join(os.path.dirname(config_file), configs["images"][0]["filepath"]))
        img = io.imread(filepath)

        # Check if it's a z-stack image
        if is_zstack_image(img):
            # Define plane parameters (assuming x is normal)
            plane_parameters = {
                'position': (img.shape[0] // 2, img.shape[1] // 2, img.shape[2] // 2), # z, y, x
                'normal': (0, 0, 1), 
                'thickness': 3
                }

            # Extract scaling factors for each dimension from config
            scale_x = scaling_of_napari_ref
            scale_y = scaling_of_napari_ref
            scale_z = zscale_display * scaling_of_napari_ref

            plane_layer = self.viewer.add_image(img, 
                                        name='rendering_plane',
                                        scale=[scale_z, scale_y, scale_x], 
                                        rendering='average',
                                        depiction='plane',
                                        blending='additive',
                                        colormap=configs["images"][0]["color"],
                                        plane=plane_parameters)

        self.viewer.axes.visible = True
        if configs["images"][0]["xyres_um"] == 1:
            self.viewer.scale_bar.visible = True
            self.viewer.scale_bar.unit = 'um'

if __name__ == "__main__":
    # Simple test main function to run the widget in the debugger
    import napari

    viewer = napari.Viewer()
    napari.run()