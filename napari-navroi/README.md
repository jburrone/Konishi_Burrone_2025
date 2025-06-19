# NavROI

A plugin to visualize results of image registrations across scales and modalities, used in [`a preprint`](https://www.biorxiv.org/content/10.1101/2024.11.29.626074v1) which is accepted in Journal of Microscopy.

----------------------------------

## Installation
Unzip `napari-navroi.zip`.
```
napari-navroi/
│
├── napari_navroi/
│   ├── __pycache__
│   ├── __init__.py
│   ├── _utils.py
│   ├── _widget.py
│   ├── napari.yaml
├── demo_affinder.gif
├── demo_segreg.gif
├── envrionment.yaml
├── LICENSE
├── MANIFEST.in
├── pyproject.toml
├── README.md
├── setup.cfg
```

Create a conda environment with napari by typing:
```
(base) C: /your/path/to/napari-navroi/ > conda env create -f environment.yaml
```

Activate the environment:
```
(base) C: /your/path/to/napari-navroi/ > conda activate napari-env
```

Simply use pip install and run napari to test the plugin in the same environment:
```
(napari-env) C: /your/path/to/napari-navroi/ > pip install -e .
(napari-env) C: /your/path/to/napari-navroi/ > napari
```

## Usage
Open `napari` and find `NavROI` plugin.
```
(napari-env) C: /any/path/ > napari
```
### Visualizing results from landmark-based registration plugin
1. Prepare image files in tif or zarr format.
2. Estimate the relative positions between images using [affinder](https://www.napari-hub.org/plugins/affinder) (Plugin > affinder).
3. Prepare a config file in a yaml format based on the position estimation. 
4. Open this plugin (Plugin > NavROI).

![demo_affinder](./assets/demo_affinder.gif)

### Visualizing results from `segmentation-based image registration (SegReg)` tool (Fig 1(B) of [`a preprint`](https://www.biorxiv.org/content/10.1101/2024.11.29.626074v1))
1. Prepare image files in tif or zarr format.
2. Prepare a config file in a yaml format based on [`SegReg`](https://github.com/jburrone/Konishi_Burrone_2025/segreg/).
3. Open this plugin (Plugin > NavROI).

![demo_segreg](./assets/demo_segreg.gif)

### Data
Data can be downloaded from [BioImage Archive](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD2077) and [EMPIRE](https://www.ebi.ac.uk/empiar/EMPIAR-xxx/).

### TIPS: [napari](https://napari.org/stable/) functions useful for this plugin
- Change opacity and color map of the layer
  1. Click the layer in the panel of the layer list.
  2. Tweat the parameters in the layer panel.
- Maximum intensity projection of the layer
  1. Right-click the layer in the panel of the layer list.
  2. Select `Projection` > `Maximum Projection`.
- [Plane rendering](https://napari.org/stable/gallery/3Dimage_plane_rendering.html)
  1. Click the layer in the panel of the layer list.
  2. Drag while pressing `Shift` key.
- Make ROI
  1. Click `New Shape Layer` in the layer list.
  2. Draw ROIs using the parameters in the panel of the layer controls.
  3. Save the ROIs as a CSV file by `File` > `Save Selected Layer`.

## License
Distributed under the terms of the [MIT](https://opensource.org/license/MIT) license,
"NavROI" is free and open source software.

## Contributing
Contributions are very welcome. 
If you encounter any problems, please send an email to: juan.burrone@kcl.ac.uk along with a detailed description. 

## Citing
If you use this plugin in your research, please use the following BibTeX entry.
```
@article{konishi2024navroi,
  title={Semi-automated navigation for efficient targeting of electron tomography to regions of interest in volume correlative light and electron microscopy},
  author={Kohki Konishi, Guilherme Neves, Matthew Russell, Masafumi Mimura, Juan Burrone, Roland Fleck},
  journal={bioRxiv: 2024.11.29.626074},
  year={2024}
}
```