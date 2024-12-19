# Utilities

This folder contains the utilities that could assist in the usage of Tandem or auxilliary tools for pre/post processing.

## 1. HDF5 to GMSH Mesh Conversion (convert_h5_to_msh.py)

This python script converts HDF5 (`.h5`) format into a GMSH (`.msh`) format. This can be used with a SeisSol/Sim Modeler mesh that has complex geometrical properties that could be imported into Tandem for simulations.

---


### Usage

#### Example Usage

Convert an HDF5 mesh file to GMSH format:

```bash
python convert_h5_to_msh.py \
  --input_mesh_file input_mesh.h5 \
  --tag_map_yaml_file tag_mapping.yaml \
  --output_mesh_file output_mesh.msh
```
#### Command-Line Arguments

The script accepts the following command-line arguments:

| Argument | Description |
|----------|-------------|
| `--input_mesh_file` (required) | Path to the input HDF5 (`.h5`) mesh file. |
| `--tag_map_yaml_file` (required) | Path to the YAML file mapping SeisSol boundary tags to Tandem boundary tags. |
| `--output_mesh_file` (optional) | Path to the output GMSH (`.msh`) file. If not provided, the output filename will be derived from the input filename. |


### Features

- Converts `.h5` mesh files to `.msh` files in GMSH version 2.2 format.
- Reads and validates mapping from SeisSol boundary tags to Tandem boundary tags via a YAML file.
- Handles boundary condition decoding and lower-order element extraction.

---

### Requirements

The script depends on the following Python libraries:

- `h5py`
- `meshio`
- `numpy`
- `argparse`
- `yaml`
- `os`
- `warnings`

Make sure these libraries are installed before running the script. You can install them using pip:

```bash
pip install h5py meshio numpy pyyaml

```
In case you are on a remote system without sudo access, you can also create a virtual environment within utils and then use pip and the corresponding python file in the virtual environment binary folder to run the script.

```bash

python3 -m venv .
bin/pip install h5py meshio numpy pyyaml
bin/python convert_h5_to_msh.py \
  --input_mesh_file input_mesh.h5 \
  --tag_map_yaml_file tag_mapping.yaml \
  --output_mesh_file output_mesh.msh
```

### YAML file

The user-defined mapping yaml file, that defines the mapping from a SeisSol tag to a Tandem tag, can be created and the contents have to look as follows:

```tag_mapping.yaml```
```yaml
1 : 1
3 : 4
7 : 8
```

Here the first number on each row before the colon denotes a SeisSol tag and each second number after the colon denotes the Tandem tag that the converter should map the SeisSol tag to. For reference, the SeisSol and Tandem tag values are listed on the following links:

* https://seissol.readthedocs.io/en/latest/PUML-mesh-format.html
* https://tandem.readthedocs.io/en/latest/first-model/mesh.html
