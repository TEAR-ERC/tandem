import h5py
import meshio
import numpy as np
import argparse
import os
import warnings
import yaml

# Seissol specific vertexing for a particular tetrahedral face.
# https://seissol.readthedocs.io/en/latest/PUML-mesh-format.html
s_vert_seissol = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]]
# All available SeisSol tags to date
# https://seissol.readthedocs.io/en/latest/PUML-mesh-format.html
seissol_tags = {
    0: "SEISSOL_REGULAR_BOUNDARY_TAG",
    1: "SEISSOL_FREE_SURFACE_BOUNDARY_TAG",
    2: "SEISSOL_GRAVITY_BASED_FREE_SURFACE_TAG",
    3: "SEISSOL_DYNAMIC_RUPTURE_BOUNDARY_TAG",
    4: "SEISSOL_DIRICHLET_BOUNDARY_TAG",
    5: "SEISSOL_ABSORBING_BOUNDARY_TAG",
    6: "SEISSOL_PERIODIC_BOUNDARY_TAG",
    7: "SEISSOL_ANALYTICAL_BOUNDARY_TAG",
}
# Add values from 65 to 255 also as dynamic rupture tags
for i in range(65, 256):
    seissol_tags[i] = "SEISSOL_DYNAMIC_RUPTURE_BOUNDARY_TAG"

# All available tandem tags to date
# https://tandem.readthedocs.io/en/latest/first-model/mesh.html
tandem_tags = {
    1: "TANDEM_FREE_SURFACE_BOUNDARY_TAG",
    3: "TANDEM_DYNAMIC_RUPTURE_BOUNDARY_TAG",
    5: "TANDEM_DIRICHLET_BOUNDARY_TAG",
}
# Define the bit pattern in which the boundary tag needs to be decoded
each_face_bc_tag_i8 = np.dtype(
    [("i0", np.int8), ("i1", np.int8), ("i2", np.int8), ("i3", np.int8)]
)


def check_tag_map_validity(seissol_to_tandem_tag_map):
    """
    Check the validity of tags used to define mapping between SeisSol and Tandem.

    Parameters:
    seissol_to_tandem_tag_map [dict]: A dictionary mapping SeisSol boundary tags to Tandem boundary tags.
    """
    seissol_map_tags = seissol_to_tandem_tag_map.keys()
    tandem_map_tags = seissol_to_tandem_tag_map.values()

    for tag in seissol_map_tags:
        if tag not in seissol_tags:
            raise ValueError(
                f"Invalid SeisSol tag provided in tag mapping yaml file: {tag}"
            )

    for tag in tandem_map_tags:
        if tag not in tandem_tags:
            raise ValueError(
                f"Invalid Tandem tag provided in tag mapping yaml file: {tag}"
            )


def load_yaml_mapping(file_path):
    """
    Load the SeisSol to Tandem boundary tag mapping from a YAML file.

    Parameters:
    file_path (str): Path to the YAML file.

    Returns:
    dict: Mapping between SeisSol tags and Tandem tags.
    """
    try:
        with open(file_path, "r") as yaml_file:
            mapping = yaml.safe_load(yaml_file)
            return mapping
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing the tag mapping YAML file: {e}")


def seissol_to_tandem_tag_conversion(seissol_tag, seissol_to_tandem_tag_map):
    """
    Converts SeisSol tag to its corresponding Tandem tag.

    Parameters:
    x (int8): The SeisSol tag value to be converted.
    seissol_to_tandem_tag_map [dict]: A dictionary mapping SeisSol boundary tags to Tandem boundary tagsu

    Returns:
    int8: The corresponding Tandem tag value.
    """
    if seissol_tag in seissol_to_tandem_tag_map:
        tandem_tag = seissol_to_tandem_tag_map[seissol_tag]
    else:
        tandem_tag = seissol_tag
    return np.int8(tandem_tag)


def get_boundary_condition_masks(boundary, seissol_to_tandem_tag_map):
    """
    Get 4 boolean mask for each 32 bit boundary condition to retrieve valid boundary conditions.

    There are a limited set of boundary tags within both Seissol. These 32 bit boundary tags consist
    of information for 4 faces of a tetrahedra in 8 bit formats. This function creates a look up data
    structure for each of these limited set of values.

    Parameters:
    boundary [List[int32]]: The list of boundary condition per element in a 32 bit format.
    seissol_to_tandem_tag_map [dict]: A dictionary mapping SeisSol boundary tags to Tandem boundary tags.

    Returns:
    Dict[int32:[bool]]: A dict where each key is an encoded 32 bit boundary tag and the value is a
    list of bools. True if it has a number > 0 else False.
    """
    # Initialize data structures
    boundary_unique_encodings_i32 = set(boundary)
    boundary_conditions_masks = {}

    # Find a mapping from each boundary condition number to its
    for element_boundary_condition in boundary_unique_encodings_i32:
        # View the cell as the defined structured dtype
        boundary_condition_decoded_i8 = element_boundary_condition.view(
            dtype=each_face_bc_tag_i8
        )
        masking = [x > 0 for x in boundary_condition_decoded_i8]
        for x in boundary_condition_decoded_i8:
            # Handle unsupported boundary conditions with a warning
            if x not in seissol_tags:
                raise KeyError(f"Tag {x} in the h5 file is not a valid SeisSol tag")
            if x not in seissol_to_tandem_tag_map:
                warnings.warn(
                    f"No mapping defined for Seissol Tag: {x} = {seissol_tags[x]}. Writing as it is for now."
                )

        # Store the results
        boundary_conditions_masks[element_boundary_condition] = masking
    return boundary_conditions_masks


def retrieve_lower_order_elements(
    nodes_per_higher_order_element, mask_list, boundary_encoding, s_vert=s_vert_seissol
):
    """
    Infer nodes of lower order elements from Seissol boundary encoding and higher order elements.

    Since .h5 files converted from .msh with pumgen lose lower order elements, we want to retrieve it
    in order to maintain consistency with the .msh file. These elements can be inferred using a
    combination of the nodes of a higher order element and the 4 boundary values decoded from the
    32 bit encoding.

    Parameters:
    values_list [List[List]]: The list of lists of nodes of higher order elements (e.g., [[234, 432, ...], [...], ...]).
    mask_list [List[List]]: The list of boolean mask arrays (e.g., [[False, True, ...], [...], ...]).
    boundary_encoding (list): A list mapping elements in `values_list` to specific masks in `mask_list`.
    s_vert (list): A list of arrays defining vertex mappings within Seissol (e.g., [[0, 2, 1], [0, 1, 3], ...]).

    Returns:
    List[numpy.ndarray]: A list where each element is a numpy.ndarray containing the subset of `nodes_per_higher_order_element`
    corresponding to the masks and `s_vert`. Multiple
    """
    print("Retrieving lower order elements.")
    lower_order_elements = []
    for single_element_nodes, bc in zip(
        nodes_per_higher_order_element, boundary_encoding
    ):
        for true_index in [i for i, val in enumerate(mask_list[bc]) if val]:
            # Use the corresponding row in s_vert for each True index
            selected_indices = s_vert[true_index]

            # Map the indices to values, convert to ndarray, and enforce dtype numpy.int64
            selected_values = np.array(
                [single_element_nodes[i] for i in selected_indices], dtype=np.int64
            )
            lower_order_elements.append(selected_values)
    print("Lower order elements retrieved.")
    return lower_order_elements


def convert_h5_to_msh(h5_file, msh_file, seissol_to_tandem_tag_map):
    """

    Convert an h5 file to a msh file.

    """
    # Extract values from the .h5 mesh file
    with h5py.File(h5_file, "r") as h5:
        # Extract the datasets
        geometry = h5["geometry"][:]  # Node coordinates
        connect = h5["connect"][:]  # Connectivity (elements)
        boundary = h5["boundary"][:]  # Boundary markers (8 bit encoding)
        group = h5["group"][:]  # physical groups
    # Get boolean mask for each 32 bit boundary tag in the boundary list
    boundary_conditions_masks = get_boundary_condition_masks(
        boundary, seissol_to_tandem_tag_map
    )
    # Get details for connectivity of lower order elements
    # (E.g., For 3D mesh with tetrahedrons, get triangle element connectivity nodes)
    lower_order_elements = retrieve_lower_order_elements(
        connect, boundary_conditions_masks, boundary
    )
    # Aggregate lower and higher order elements
    elements = []
    elements.append(("tetra", connect.astype(np.int64)))
    elements.append(("triangle", lower_order_elements))

    # Process all elements in the boundary to collect physical tags for lower order elements
    lower_order_group = [
        seissol_to_tandem_tag_conversion(x, seissol_to_tandem_tag_map)
        for element in boundary
        for x in element.view(dtype=each_face_bc_tag_i8)
        if x > 0
    ]
    # Aggregate lower and higher order physical tags for Tandem
    cell_data = {
        "gmsh:physical": [
            group.astype(np.int64),  # Tags for tetra elements
            np.array(lower_order_group, dtype=np.int64),  # Tags for triangle elements
        ]
    }

    # Create the mesh using meshio
    # TODO: Explicitly define geometric tags to avoid warning. Since the geometric tag information is not
    # parsed/lost during the pumgen conversion, currently there is an error which should be explicitly
    # coded here.
    mesh = meshio.Mesh(
        points=geometry,  # Node coordinates
        cells=elements,  # Element connectivity
        cell_data=cell_data,  # Physical tags for each element type
    )

    # Write the mesh to the .msh file in Gmsh format
    # Set binary explicitly False to avoid binary file writing
    mesh.write(msh_file, file_format="gmsh22", binary=False)  # Gmsh version 2.2 format


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Convert an HDF5 (.h5) mesh file to GMSH (.msh) format."
    )
    parser.add_argument(
        "--input_mesh_file",
        required=True,
        help="Path to the input HDF5 mesh file (required).",
    )
    parser.add_argument(
        "--tag_map_yaml_file",
        required=True,
        help="Path to the yaml file consisting of mapping from Seissol boundary tags to Tandem boundary tags.",
    )
    parser.add_argument(
        "--output_mesh_file",
        default=None,
        help=(
            "Path to the output GMSH mesh file (optional). "
            "If not provided, the output filename will be "
            "derived from the input filename."
        ),
    )

    # Parse the command-line arguments
    args = parser.parse_args()
    input_file = args.input_mesh_file

    output_file = args.output_mesh_file
    # If no output file is provided, derive it from the input file
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_converted.msh"

    seissol_to_tandem_map_file = args.tag_map_yaml_file
    # Get mapping for seissol to tandem tags
    seissol_to_tandem_tag_map = load_yaml_mapping(seissol_to_tandem_map_file)
    check_tag_map_validity(seissol_to_tandem_tag_map)

    # Run the conversion
    convert_h5_to_msh(input_file, output_file, seissol_to_tandem_tag_map)
