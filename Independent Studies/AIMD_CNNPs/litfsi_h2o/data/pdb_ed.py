import sys


def map_atom_to_element(atom_name):
    """Map atom names to their corresponding element symbols."""
    element_map = {
        "LI": "Li",  # Lithium ion
        "NBT": "N",  # Nitrogen in TFSI
        "SBT": "S",  # Sulfur in TFSI
        "OBT": "O",  # Oxygen in TFSI
        "CBT": "C",  # Carbon in TFSI
        "F1": "F",  # Fluorine in TFSI
        "OW": "O",  # Oxygen in water (preserved)
        "HW1": "H",  # Hydrogen in water (preserved)
        "HW2": "H",  # Hydrogen in water (preserved)
    }
    return element_map.get(atom_name.strip(), "")


def process_pdb_line(line):
    """Process a PDB line, adding element symbol if needed."""
    if line.startswith("ATOM") or line.startswith("HETATM"):
        # Extract atom name (columns 13-16, left-aligned)
        atom_name = line[12:16].strip()
        # Get element symbol
        element = map_atom_to_element(atom_name)
        if element:
            # Ensure element is right-aligned in columns 77-78
            # Pad or truncate to fit PDB format
            element_padded = f"{element:>2}"
            # Reconstruct line, preserving all other columns
            new_line = line[:76] + element_padded + line[78:]
            return new_line
    return line


def add_elements_to_pdb(input_file, output_file):
    """Read input PDB, add element symbols, and write to output PDB."""
    try:
        with open(input_file, "r") as infile, open(output_file, "w") as outfile:
            for line in infile:
                modified_line = process_pdb_line(line)
                outfile.write(modified_line)
        print(f"Successfully wrote modified PDB to {output_file}")
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


def main():
    input_pdb = "gro_md/final_npt.pdb"
    output_pdb = "gro_md/final_npt_elements.pdb"
    add_elements_to_pdb(input_pdb, output_pdb)


if __name__ == "__main__":
    main()
