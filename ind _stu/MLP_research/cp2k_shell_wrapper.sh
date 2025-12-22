#!/bin/bash

# Parse input and output file arguments
while getopts "i:o:" opt; do
    case $opt in
        i) INPUT_FILE="$OPTARG" ;;
        o) OUTPUT_FILE="$OPTARG" ;;
        \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    esac
done

if [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_FILE" ]; then
    echo "Usage: $0 -i input_file -o output_file"
    exit 1
fi

# Set environment variables for CP2K and OpenMPI
export CP2K_DATA_DIR="/opt/homebrew/share/cp2k/data"
export DYLD_LIBRARY_PATH="/opt/homebrew/lib:/opt/homebrew/Cellar/open-mpi/5.0.7/lib:${DYLD_LIBRARY_PATH}"
export PATH="/opt/homebrew/Cellar/open-mpi/5.0.7/bin:/opt/homebrew/bin:${PATH}"
export OMPI_MCA_btl_tcp_port_min_v4="10000"
export OMPI_MCA_btl_tcp_port_range_v4="1000"

# Run CP2K with the input and output files
/opt/homebrew/bin/cp2k.psmp -i "$INPUT_FILE" -o "$OUTPUT_FILE"

# Exit with the CP2K exit code
exit $?
