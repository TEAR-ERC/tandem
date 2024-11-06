#! /usr/bin/env sh

# NOTE: This script was taken from PointCloudLibrary/pcl and adapted for SeisSol, and then Tandem

# sample command line usage: $0 clang-format(version == 18.1.5) $TANDEM_SOURCE_DIR
# $ cd $TANDEM_SOURCE_DIR; sh ./.ci/format.sh `which clang-format` ./
# $ cd $TANDEM_SOURCE_DIR/.ci; sh format.sh `which clang-format` ../

format() {
    # don't use a directory with whitespace
    local allowlist_dir="src app test"

    local TANDEM_SOURCE_DIR="${2}"
    local formatter="${1}"

    if [ ! -f "${formatter}" ]; then
        echo "Could not find clang-format. Please specify one as the first argument"
        exit 176
    fi
    clangFormatVersion="19.1.3"
    local formatter_version=$(${formatter} --version)
    if [ "${formatter_version}" != "clang-format version $clangFormatVersion" ]; then
        echo "Your clang-format tool in \"${formatter}\" does not have the correct version (should be $clangFormatVersion). Given: ${formatter_version}"
        echo "Hint: you may install the required clang-format via pip, by typing: pip3 install clang-format==$clangFormatVersion"
        exit 176
    fi

    # check for self
    if [ ! -f "${TANDEM_SOURCE_DIR}/.ci/format.sh" ]; then
        echo "Please ensure that TANDEM_SOURCE_DIR is passed as the second argument"
        exit 176
    fi

    for dir in ${allowlist_dir}; do
        path=${TANDEM_SOURCE_DIR}/${dir}
        files=$(find ${path} -type f -iname *.[ch] -o -iname *.[ch]pp -o -iname *.[ch]xx -iname *.cu)
        for file in ${files}; do
            sed -i 's/#pragma omp/\/\/#pragma omp/g' $file
            ${formatter} -i -style=file $file
            sed -i 's/\/\/ *#pragma omp/#pragma omp/g' $file
        done
    done

}

format $@
