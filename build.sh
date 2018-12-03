#!/bin/bash

function print_usage() {
    echo -e "Build script commands: build.sh cmd [options]\n"
    echo "Commands:"
    echo -e "\tbuild: build for the current system architecture"
    echo -e "\tclean: remove all build artifacts"
    echo -e "\thelp: print this message"

    echo "Options:"
    echo -e "\t-noviz: build without OpenCV support, so no ability to visualize the network"
    echo -e "\t-arm: build for ARM architecture"
    echo -e "\t-rel: build release version (optimized)"
}


function build_command() {
    mkdir -p build
    pushd build > /dev/null

    rm -rf CMake* cmake* *.so Make* test
    echo "Build command: '${1}'"
    ${1}
    local num_cores=$(cat /proc/cpuinfo | grep processor | wc -l)
    echo "Building with ${num_cores} threads..."
    make -j${num_cores}
    popd > /dev/null
}


# Do not echo in this function anything besides the final string
function make_build_string() {
    declare -A options=(["-arm"]="-DCMAKE_TOOLCHAIN_FILE=../pi.cmake"
                        ["-noviz"]="-DNO_OPENCV:bool=true"
                        ["-rel"]="-DCMAKE_BUILD_TYPE=Release");
    declare -A dups;
    local build_string="";
    for option in "${@}";do
        if ! test "${dups[${option}]+isset}";then
            dups[${option}]="isset";
        else
            continue
        fi

        dups["${option}"]="already_exists";

        if test "${options[${option}]+isset}";then
            build_string+=${options[${option}]};
            build_string+=" ";
        fi
    done
    echo "${build_string}";
}


function main() {
    local cmd=${1};
    shift;

    case ${cmd} in
        build)
            local build_str=$(make_build_string $@)
            build_str="cmake ${build_str} ..";
            build_command "${build_str}"
            ;;
        clean)
            echo "Cleaning the build artifacts..."
            rm -rf build
            ;;
        help)
            print_usage
            ;;
        *)
            print_usage
            ;;
    esac
}

main $@
