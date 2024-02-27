#!/bin/bash

# $1: seqnum to be processed
# $2: true/false, activate/deactivate generation of raws with galsim
# $3: true/false, activate/deactivate ingesting raws in the repo
# $4: true/false, activate/deactivate pipetask to generate calexp from raws
# $5: true/false, activate/deactivate the update of collection_dict
# $6: 0: save to personal area; 1: save to shared area

if [ "$6" ==  1  ]; then
    REPO="/sdf/data/rubin/shared/image_quality/imsim/repo/"
    collection_dictionary_github="/sdf/home/v/vittorio/rubin-user/projects/imsim_2024_03/imSim/github/RubinIQ4AO/notebooks/collection_dictionary_shared.py"
    collection_dictionary_personal="/sdf/home/v/vittorio/notebooks/my_notebooks/collection_dictionary_shared.py"
    folder_drp="/sdf/data/rubin/shared/image_quality/imsim/"
else
    REPO="/sdf/home/v/vittorio/rubin-user/projects/imsim_2024_03/imSim/repo/"
    collection_dictionary_github="/sdf/home/v/vittorio/rubin-user/projects/imsim_2024_03/imSim/github/RubinIQ4AO/notebooks/collection_dictionary.py"
    collection_dictionary_personal="/sdf/home/v/vittorio/notebooks/my_notebooks/collection_dictionary.py"
    folder_drp="/sdf/home/v/vittorio/rubin-user/projects/imsim_2024_03/imSim/"
fi

folder_imsim="/sdf/home/v/vittorio/rubin-user/projects/imsim_2024_03/imSim/config/"
subfolder_1="config_files/"
subfolder_2="instance_catalogs/"
subfolder_3="output/"

data_dir="$( printf "%s%s" $folder_imsim $subfolder_3)"

seqnum_in=$1
seqnum_in_str="$( printf "%05d" $seqnum_in)" # convert seqnum_in to string (5 digits, to search visit id in the repo)
seqnum_in_str2="$( printf "%04d" $seqnum_in)" # convert seqnum_in to string (4 digits, for ingest)
f_in_config="$( printf "%s%simsim-user-instcat_seqnum%04d.yaml" $folder_imsim $subfolder_1 $seqnum_in )"

#-------------------------------------
# Image generation with ImSim
#-------------------------------------
if $2; then
    echo "---IMSIM---"
    galsim ${f_in_config}
fi

#-------------------------------------
# Ingest raws within the repo
#-------------------------------------
if $3; then
    echo "---INGEST RAWS---"
    butler ingest-raws --transfer copy ${REPO} ${data_dir}/amp_*${seqnum_in_str2}*fz
    butler define-visits --collections LSSTCam/raw/all ${REPO} LSSTCam
fi

#-------------------------------------
# PIPETASK
#-------------------------------------

if $4; then
    echo "---PIPETASK---"

# folder containing the DRP file to be read by pipetask
# folder_drp="/sdf/data/rubin/shared/image_quality/imsim/"
    # Prima di 240220, la colonna della visita è 8...
    # visit_id=$(butler query-datasets ${REPO} | grep ${seqnum_in_str} | grep "raw" | awk 'NR==1{print $8}')
    visit_id=$(butler query-datasets ${REPO} | grep ${seqnum_in_str} | grep "raw" | awk 'NR==1{print $6}')
    
    TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
    pipetask --long-log run -p ${folder_drp}"drp_230921.yaml#isr,characterizeImage,calibrate"          -i "LSSTCam/raw/all,LSSTCam/calib,u/jchiang/calib_imSim2.0_w_2023_04"          -o pipetask_output          -b ${REPO}          -d "instrument='LSSTCam' AND (visit=${visit_id})"  -c characterizeImage:installSimplePsf.fwhm=2.5        --register-dataset-types  >& ${folder_drp}"run_tasks012_$TIMESTAMP.log"
fi

#-------------------------------------
# Change/Add entries in the collection dictionary file (both personal and shared area!!)
#-------------------------------------

if $5; then
    echo "---CHANGE/ADD COLLECTION DICTIONARY---"
    repo_folder=$(butler query-datasets ${REPO} | grep ${seqnum_in_str} | grep "isr_log" | awk 'NR==1{print $2}')
    
    #Prima dovrei controllare che l'attuale seqnum non sia nel collection_dictionary e poi aggiungerlo o sostituirlo
    seqnum_in_collection=$(grep $seqnum_in":" $collection_dictionary_github)
    seqnum_in_collection_personal=$(grep $seqnum_in":" $collection_dictionary_personal)
    
    # Manipolo la stringa da inserire per essere usabile con sed
    repo_folder_to_substitute=${repo_folder/"/"/"\/"}

    if [ "$seqnum_in_collection" == "" ]; then
        echo "Not in collection (github)"
        sed -i "s/{/{\n      ${seqnum_in}: '${repo_folder_to_substitute}',/" ${collection_dictionary_github}
    else
        echo "Already in collection (github)"
    # Taglio la stringa da sostituire e la manipolo per essere usabile con sed
        # string_to_substitute=${seqnum_in_collection#*"'"}  # retain the part after the last slash
        string_to_substitute=${seqnum_in_collection#*"'"}
        string_to_substitute=${seqnum_in_collection%"',"}  # retain the part before the '}
        string_to_substitute=${string_to_substitute/"/"/"\/"}
        sed -i "s/${string_to_substitute}/${seqnum_in}: '${repo_folder_to_substitute}/" ${collection_dictionary_github}
    fi

    if [ "$seqnum_in_collection_personal" == "" ]; then
        echo "Not in collection (personal)"
        sed -i "s/{/{\n      ${seqnum_in}: '${repo_folder_to_substitute}',/" ${collection_dictionary_personal}
    else
        echo "Already in collection (personal)"
    # Taglio la stringa da sostituire e la manipolo per essere usabile con sed
        # string_to_substitute=${seqnum_in_collection#*"'"}  # retain the part after the last slash
        string_to_substitute_personal=${seqnum_in_collection_personal%"',"}  # retain the part before the '}
        string_to_substitute_personal=${string_to_substitute_personal%/"/"/"\/"}  # retain the part before the '}
        sed -i "s/${string_to_substitute_personal}/${repo_folder_to_substitute}/" ${collection_dictionary_personal}"
    fi

fi
