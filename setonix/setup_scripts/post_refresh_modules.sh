#!/bin/bash

# this script goes through 5 steps
# 1. refresh spack modules
# 2. creates all missing module directories
# 3. update singularity modules
# 4. refresh wrf/roms dependency modules
# 5. create hpc-python view and module
# 6. apply licensing permissions
# 7. refresh shpc symlink modules

# source setup variables
# if copy/pasting these commands, need to run from this directory
script_dir="$(readlink -f "$(dirname $0 2>/dev/null)" || pwd)"
. ${script_dir}/variables.sh

# for provisional setup (no spack modulepaths yet)
is_avail_spack="$( module is-avail spack/${spack_version} ; echo "$?" )"
if [ "${is_avail_spack}" != "0" ] ; then
  module use ${root_dir}/${pawsey_temp}
  module load ${pawsey_temp}
  module swap PrgEnv-gnu PrgEnv-cray
  module swap PrgEnv-cray PrgEnv-gnu
  module swap gcc gcc/${gcc_version}
fi
# spack module
is_loaded_spack="$( module is-loaded spack/${spack_version} ; echo "$?" )"
if [ "${is_loaded_spack}" != "0" ] ; then
  module load spack/${spack_version}
fi


# step 1. refresh spack modules
echo "Do you want to delete and re-create the Spack modules? (yes/no)"
read spack_answer
if [ ${spack_answer,,} == "yes" ] ; then
  echo "Deleting and re-creating Spack modules.."
  spack module lmod refresh -y --delete-tree
else
  echo "Skipping refresh of Spack modules."
fi


# step 2. creates all missing module directories
echo "Creating all missing module directories.."
bash "${script_dir}/setup_create_system_moduletree.sh"


# step 3. update singularity modules
echo "Updating Singularity modules.."
bash "${script_dir}/setup_singularity_pawsey_modules.sh"


# step 4. refresh wrf/roms dependency modules
echo "Do you want to refresh the wrf/roms dependency modules? (yes/no)"
read dependency_answer
if [ ${dependency_answer,,} == "yes" ] ; then
  echo "Refreshing wrf/roms dependency modules.."
  bash "${script_dir}/post_make_wrf_roms_dependency_modules.sh"
else
  echo "Skipping refresh of wrf/roms dependency modules."
fi


# step # 5. create hpc-python view and module
echo "Do you want to create the hpc-python view and module? (yes/no)"
read hpc_python_answer
if [ ${hpc_python_answer,,} == "yes" ] ; then
  echo "Creating hpc-python view and module.."
  bash "${script_dir}/post_make_hpc_python_collection_view_module.sh"
else
  echo "Skipping creation of hpc-python view and module."
fi


# step 6. apply licensing permissions
echo "In addition to module directories, do you want to apply "
echo "licensing permissions to software directories, too? (yes/no)"
read license_answer
if [ ${license_answer,,} == "yes" ] ; then
  bash ${script_dir}/post_set_licensing_permissions.sh
else
  bash ${script_dir}/post_set_licensing_permissions.sh only-modules
fi


# step 7. refresh shpc symlink modules
echo "Do you want to refresh the shpc symlink modules? (yes/no)"
read shpc_answer
if [ ${shpc_answer,,} == "yes" ] ; then
  echo "Refreshing shpc symlink modules.."
  bash ${script_dir}/post_create_shpc_symlink_modules.sh
else
  echo "Skipping deletion and re-creation of shpc symlink modules."
fi
