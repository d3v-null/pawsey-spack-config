#!/bin/bash

nprocs="128"
if [ ! -z $1 ]; then
  nprocs="$1"
fi

script_dir="$(readlink -f "$(dirname $0 2>/dev/null)" || pwd)"

#list of environments
envs=( \
env_utils \
env_num_libs \
env_python \
env_io_libs \
env_langs \
env_apps \
env_devel \
env_bench \
env_s3_clients \
env_astro \
env_bio \
)

envs_depsonly=( \
env_roms \
env_wrf \
)

envdir="${script_dir}/../environments"

timestamp="$(date +"%Y-%m-%d_%Hh%M")"
top_logdir="${SPACK_LOGS_BASEDIR:-"${script_dir}/logs"}"
logdir="${top_logdir}/install.${timestamp}"
mkdir -p ${logdir}

for env in ${envs[@]}
do
  cd ${envdir}/${env}
  spack env activate . 
  spack concretize -f 1> ${logdir}/spack.concretize.${env}.log 2> ${logdir}/spack.concretize.${env}.err
  sg $PAWSEY_PROJECT -c "spack install --no-checksum -j${nprocs} > ${logdir}/spack.install.${env}.log 2> ${logdir}/spack.install.${env}.err"
  spack env deactivate
  cd ${script_dir}
done

for env in ${envs_depsonly[@]}
do
  cd ${envdir}/${env}
  spack env activate .
  spack concretize -f 1> ${logdir}/spack.concretize.${env}.log 2> ${logdir}/spack.concretize.${env}.err
  sg $PAWSEY_PROJECT -c "spack install --no-checksum -j${nprocs} --only dependencies 1> ${logdir}/spack.install.${env}.log 2> ${logdir}/spack.install.${env}.err"
  spack env deactivate
  cd ${script_dir}
done
