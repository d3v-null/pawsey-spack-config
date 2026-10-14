# Copyright 2013-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
# Contribute back the recipe
import os

from spack.package import *


class Idg(CMakePackage, CudaPackage, ROCmPackage):
    """
    Image Domain Gridding (IDG) is a fast method for convolutional resampling (gridding/degridding) 
    of radio astronomical data (visibilities). Direction-dependent effects (DDEs) 
    or A-tems can be applied in the gridding process.
    """

    homepage = "https://www.astron.nl/citt/IDG/"
    git      = "https://gitlab.com/astron-idg/idg.git"

    version("1.2.0", tag="1.2.0")
    version("1.1.0", tag="1.1.0")
    version("1.0.0", tag="1.0.0")
    version("0.7", tag="0.7")
    # ASTRON's experimental HIP/ROCm port (Dec 2021, forked from a pre-1.0 snapshot).
    # The branch only exists on ASTRON's own GitLab, not on the gitlab.com mirror.
    # Status on Setonix (ROCm 6.3.0, MI250X gfx90a), Oct 2026: Generic and Hybrid proxies
    # build and run; gridding/degridding match the CPU reference, the average-beam test
    # does not, and the Unified (managed-memory) proxy segfaults in its final FFT.
    version("hip-rocm", branch="llu/hip-rocm", git="https://git.astron.nl/RD/idg.git")

    depends_on('boost')
    depends_on('fftw-api@3')
    depends_on('blas')

    # The HIP port compiles its GPU kernels at runtime by shelling out to hipcc.
    depends_on("cmake@3.21:", type="build", when="+rocm")  # enable_language(HIP)
    depends_on("hip", when="+rocm")
    depends_on("hipfft", when="+rocm")
    depends_on("llvm-amdgpu", when="+rocm")

    requires("@hip-rocm", when="+rocm", msg="ROCm support only exists on the idg@hip-rocm branch")
    conflicts("+cuda", when="+rocm", msg="idg: enable either +cuda or +rocm, not both")
    conflicts("+cuda", when="@hip-rocm", msg="idg@hip-rocm is the ROCm port; use a tagged release for CUDA")

    patch("fix_FFT.cpp.patch")
    # wsclean >= 3.1 needs the stokes_I_only idg-api option (upstream 343e78cd, after the branch point).
    patch("hip-rocm-stokes-I-only.patch", when="@hip-rocm")

    def _hip_clangxx(self):
        """clang++ shipped with ROCm, needed as CMAKE_HIP_COMPILER (hipcc is not accepted by CMake>=3.28)."""
        spec = self.spec
        candidates = [
            spec["llvm-amdgpu"].prefix.bin,
            spec["hip"].prefix.lib.llvm.bin,
            spec["hip"].prefix.llvm.bin,
        ]
        for d in candidates:
            clangxx = join_path(d, "clang++")
            if os.path.exists(clangxx):
                return clangxx
        raise InstallError("idg+rocm: could not find ROCm clang++ in {}".format(candidates))

    def _amdgpu_targets(self):
        return ";".join(self.spec.variants["amdgpu_target"].value)

    def cmake_args(self):
        spec = self.spec
        args = [self.define("BUILD_TESTING", self.run_tests)]
        if spec.satisfies("+cuda"):
            args.append("-DBUILD_LIB_CUDA=ON")
            args.append(f"-DCUDA_ROOT_DIR={spec['cuda'].prefix}")
        if spec.satisfies("+rocm"):
            rocm = spec["hip"].prefix
            hipfft = spec["hipfft"].prefix
            targets = self._amdgpu_targets()
            args += [
                # The HIP sources live in the CUDA tree, selected by BUILD_WITH_HIP.
                self.define("BUILD_WITH_HIP", True),
                self.define("BUILD_LIB_CUDA", True),
                self.define("CMAKE_HIP_COMPILER", self._hip_clangxx()),
                self.define("CMAKE_HIP_ARCHITECTURES", targets),
                self.define("AMDGPU_TARGETS", targets),
                self.define("GPU_TARGETS", targets),
                # The branch still calls idg's own FindCUDA.cmake (idg-lib/cmake/modules), which
                # requires CUDA_CUDA_LIBRARY and CUDA_INCLUDE_DIR. Pre-seed every variable it
                # would set with the ROCm equivalents so the search is skipped.
                self.define("CUDA_ROOT_DIR", rocm),
                self.define("CUDA_INCLUDE_DIR", rocm.include),
                self.define("CUDA_LIB_DIR", rocm.lib),
                self.define("CUDA_CUDART_LIBRARY", join_path(rocm.lib, "libamdhip64.so")),
                self.define("CUDA_CUDA_LIBRARY", join_path(rocm.lib, "libamdhip64.so")),
                self.define("CUDA_FFT_LIBRARY", join_path(hipfft.lib, "libhipfft.so")),
                # NVTX markers are compiled out (#if 0) on this branch; must not be left -NOTFOUND.
                self.define("CUDA_NVTX_LIBRARY", join_path(rocm.lib, "libamdhip64.so")),
                # Baked into idg-config.h as NVCC and used for runtime kernel compilation.
                self.define("CUDA_NVCC", spec["hip"].hipcc),
                # Referenced by the top-level CMakeLists.txt but never set by find_package(hip).
                self.define("HIPFFT_INCLUDE_DIR", hipfft.include.hipfft),
                self.define("HIPFFT_LIBRARIES", join_path(hipfft.lib, "libhipfft.so")),
            ]
        return args

    def setup_build_environment(self, env):
        if self.spec.satisfies("+rocm"):
            rocm = self.spec["hip"].prefix
            env.set("HIP_PATH", rocm)
            env.set("ROCM_PATH", rocm)

    def _hip_runtime_env(self, env):
        if self.spec.satisfies("+rocm"):
            # Kernels are compiled at runtime with `hipcc --genco $IDG_HIP_FLAGS ...`;
            # without an offload arch hipcc defaults to gfx906 and the code object will
            # not load on the device.
            archs = " ".join(f"--offload-arch={a}" for a in self.spec.variants["amdgpu_target"].value)
            env.set("IDG_HIP_FLAGS", archs)
            env.set("HIP_PATH", self.spec["hip"].prefix)
            env.set("ROCM_PATH", self.spec["hip"].prefix)

    def setup_run_environment(self, env):
        self._hip_runtime_env(env)

    def setup_dependent_run_environment(self, env, dependent_spec):
        # Dependents such as wsclean load idg's GPU kernels at runtime, so their module
        # must carry IDG_HIP_FLAGS too (project modules on Setonix do not autoload deps).
        self._hip_runtime_env(env)
