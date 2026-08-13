import subprocess
from pathlib import Path

from setuptools import setup, find_packages
from torch.utils.cpp_extension import (
    BuildExtension, CUDAExtension
)

import toast
import pybind11

DEFAULT_EXTRA_ARGS = {
    "cxx": ["-fvisibility=hidden", '-DAT_PARALLEL_OPENMP', '-fopenmp', '-O3'],
    "nvcc": [
        "-DCUDA_HAS_FP16=1",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
        "--extended-lambda",
        "-O3",
        "--resource-usage",
        '-DAT_PARALLEL_OPENMP',
        "-Xcompiler", "-fopenmp"
    ]
}

ROOT = Path(__file__).parent.resolve()

# (submodule path, a file that must exist once it is checked out)
SUBMODULES = [
    ('third_party/glm', 'glm/glm.hpp'),
]


def ensure_submodules():
    missing = [p for p, sentinel in SUBMODULES if not (ROOT / p / sentinel).is_file()]
    if not missing:
        return

    if not (ROOT / '.git').exists():
        raise RuntimeError(
            f"Missing submodule(s): {', '.join(missing)}. This is not a git "
            "checkout, so they cannot be fetched automatically -- please "
            "obtain the sources with the submodules included."
        )

    print(f"Initializing git submodule(s): {', '.join(missing)}")
    subprocess.check_call(
        ['git', 'submodule', 'update', '--init', '--recursive', '--'] + missing,
        cwd=str(ROOT),
    )

    still_missing = [p for p, sentinel in SUBMODULES if not (ROOT / p / sentinel).is_file()]
    if still_missing:
        raise RuntimeError(
            f"Submodule(s) {', '.join(still_missing)} are still missing after "
            "'git submodule update --init'."
        )


def find_sources(path):
    path = Path(path)
    return list(filter(lambda x: ('cmake-' not in x), map(str,
        list(path.rglob('*.cpp')) + list(path.rglob('*.cu'))
    )))


ensure_submodules()

setup(
    name='xsimgs',
    license='Apache-2.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    ext_modules=[
        CUDAExtension(
            'xsimgs._C',
            sources=find_sources('csrc'),
            include_dirs=[
                str(Path('include/').absolute()),
                toast.get_include_path(),
                str(Path('third_party/glm/').absolute()),
                pybind11.get_include(),
            ],
            extra_compile_args=DEFAULT_EXTRA_ARGS
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    version="0.1.0",
)
