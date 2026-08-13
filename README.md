# xsimgs

XSIM Gaussian splatting rendering library for self-driving sensor simulation, built as a
PyTorch C++/CUDA extension. Part of the [XSIM framework](https://github.com/whesense/xsim).

## Install

The build compiles a C++/CUDA extension against your existing environment, so it needs:

- **PyTorch 2.x** already installed.
- **GCC/G++ with C++17 support**.
- **A CUDA toolkit with `nvcc`** matching your PyTorch build. 
- **[pytoast](https://github.com/npatakin/pytoast)** already installed — xsimgs builds
  on its math headers and runtime.
- **pybind11 >= 3.0.4**: `pip install "pybind11>=3.0.4"`.

```bash
git clone --recursive <repo-url> xsimgs
cd xsimgs
pip install --no-build-isolation .
```

The `--recursive` flag pulls the GLM submodule (`third_party/glm`); if you forgot it,
the build will attempt `git submodule update --init` for you.

If you are running the build on a machine with a low RAM to CPU cores ratio, you can
limit the number of compile processes to avoid running out of memory:

```bash
MAX_JOBS=8 pip install --no-build-isolation .
```

Test the installation is successful with an import:

```python
import xsimgs
```

## Citation

If you use xsimgs in your research, please cite our paper
([arXiv:2602.05617](https://arxiv.org/abs/2602.05617), accepted to IJCAI 2026):

```bibtex
@article{patakin2026xsim,
    title   = {Unified Sensor Simulation for Autonomous Driving},
    author  = {Patakin, Nikolay and Shirokov, Arsenii and Konushin, Anton and Senushkin, Dmitry},
    journal = {arXiv preprint arXiv:2602.05617},
    year    = {2026},
    note    = {Accepted to IJCAI 2026}
}
```

## Acknowledgments

Substantial portions of the CUDA rasterization and spherical-harmonics kernels are derived from
[gsplat](https://github.com/nerfstudio-project/gsplat) (Apache 2.0).

## License

Apache 2.0 -- see [LICENSE](LICENSE); [NOTICE](NOTICE) for details.
