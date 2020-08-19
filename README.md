# Flatiron Institute Nonuniform Fast Fourier Transform library: FINUFFT

Principal author **Alex H. Barnett**,
main co-developers Jeremy F. Magland,
Ludvig af Klinteberg, Yu-hsuan "Melody" Shih, Andrea Malleo, Libin Lu.
see `docs/ackn.rst` for details and full list of contributors.

<p>
<img src="docs/logo.png" width="350"/>
<img src="docs/spreadpic.png" width="400"/>
</p>

This is a lightweight library to compute the three standard types of nonuniform FFT to a specified precision, in one, two, or three dimensions. It is written in C++ with interfaces to C, Fortran, MATLAB/octave, and python. A julia interface
also exists.

Please see the [online documentation](http://finufft.readthedocs.io/en/latest/index.html), or its equivalent, the [user manual](finufft-manual.pdf).
You will also want to see example codes in the directories
`examples`, `test`, `fortran`, `matlab/test`, and `python/test`.
If you cannot compile, try our [precompiled binaries](http://users.flatironinstitute.org/~ahb/codes/finufft-binaries).

If you prefer to read text files, the source to generate the above documentation is in human-readable (mostly .rst) files as follows:

- `docs/install.rst` : installation and compilation instructions
- `docs/dirs.rst`    : explanation of directories and files in the package
- `docs/math.rst`    : mathematical definitions
- `docs/cex.rst`     : example usage from C++/C
- `docs/c.rst`       : documentation of C++/C functions
- `docs/opts.rst`    : optional parameters
- `docs/error.rst`   : error codes
- `docs/trouble.rst` : troubleshooting
- `docs/tut.rst`     : tutorial application examples
- `docs/fortran.rst` : usage examples from Fortran, documentation of interface
- `docs/matlab.rst` and `docs/matlabhelp.raw` : using the MATLAB/octave interface
- `docs/python.rst` and `python/*/_interfaces.py` : using the python interface
- `docs/julia.rst`   : using the Julia interface
- `docs/devnotes.rst`: notes/guide for developers
- `docs/related.rst` : other recommended NUFFT packages
- `docs/users.rst`   : users of FINUFFT and dependent packages
- `docs/ackn.rst`    : authors and acknowledgments
- `docs/refs.rst`    : journal article references (ours and others)


If you find FINUFFT useful in your work, please cite this repository and
our paper:

A parallel non-uniform fast Fourier transform library based on an ``exponential of semicircle'' kernel.
A. H. Barnett, J. F. Magland, and L. af Klinteberg.
SIAM J. Sci. Comput. 41(5), C479-C504 (2019).
