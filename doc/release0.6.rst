======================================
 Release notes for nitime version 0.4
======================================

Summary of changes
------------------

Major changes introduced in version 0.4 of :mod:`nitime`:

#.  :mod:`LazyImports <nitime.lazy>`: Imports of modules
     are delayed until they are actually used. Work led by Paul Ivanov

#. :class:`TimeArray <nitime.timeseries.TimeArray>` math: Mathematical
     operations such as multiplication/division, as well as min/max/mean/sum
     are now implemented for the TimeArray class. Work led by Paul Ivanov.

#. Replace numpy FFT with scipy FFT. This should improve performance. Work
    instigated and led by Alex Gramfort.

#. Scipy > 0.10 compatibility: Changes to recent versions of scipy have caused
    import of some modules of nitime to break. This version should have fixed this
    issue.

Contributors to this release
----------------------------

The following people contributed to this release:

* Alexandre Gramfort
* Ariel Rokem
* endolith
* Paul Ivanov
* Sergey Karayev
* Yaroslav Halchenko


.. Note::

   This list was generated using::

   git log --pretty=format:"* %aN" PREV_RELEASE... | sort | uniq

   Please let us know if you should appear on this list and do not, so that we
   can add your name in future release notes.


Detailed stats from the github repository
-----------------------------------------

GitHub stats for the last  311 days.
We closed a total of 51 issues, 17 pull requests and 34 regular
issues; this is the full list (generated with the script
`tools/github_stats.py`):

Pull Requests (17):

* :ghissue:`104`: nose_arg gives test finer granularity
* :ghissue:`103`: make the LazyImport jig pickleable
* :ghissue:`101`: fix some typos
* :ghissue:`99`: First of all, thanks a lot for the sweet correlation matrix visualization method! I noticed that the behavior of the color_anchor parameter is not what I expected. Let me know what you think
* :ghissue:`98`: RF: utils.multi_interesect no longer relies on deprecated intersect1d_nu
* :ghissue:`96`: BF: Import factorial from the scipy.misc namespace.
* :ghissue:`94`: BF: Account for situations in which TimeSeries has more than two dimensions
* :ghissue:`92`: Time array math functions
* :ghissue:`91`: Timearray math
* :ghissue:`88`: Lazy imports
* :ghissue:`89`: Masked arrays
* :ghissue:`86`: ENH: Different versions of nose require different input to first-package-
* :ghissue:`83`: BF: Improvements and fixes to nosetesting.
* :ghissue:`81`: ENH : s/numpy.fft/scipy.fftpack
* :ghissue:`77`: BF: Carry around a copy of some of the spectral analysis functions.
* :ghissue:`79`: pep8 + pyflakes + misc readability
* :ghissue:`78`: Doctests

Issues (34):

* :ghissue:`30`: Make default behavior for fmri.io.time_series_from_file
* :ghissue:`84`: Note on examples
* :ghissue:`93`: TimeArray .prod is borked (because of overflow?)
* :ghissue:`104`: nose_arg gives test finer granularity
* :ghissue:`103`: make the LazyImport jig pickleable
* :ghissue:`102`: sphinx docs won't build (related to lazyimports?)
* :ghissue:`87`: Test failures on 10.4
* :ghissue:`100`: magnitude of fft showing negative values
* :ghissue:`101`: fix some typos
* :ghissue:`99`: First of all, thanks a lot for the sweet correlation matrix visualization method! I noticed that the behavior of the color_anchor parameter is not what I expected. Let me know what you think
* :ghissue:`97`: utils.py uses feature removed from numpy1.6
* :ghissue:`98`: RF: utils.multi_interesect no longer relies on deprecated intersect1d_nu
* :ghissue:`95`: ImportError: Cannot Import name Factorial
* :ghissue:`96`: BF: Import factorial from the scipy.misc namespace.
* :ghissue:`94`: BF: Account for situations in which TimeSeries has more than two dimensions
* :ghissue:`92`: Time array math functions
* :ghissue:`91`: Timearray math
* :ghissue:`88`: Lazy imports
* :ghissue:`89`: Masked arrays
* :ghissue:`80`: Replace numpy fft with scipy fft
* :ghissue:`86`: ENH: Different versions of nose require different input to first-package-
* :ghissue:`85`: slicing time using epochs that start before or end after
* :ghissue:`83`: BF: Improvements and fixes to nosetesting.
* :ghissue:`82`: nosetest w/o exit=False funks up in ipython
* :ghissue:`81`: ENH : s/numpy.fft/scipy.fftpack
* :ghissue:`32`: Add a "how to release" page in the docs
* :ghissue:`35`: index_at seems to fail with negative times
* :ghissue:`50`: Setting IIR filter lower bound to 0
* :ghissue:`44`: Warning when using coherence with welch method and NFFT longer than the time-series itself
* :ghissue:`68`: tril_indices not available in fairly recent numpy versions
* :ghissue:`77`: BF: Carry around a copy of some of the spectral analysis functions.
* :ghissue:`55`: Warning in analysis.coherence might be a bug
* :ghissue:`79`: pep8 + pyflakes + misc readability
* :ghissue:`78`: Doctests

