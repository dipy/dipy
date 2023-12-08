// This config file is almost similar to 'asv.conf.json' except it contains
// custom tokens that can be substituted by 'runtests.py' and ASV,
// due to the necessity to add custom build options when `--bench-compare`
// is used.
{
    // The version of the config file format.  Do not change, unless
    // you know what you are doing.
    "version": 1,

    // The name of the project being benchmarked
    "project": "dipy",

    // The project's homepage
    "project_url": "https://www.dipy.org/",

    // The URL or local path of the source code repository for the
    // project being benchmarked
    "repo": "..",

    // List of branches to benchmark. If not provided, defaults to "master"
    // (for git) or "tip" (for mercurial).
    "branches": ["HEAD"],

    // The DVCS being used.  If not set, it will be automatically
    // determined from "repo" by looking at the protocol in the URL
    // (if remote), or by looking for special directories, such as
    // ".git" (if local).
    "dvcs": "git",

    // The tool to use to create environments.  May be "conda",
    // "virtualenv" or other value depending on the plugins in use.
    // If missing or the empty string, the tool will be automatically
    // determined by looking for tools on the PATH environment
    // variable.
    "environment_type": "virtualenv",

    // the base URL to show a commit for the project.
    "show_commit_url": "https://github.com/dipy/dipy/commit/",

    // The Pythons you'd like to test against.  If not provided, defaults
    // to the current version of Python used to run `asv`.
    // "pythons": ["3.9"],

    // The matrix of dependencies to test.  Each key is the name of a
    // package (in PyPI) and the values are version numbers.  An empty
    // list indicates to just test against the default (latest)
    // version.
    "matrix": {
        "Cython": [],
        "setuptools": ["59.2.0"],
        "packaging": []
    },

    // The directory (relative to the current directory) that benchmarks are
    // stored in.  If not provided, defaults to "benchmarks"
    "benchmark_dir": "benchmarks",

    // The directory (relative to the current directory) to cache the Python
    // environments in.  If not provided, defaults to "env"
    // NOTE: changes dir name will requires update `generate_asv_config()` in
    // runtests.py
    "env_dir": "env",


    // The directory (relative to the current directory) that raw benchmark
    // results are stored in.  If not provided, defaults to "results".
    "results_dir": "results",

    // The directory (relative to the current directory) that the html tree
    // should be written to.  If not provided, defaults to "html".
    "html_dir": "html",

    // The number of characters to retain in the commit hashes.
    // "hash_length": 8,

    // `asv` will cache wheels of the recent builds in each
    // environment, making them faster to install next time.  This is
    // number of builds to keep, per environment.
    "build_cache_size": 8,

    "build_command" : [
        "python -m build {dipy_build_options}",
        // pip ignores '--global-option' when pep517 is enabled, we also enabling pip verbose to
        // be reached from asv `--verbose` so we can verify the build options.
        "PIP_NO_BUILD_ISOLATION=false python {build_dir}/benchmarks/asv_pip_nopep517.py -v {dipy_global_options} --no-deps --no-index -w {build_cache_dir} {build_dir}"
    ],
    // The commits after which the regression search in `asv publish`
    // should start looking for regressions. Dictionary whose keys are
    // regexps matching to benchmark names, and values corresponding to
    // the commit (exclusive) after which to start looking for
    // regressions.  The default is to start from the first commit
    // with results. If the commit is `null`, regression detection is
    // skipped for the matching benchmark.
    //
    // "regressions_first_commits": {
    //    "some_benchmark": "352cdf",  // Consider regressions only after this commit
    //    "another_benchmark": null,   // Skip regression detection altogether
    // }
}