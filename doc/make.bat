@ECHO OFF

REM Command file for Sphinx documentation

set PYTHON=python
set SPHINXBUILD=sphinx-build
set ALLSPHINXOPTS=-d _build/doctrees %SPHINXOPTS% .
if NOT "%PAPER%" == "" (
	set ALLSPHINXOPTS=-D latex_paper_size=%PAPER% %ALLSPHINXOPTS%
)

if "%1" == "" goto help

if "%1" == "help" (
	:help
	echo.Please use `make ^<target^>` where ^<target^> is one of
	echo.  html      to make standalone HTML files
	echo.  dirhtml   to make HTML files named index.html in directories
	echo.  pickle    to make pickle files
	echo.  json      to make JSON files
	echo.  htmlhelp  to make HTML files and a HTML help project
	echo.  qthelp    to make HTML files and a qthelp project
	echo.  latex     to make LaTeX files, you can set PAPER=a4 or PAPER=letter
	echo.  changes   to make an overview over all changed/added/deprecated items
	echo.  linkcheck to check all external links for integrity
	echo.  doctest   to run all doctests embedded in the documentation if enabled
	goto end
)

if "%1" == "clean" (
	for /d %%i in (_build\*) do rmdir /q /s %%i
	del /q /s _build\*
	call :api-clean
	call :examples-clean
	goto end
)

if "%1" == "api-clean" (
    :api-clean
	del /q /s reference reference_cmd
	rmdir reference reference_cmd
	exit /B
	)

if "%1" == "api" (
    :api
	if not exist reference mkdir reference
	%PYTHON% tools/build_modref_templates.py dipy reference
	if not exist reference_cmd mkdir reference_cmd
	%PYTHON% tools/docgen_cmd.py dipy reference_cmd
	echo.Build API docs...done.
	exit /B
	)

if "%1" == "examples-clean" (
    :examples-clean
	cd examples_built && del /q /s *.py *.rst *.png fig
	cd ..
	exit /B
	)

if "%1" == "examples-clean-tgz" (
    call :examples-clean
    call :examples-tgz %*
	%PYTHON% ../tools/pack_examples.py ../dist
	exit /B
	)

if "%1" == "examples-tgz" (
    :examples-tgz
    call :rstexamples %*
	%PYTHON% ../tools/pack_examples.py ../dist
	exit /B
	)

if "%1" == "gitwash-update" (
	%PYTHON% ../tools/gitwash_dumper.py devel dipy --repo-name=dipy ^
	                                               --github-user=dipy ^
	                                               --project-url=https://dipy.org ^
	                                               --project-ml-url=https://mail.python.org/mailman/listinfo/neuroimaging
    )

if "%1" == "rstexamples" (
    :rstexamples
	cd examples_built && %PYTHON% ..\\..\\tools\\make_examples.py
	type nul > %*
	cd ..
	exit /B
	)

if "%1" == "html" (
    :html
    echo "build full docs including examples"
    call :api
    call :rstexamples %*
    call :html-after-examples
    exit /B
	)

if "%1" == "html-after-examples" (
    :html-after-examples
	%SPHINXBUILD% -b html %ALLSPHINXOPTS% _build/html
	echo.
	echo.Build finished. The HTML pages are in _build/html.
	exit /B
    )

if "%1" == "dirhtml" (
	%SPHINXBUILD% -b dirhtml %ALLSPHINXOPTS% _build/dirhtml
	echo.
	echo.Build finished. The HTML pages are in _build/dirhtml.
	goto end
)

if "%1" == "pickle" (
	%SPHINXBUILD% -b pickle %ALLSPHINXOPTS% _build/pickle
	echo.
	echo.Build finished; now you can process the pickle files.
	goto end
)

if "%1" == "json" (
	%SPHINXBUILD% -b json %ALLSPHINXOPTS% _build/json
	echo.
	echo.Build finished; now you can process the JSON files.
	goto end
)

if "%1" == "htmlhelp" (
	%SPHINXBUILD% -b htmlhelp %ALLSPHINXOPTS% _build/htmlhelp
	echo.
	echo.Build finished; now you can run HTML Help Workshop with the ^
.hhp project file in _build/htmlhelp.
	goto end
)

if "%1" == "qthelp" (
	%SPHINXBUILD% -b qthelp %ALLSPHINXOPTS% _build/qthelp
	echo.
	echo.Build finished; now you can run "qcollectiongenerator" with the ^
.qhcp project file in _build/qthelp, like this:
	echo.^> qcollectiongenerator _build\qthelp\dipy.qhcp
	echo.To view the help file:
	echo.^> assistant -collectionFile _build\qthelp\dipy.ghc
	goto end
)

if "%1" == "latex" (
    :latex
    call :rstexamples %*
    call :latex-after-examples
    exit /B
)

if "%1" == "latex-after-examples" (
    :latex-after-examples
	%SPHINXBUILD% -b latex %ALLSPHINXOPTS% _build/latex
	echo.
	echo.Build finished; the LaTeX files are in _build/latex.
	echo.Run make all-pdf or make all-ps in that directory to ^
	     run these through (pdf)latex.
	exit /B
)

if "%1" == "changes" (
	%SPHINXBUILD% -b changes %ALLSPHINXOPTS% _build/changes
	echo.
	echo.The overview file is in _build/changes.
	goto end
)

if "%1" == "linkcheck" (
	%SPHINXBUILD% -b linkcheck %ALLSPHINXOPTS% _build/linkcheck
	echo.
	echo.Link check complete; look for any errors in the above output ^
or in _build/linkcheck/output.txt.
	goto end
)

if "%1" == "doctest" (
	%SPHINXBUILD% -b doctest %ALLSPHINXOPTS% _build/doctest
	echo.
	echo.Testing of doctests in the sources finished, look at the ^
results in _build/doctest/output.txt.
	goto end
)

if "%1" == "pdf" (
    call :latex
	cd _build/latex && make all-pdf
	type nul > %*
	goto end
	)

if "%1" == "upload" (
    call :html %*
	./upload-gh-pages.sh _build/html/ dipy dipy
	goto end
	)

if "%1" == "xvfb" (
	set TEST_WITH_XVFB=true && make html
	goto end
	)

if "%1" == "memory_profile" (
	set TEST_WITH_MEMPROF=true && make html
	goto end
	)

:end
