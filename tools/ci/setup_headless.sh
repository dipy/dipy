#!/bin/bash

export DISPLAY=:99.0

if [ "$RUNNER_OS" == "Linux" ]; then
	export LIBGL_ALWAYS_SOFTWARE=1
    export LIBGL_ALWAYS_INDIRECT=0
    sudo apt-get install -y libgl1-mesa-glx xvfb;
	Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1  &
    sleep 3
elif [ "$RUNNER_OS" == "Windows" ]; then
    # Get Opengl Mesa from Vispy. We might need to move it in dipy-data
	export MESA_GL_URL="https://github.com/vispy/demo-data/raw/master/mesa/"
	export ARCHITECTURE=64
	export URL= $MESA_GL_URL + "opengl32_mingw_" + $ARCHITECTURE + ".dll"
	if [ "$ARCHITECTURE" == "32"]; then
        export OPENGL_FILEPATH="C:\Windows\SysWOW64\opengl32.dll"
    else
        export OPENGL_FILEPATH="C:\Windows\system32\opengl32.dll"
    fi

	mv $OPENGL_FILEPATH $OPENGL_FILEPATH.old.bak
	wget -O $OPENGL_FILEPATH $URL


elif [ "$RUNNER_OS" == "macOS" ]; then
	echo 'Install Xquartz package for headless'
	brew install --cask xquartz
fi
