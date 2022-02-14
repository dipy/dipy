#!/bin/bash

export DISPLAY=:99.0

if [ "$RUNNER_OS" == "Linux" ]; then
	export LIBGL_ALWAYS_SOFTWARE=1
    export LIBGL_ALWAYS_INDIRECT=0
    sudo apt-get install -y libgl1-mesa-glx xvfb;
	Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1  &
    sleep 3
elif [ "$RUNNER_OS" == "Windows" ]; then
	powershell ./tools/ci/install_opengl.ps1
elif [ "$RUNNER_OS" == "macOS" ]; then
	echo 'Install Xquartz package for headless'
	brew install --cask xquartz
fi
