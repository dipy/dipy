#!/bin/bash

echo "DISPLAY=:99.0" >> $GITHUB_ENV

if [ "$RUNNER_OS" == "Linux" ]; then
	sudo apt-get update -y -qq;
  sudo apt-get install --no-install-recommends -y libegl1-mesa-dev libglu1-mesa-dev libgl1-mesa-dri \
	libxcb-xfixes0-dev mesa-vulkan-drivers xvfb libxcb-cursor0 libx11-dev libxrandr-dev libxinerama-dev \
	libxcursor-dev libxi-dev libgl1-mesa-dev libglu1-mesa-dev x11proto-core-dev \
	libxcb1-dev libx11-xcb-dev libxkbcommon-dev libxcb-keysyms1 libxcb-image0 libxcb-render-util0 \
	libxcb-shape0 libxcb-randr0 libxcb-icccm4 libxcb-xinerama0 libxkbcommon-x11-0;
	Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1  &
    sleep 3
elif [ "$RUNNER_OS" == "Windows" ]; then
	powershell ./tools/ci/install_opengl.ps1
elif [ "$RUNNER_OS" == "macOS" ]; then
	echo 'Install Xquartz package for headless'
	brew install --cask xquartz
fi
