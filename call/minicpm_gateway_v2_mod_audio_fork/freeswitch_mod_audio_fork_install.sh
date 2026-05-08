#!/bin/bash
# mod_audio_fork 安装脚本

FS_SRC="/usr/src/freeswitch"
MOD_URL="https://github.com/thehunmonkgroup/mod_audio_fork.git"

echo "Installing mod_audio_fork..."

cd "$FS_SRC" || exit 1
git clone "$MOD_URL" || exit 1
cd mod_audio_fork || exit 1

make || exit 1
make install || exit 1

echo '<load module="mod_audio_fork"/>' >> /etc/freeswitch/autoload_configs/modules.conf.xml

fs_cli -x "reload mod_audio_fork"

echo "Installation complete. Verify with: fs_cli -x 'module_exists mod_audio_fork'"
