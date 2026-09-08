@echo off
setlocal

set "MNN_SOURCE=C:\work\NPU_model\MNN-master\MNN-master"
set "VSDEVCMD=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat"
set "CMAKE=C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\native\build-tools\cmake\bin\cmake.exe"

call "%VSDEVCMD%" -arch=x64 -host_arch=x64 -no_logo || exit /b 1
"%CMAKE%" -S "%MNN_SOURCE%" -B "%MNN_SOURCE%\build_converter_win" -G Ninja ^
  -DMNN_BUILD_CONVERTER=ON ^
  -DMNN_BUILD_SHARED_LIBS=OFF ^
  -DMNN_BUILD_TOOLS=OFF ^
  -DMNN_BUILD_QUANTOOLS=OFF ^
  -DMNN_WIN_RUNTIME_MT=ON ^
  -DCMAKE_BUILD_TYPE=Release || exit /b 1
"%CMAKE%" --build "%MNN_SOURCE%\build_converter_win" --target MNNConvert -j 8 || exit /b 1

echo MNNConvert ready: %MNN_SOURCE%\build_converter_win\MNNConvert.exe
