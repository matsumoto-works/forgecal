@echo off
REM ============================================================================
REM Build the standalone "ForgeCal Post" EXE (run from the PlasticFEM_v4 root).
REM Output: dist\ForgeCalPost\ForgeCalPost.exe  (a folder you can zip & ship)
REM Dev/test stays "streamlit run app\post_app.py"; only build the EXE to release.
REM
REM NOTE: do NOT put REM comments between the ^-continued lines below -- in a
REM batch file a REM on a continued line eats the trailing ^ and breaks the
REM line continuation (PyInstaller then sees "REM" as the script and the rest
REM as unknown commands). Keep all notes up here instead.
REM
REM The --add-binary DLLs are anaconda Library\bin runtimes that PyInstaller's
REM hooks miss in the conda layout, each fixing a "DLL load failed" at runtime:
REM   libifcoremd.dll                    -> scipy _arpack (Intel Fortran runtime)
REM   hdf5.dll / hdf5_hl.dll             -> h5py _errors  (HDF5 runtime)
REM   tiff/libjpeg/openjp2/lcms2/freetype/libwebp(+mux,demux)/Lerc/deflate/
REM   libpng16/libsharpyuv/zstd          -> Pillow _imaging (matplotlib image I/O)
REM
REM Size-reduction exclusions (not used by post_app.py):
REM   vtk/vtkmodules: 306 MB, botocore(AWS): 97 MB, panel/bokeh: 131 MB,
REM   llvmlite/numba: 70 MB  -> total ~604 MB saved
REM   gmsh (85 MB) is intentionally INCLUDED: tool stress uses mesher.mesh_polygon
REM   and the scipy fallback gives +50~60% error on die/punch vs GMSH.
REM Run this script from the PlasticFEM_v4 root with the conda base env active:
REM   conda activate base
REM   cd /d path\to\PlasticFEM_v4
REM   build_post_exe.bat
REM
REM %CONDA_PREFIX% is set automatically by conda activate.
REM If python is not found, run: conda activate base
REM ============================================================================
set PY=python
set BIN=%CONDA_PREFIX%\Library\bin

%PY% -m PyInstaller --noconfirm --clean --onedir --name ForgeCalPost ^
  --collect-all streamlit ^
  --collect-all plotly ^
  --collect-all altair ^
  --collect-all scipy ^
  --collect-all numpy ^
  --copy-metadata streamlit ^
  --collect-submodules plasticfem ^
  --collect-data plasticfem ^
  --hidden-import pandas ^
  --hidden-import h5py ^
  --hidden-import matplotlib.backends.backend_agg ^
  --hidden-import imageio ^
  --hidden-import imageio.v2 ^
  --hidden-import ezdxf ^
  --add-binary "%BIN%\libifcoremd.dll;." ^
  --add-binary "%BIN%\hdf5.dll;." ^
  --add-binary "%BIN%\hdf5_hl.dll;." ^
  --add-binary "%BIN%\tiff.dll;." ^
  --add-binary "%BIN%\libjpeg.dll;." ^
  --add-binary "%BIN%\openjp2.dll;." ^
  --add-binary "%BIN%\lcms2.dll;." ^
  --add-binary "%BIN%\freetype.dll;." ^
  --add-binary "%BIN%\libwebp.dll;." ^
  --add-binary "%BIN%\libwebpmux.dll;." ^
  --add-binary "%BIN%\libwebpdemux.dll;." ^
  --add-binary "%BIN%\Lerc.dll;." ^
  --add-binary "%BIN%\deflate.dll;." ^
  --add-binary "%BIN%\libpng16.dll;." ^
  --add-binary "%BIN%\libsharpyuv.dll;." ^
  --add-binary "%BIN%\zstd.dll;." ^
  --exclude-module PyQt5 --exclude-module PyQt6 ^
  --exclude-module PySide2 --exclude-module PySide6 ^
  --exclude-module tkinter --exclude-module _tkinter ^
  --exclude-module IPython --exclude-module notebook ^
  --exclude-module vtk --exclude-module vtkmodules ^
  --exclude-module botocore --exclude-module boto3 --exclude-module s3transfer ^
  --exclude-module panel --exclude-module bokeh ^
  --exclude-module llvmlite --exclude-module numba ^
  --add-data "app\post_app.py;." ^
  --paths . ^
  app\run_post.py

echo.
echo Done. EXE at: dist\ForgeCalPost\ForgeCalPost.exe
