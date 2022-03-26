# CMAKE generated file: DO NOT EDIT!
# Generated by "NMake Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE
NULL=nul
!ENDIF
SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\JetBrains\CLion 2021.1.2\bin\cmake\win\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\JetBrains\CLion 2021.1.2\bin\cmake\win\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\sasha\CLionProjects\cg_2_2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\sasha\CLionProjects\cg_2_2\cmake-build-debug-visual-studio-cuda

# Include any dependencies generated for this target.
include CMakeFiles\cg_2_2.dir\depend.make

# Include the progress variables for this target.
include CMakeFiles\cg_2_2.dir\progress.make

# Include the compile flags for this target's objects.
include CMakeFiles\cg_2_2.dir\flags.make

CMakeFiles\cg_2_2.dir\main.cu.obj: CMakeFiles\cg_2_2.dir\flags.make
CMakeFiles\cg_2_2.dir\main.cu.obj: ..\main.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\sasha\CLionProjects\cg_2_2\cmake-build-debug-visual-studio-cuda\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/cg_2_2.dir/main.cu.obj"
	"E:\CUDA development\bin\nvcc.exe" -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -dc C:\Users\sasha\CLionProjects\cg_2_2\main.cu -o CMakeFiles\cg_2_2.dir\main.cu.obj -Xcompiler=-FdCMakeFiles\cg_2_2.dir\,-FS

CMakeFiles\cg_2_2.dir\main.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cg_2_2.dir/main.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles\cg_2_2.dir\main.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cg_2_2.dir/main.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target cg_2_2
cg_2_2_OBJECTS = \
"CMakeFiles\cg_2_2.dir\main.cu.obj"

# External object files for target cg_2_2
cg_2_2_EXTERNAL_OBJECTS =

CMakeFiles\cg_2_2.dir\cmake_device_link.obj: CMakeFiles\cg_2_2.dir\main.cu.obj
CMakeFiles\cg_2_2.dir\cmake_device_link.obj: CMakeFiles\cg_2_2.dir\build.make
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\sasha\CLionProjects\cg_2_2\cmake-build-debug-visual-studio-cuda\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles\cg_2_2.dir\cmake_device_link.obj"
	"E:\CUDA development\bin\nvcc.exe" -forward-unknown-to-host-compiler -D_WINDOWS -Xcompiler=" /GR /EHsc" -Xcompiler="-Zi -Ob0 -Od /RTC1" --generate-code=arch=compute_52,code=[compute_52,sm_52] -Xcompiler=-MDd -Wno-deprecated-gpu-targets -shared -dlink $(cg_2_2_OBJECTS) $(cg_2_2_EXTERNAL_OBJECTS) -o CMakeFiles\cg_2_2.dir\cmake_device_link.obj  cudadevrt.lib cudart_static.lib kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib  -Xcompiler=-FdC:\Users\sasha\CLionProjects\cg_2_2\cmake-build-debug-visual-studio-cuda\CMakeFiles\cg_2_2.dir\,-FS

# Rule to build all files generated by this target.
CMakeFiles\cg_2_2.dir\build: CMakeFiles\cg_2_2.dir\cmake_device_link.obj

.PHONY : CMakeFiles\cg_2_2.dir\build

# Object files for target cg_2_2
cg_2_2_OBJECTS = \
"CMakeFiles\cg_2_2.dir\main.cu.obj"

# External object files for target cg_2_2
cg_2_2_EXTERNAL_OBJECTS =

cg_2_2.exe: CMakeFiles\cg_2_2.dir\main.cu.obj
cg_2_2.exe: CMakeFiles\cg_2_2.dir\build.make
cg_2_2.exe: CMakeFiles\cg_2_2.dir\cmake_device_link.obj
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\sasha\CLionProjects\cg_2_2\cmake-build-debug-visual-studio-cuda\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable cg_2_2.exe"
	"C:\Program Files\JetBrains\CLion 2021.1.2\bin\cmake\win\bin\cmake.exe" -E vs_link_exe --intdir=CMakeFiles\cg_2_2.dir --rc=C:\PROGRA~2\WI3CF2~1\10\bin\100183~1.0\x64\rc.exe --mt=C:\PROGRA~2\WI3CF2~1\10\bin\100183~1.0\x64\mt.exe --manifests -- C:\PROGRA~2\MICROS~2\2019\COMMUN~1\VC\Tools\MSVC\1427~1.291\bin\Hostx64\x64\link.exe /nologo $(cg_2_2_OBJECTS) $(cg_2_2_EXTERNAL_OBJECTS) CMakeFiles\cg_2_2.dir\cmake_device_link.obj @<<
 /out:cg_2_2.exe /implib:cg_2_2.lib /pdb:C:\Users\sasha\CLionProjects\cg_2_2\cmake-build-debug-visual-studio-cuda\cg_2_2.pdb /version:0.0 /debug /INCREMENTAL /subsystem:console  cudadevrt.lib cudart_static.lib kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib  -LIBPATH:"E:/CUDA development/lib/x64" 
<<

# Rule to build all files generated by this target.
CMakeFiles\cg_2_2.dir\build: cg_2_2.exe

.PHONY : CMakeFiles\cg_2_2.dir\build

CMakeFiles\cg_2_2.dir\clean:
	$(CMAKE_COMMAND) -P CMakeFiles\cg_2_2.dir\cmake_clean.cmake
.PHONY : CMakeFiles\cg_2_2.dir\clean

CMakeFiles\cg_2_2.dir\depend:
	$(CMAKE_COMMAND) -E cmake_depends "NMake Makefiles" C:\Users\sasha\CLionProjects\cg_2_2 C:\Users\sasha\CLionProjects\cg_2_2 C:\Users\sasha\CLionProjects\cg_2_2\cmake-build-debug-visual-studio-cuda C:\Users\sasha\CLionProjects\cg_2_2\cmake-build-debug-visual-studio-cuda C:\Users\sasha\CLionProjects\cg_2_2\cmake-build-debug-visual-studio-cuda\CMakeFiles\cg_2_2.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles\cg_2_2.dir\depend

