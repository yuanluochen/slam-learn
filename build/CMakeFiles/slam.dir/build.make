# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

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

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yuanluochen/code/slam

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yuanluochen/code/slam/build

# Include any dependencies generated for this target.
include CMakeFiles/slam.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/slam.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/slam.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/slam.dir/flags.make

CMakeFiles/slam.dir/eigen.cpp.o: CMakeFiles/slam.dir/flags.make
CMakeFiles/slam.dir/eigen.cpp.o: ../eigen.cpp
CMakeFiles/slam.dir/eigen.cpp.o: CMakeFiles/slam.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yuanluochen/code/slam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/slam.dir/eigen.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/slam.dir/eigen.cpp.o -MF CMakeFiles/slam.dir/eigen.cpp.o.d -o CMakeFiles/slam.dir/eigen.cpp.o -c /home/yuanluochen/code/slam/eigen.cpp

CMakeFiles/slam.dir/eigen.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/slam.dir/eigen.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yuanluochen/code/slam/eigen.cpp > CMakeFiles/slam.dir/eigen.cpp.i

CMakeFiles/slam.dir/eigen.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/slam.dir/eigen.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yuanluochen/code/slam/eigen.cpp -o CMakeFiles/slam.dir/eigen.cpp.s

# Object files for target slam
slam_OBJECTS = \
"CMakeFiles/slam.dir/eigen.cpp.o"

# External object files for target slam
slam_EXTERNAL_OBJECTS =

slam: CMakeFiles/slam.dir/eigen.cpp.o
slam: CMakeFiles/slam.dir/build.make
slam: CMakeFiles/slam.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yuanluochen/code/slam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable slam"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/slam.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/slam.dir/build: slam
.PHONY : CMakeFiles/slam.dir/build

CMakeFiles/slam.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/slam.dir/cmake_clean.cmake
.PHONY : CMakeFiles/slam.dir/clean

CMakeFiles/slam.dir/depend:
	cd /home/yuanluochen/code/slam/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yuanluochen/code/slam /home/yuanluochen/code/slam /home/yuanluochen/code/slam/build /home/yuanluochen/code/slam/build /home/yuanluochen/code/slam/build/CMakeFiles/slam.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/slam.dir/depend

